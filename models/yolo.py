# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # 各个检测层的下采样倍数 不出意外的话 -> tensor([ 8., 16., 32.])
    onnx_dynamic = False  # ONNX 导出时输入维度是否是动态范围

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # 检测类别数
        self.no = nc + 5  # 每个anchor的输出维度
        self.nl = len(anchors)  # 检测层的数量
        self.na = len(anchors[0]) // 2  # 每个检测层anchor的数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv 普通的1x1降维卷积
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling  [[1, 128, 32, 32], [1, 256, 16, 16], [1, 512, 8, 8]] 以 bs=1, h,w=256为例
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv  降维 -> nc
            bs, _, ny, nx = x[i].shape  # [bs,no,ny,nx] -> [bs,na,no,ny,nx] -> [bs,na,ny,nx,no]
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:  # 当dynamic参数开启时,每个grid都要临时新建
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy [-0.5,1.5]
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 最大四倍anchor宽高,对应hyp中的anchor_t
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    # 这里是由于 OpenVINO TensorRT暂时不支持原地修改的操作,只能另赋值然后cat来代替
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))  # [bs,na,ny,nx,nc] -> [bs,na*ny*nx,no]

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml
            self.yaml_file = Path(cfg).name  # 这个变量貌似没有用处
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # 从 'model.yaml' 加载的配置

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 多变量赋值,如果配置中没有ch属性则默认为3
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"model.yaml中的类别数已更新 更新前:{self.yaml['nc']} 更新后:{nc}")
            self.yaml['nc'] = nc  # 覆盖yaml中的nc值
        if anchors:  # 如果anchors在模型初始化时指定,则更新进model.yaml中时anchors为int(未生成),否则为list(已生成)
            LOGGER.info(f'model.yaml中的anchors已被hyp文件的anchors已更新 更新后anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 用hyp文件中的anchors(int)覆盖yaml中的anchors值 注!原始权重内置的anchors为list
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, 需要与其他层concat的层索引(绝对)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # 暂时以各个类的"索引"为默认名称
        self.inplace = self.yaml.get('inplace', True)  # 是否原地修改操作,仅在Detect中.主要为了兼容 ONNX (因为ONNX暂不支持该操作)
        # LOGGER.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)  # 检查strides与anchors的一致性
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()  # 打印网络参数相关信息
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """
        本质就是把一张图片数据增强(放缩与翻转)后变为三张,并同时送入网络然后将这三张图片结果合并cat返回
        """
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # 各层的输出,每层的耗时
        for m in self.model:
            if m.f != -1:  # 如果当前层的输入不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 要么更早层,要么多层
            if profile:  # 统计相关层的耗时
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run 真正的forward
            y.append(x if m.i in self.save else None)  # 只将那些来自更早的层(from∈int但!=-1,一般来说没有)或会被cat的层的输出保存起来,其余为None

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # 在augmented inference(TTA）下对预测结果(x y w h) 放缩、翻转以使其匹配原始图像
        if self.inplace:
            p[..., :4] /= scale  # 逆-scale
            if flips == 2:  # 这里这个flips 是基于 [bs,na,h,w,nc]维度来翻转的 2对应h,3对应w
                p[..., 1] = img_size[0] - p[..., 1]  # 逆-上下翻转
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # 逆-左右翻转
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # 对不同检测层的降维卷积的bias的obj值加上不同的(负)值 stride[8,16,32] -> [-6.68,-5.29,-3.91]
            # 同时根据每个类出现的频率在bias加上不同的值,频率越高bias值越大.不过默认cf为None
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():  # 注:官方提供的n s m l x 模型中的m只是单纯带权重的模块,比如Detect模块是没有任何属性变量的.但如果你自己训练一个模型是会有这些属性的
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # anchor的数量(每个yolo层)
    no = na * (nc + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # 如果是strings则进行转义
        for j, a in enumerate(args):
            try:  # 这个 try是为了nearest转义之后未定义而报NameError所准备的
                args[j] = eval(a) if isinstance(a, str) else a  # 如果是strings则进行转义 这里也将 "nc"与"anchors"转义了
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # 模块的实际深度
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:
            c1, c2 = ch[f], args[0]  # 输入维度 , 输出维度
            if c2 != no:  # 如果不是最终输出,则根据gw-width_multiple缩放系数来重新设定输出
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # 重复的次数
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # 如果anchors是int则将[[0,1,2,3,4,5]*3]赋值给args[1],后续有check_anchor进行处理
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:  # 将第三第四维度缩小n倍,第二维度扩大n*n倍 view + permute + view的方法
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:  # 将第三第四维度扩大n倍,第二维度缩小n*n倍 同上
            c2 = ch[f] // args[0] ** 2
        else:  # up_sample
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type 注! str(m)中貌似没有__main__,目前不知道这行代码的作用
        n_p = sum(x.numel() for x in m_.parameters())  # number params 参数量
        m_.i, m_.f, m_.type, m_.n_p = i, f, t, n_p  # 模块索引, 输入索引, 模块类型, 模块参数量
        # 输出模型各个模块的相关信息
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n_, n_p, t, args))
        # 将from中除开-1层外的索引转换为(感觉多余,本来就是)绝对索引并按序返回,其实就是方便后续提取出相应索引层的输出以便与其他(-1)层输出concat
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
