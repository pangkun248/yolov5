# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):  # 针对开启label_smooth的情况
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7  各个fm权重,fm越大权重越大
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions (predictions.shape -> [bs,na,gh,gw,c+5], /2 ,/4)
            b, a, gj, gi = indices[i]  # image, anchor, grid_y, grid_x
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets  这个n=原始target repeat na次 再扩充3(<=3)次
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets 每个pi中与target匹配的数据

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5  # [-0.5,1.5]
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 限制与相应anchor的宽高修正系数的范围 (0,4)
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                # https://github.com/ultralytics/yolov5/issues/471#issuecomment-742524530 如果直接置为1,mAP会降
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio 为了帮助nms减少低质量的box

                # Classification 类别置信度
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE 只计算target位置及附近区域上的cls_loss

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)  # 计算该fm尺寸上所有grid的obj_loss
            lobj += obji * self.balance[i]  # obj loss 给与小目标更多的loss权重提升小目标检测能力?
            if self.autobalance:  # 大小物体weight自动平衡机制? 取当前次loss和之前weight结合,obj_loss大->小 小->大
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]  # 以stride=16的fm_weight作为基准?
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """
        p: list型数据 其中元素为nl个yolo层的输出[16, 3, 80/2/4, 80/2/4, 85]  img_size:640*640 bs:16 nc:80
        该方法实际就是先对target repeat na次顺带附上对应的na索引
        再在不同feature map下生成不同的target
            1.将target_box放缩到fm尺寸(0,1)-> (0,fm)
            2.过滤与anchor同边比差异过大的target
            3.选取距target最近的三个grid(包含自身),同时保证grid不超出边界以及target_xy!=0.5等... 此时 target的翻倍数 <= 3
            4.计算该fm尺寸下所有合适target对应的target_off(偏移) 限制grid_xy的grid索引 bs索引 anchor索引啊之类的
            5.将该fm尺寸下的相关数据打包在不同list中
        返回各个相关list len(list)==self.nl
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # target.shape [56, 6] repeat-> [3, 56, 6] +anchor -> torch.Size([3, 56, 7]) 56为一个batch内总target数量
        # 这里target重复na次 因为target要在3种anchor(同一feature map,不同长宽比)上计算loss

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m 左上右下 因为下面off是要被减的,所以"左正右负,上正下负"
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets  乘以偏移值  [5, 2]
        for i in range(self.nl):
            anchors = self.anchors[i]  # 这里的anchor的大小是相对于特征图的尺寸  yolo.py m.anchors /= m.stride.view(-1, 1, 1)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # x y x y gain
            # Match targets to anchors  因为target是相对值.这里把其从(0,1)转换成(0,grid)和上面anchor同一尺寸  b,c,x,y,w,h,a
            t = targets * gain  # 将target信息由(0,1)大小转换成特征图尺寸相对大小
            if nt:
                # Matches  实际就是匹配与相应anchor有合适同边比的那些target
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio  gt_box与anchor同边的倍数比值
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # 最大比值不能超过anchor_t,否则差异过大会被过滤
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # 过滤掉那些与anchor边长比值异常的target  gt_w/an_w > 4 or  gt_w/an_w < 1/4 h同理  shape->[113, 7]

                # Offsets
                gxy = t[:, 2:4]  # grid xy              [113, 2]
                gxi = gain[[2, 3]] - gxy  # inverse
                """
                 j,k是target是否距离左边上边的grid更近,以及是否不在第一行列,因为在第一行列中左边上边的grid超出边界,所以排除
                 l,m是target是否距离右边下边的grid更近,以及是否不在最后一行列,因为在最后一行列中右边下边的grid超出边界,所以排除
                 j,k,l,m分别指该target的中心点坐标是否距离左边界、上边界、右边界、下边界更近.所以只能选出两个来(x=0.5时放弃该点,y同理)
                 这四个值在某一target上一定是两True与两False(x或y=0.5以及"个别"边界target四个False),即stack之后算上自身实际最多三个True
                """
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # [113]
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # [5, 113]
                t = t.repeat((5, 1, 1))[j]  # 先把target复制五份,再选取最近的三个点(包含自身) 筛选后t的个数<=3*t [339, 7]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 偏移值同上  [1, 113, 2] + [5, 1, 2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()  # 减去一个正数就代表离左上边更近,否则离右下边更近,最后取整代表获取grid索引
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices 每个target对应某一尺度下的第几个anchor ([3,3,2]中的第二维度的索引)
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box  这里的gxy - gij实际上是target的偏移值[0,1) (包括其领域的target)
            anch.append(anchors[a])  # anchors (index ie 0,1,2)
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
