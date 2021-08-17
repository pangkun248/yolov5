import json
import sys
from pathlib import Path
from utils.datasets import letterbox
import cv2
import numpy as np
import torch
from tqdm import tqdm
from utils.deploy.deploy_utils import process_np
from val import process_batch, save_one_json

from utils.deploy.deploy_utils import coco_list
from utils.general import coco80_to_coco91_class, check_requirements, non_max_suppression, scale_coords, xywh2xyxy
from utils.metrics import ap_per_class

"""
 转onnx之前需要修改yolo.py,sigmoid后直接放进一个list返回
"""


def calc_ap(onnx_path, val_path, anno_json, cls_list):
    net = cv2.dnn.readNetFromONNX(onnx_path)
    nc = len(cls_list)  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    names = dict(zip(range(nc), cls_list))
    class_map = coco80_to_coco91_class()
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []

    img_list = open(val_path, 'r').readlines()
    for img_path in tqdm(img_list, desc=s):
        label_path = img_path.replace('images', 'labels').rsplit('.', maxsplit=1)[0] + '.txt'
        targets = np.fromfile(label_path, sep=' ').reshape(-1, 5)  # add bs dim
        img0 = cv2.imread(img_path.rstrip())
        height, width, _ = img0.shape
        img = letterbox(img0, new_shape=(640, 640), auto=False, stride=32)[0]  # 静态输入时auto=False 0.1~0.2%mAP↑  0~1ms↑
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, hwc -> chw
        img = np.ascontiguousarray(img)[None]  # add bs dim
        img = img.astype('float32')
        # Run inference
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # trt run
        net.setInput(img)
        outputs = net.forward(net.getUnconnectedOutLayersNames())  # Inference

        outputs_ = []
        for i, p in enumerate(outputs):
            p[..., :2] = (p[..., :2] * 2 - 0.5 + grid[i]) * stride[i]
            p[..., 2:4] = (p[..., 2:4] * 2) ** 2 * anchor_grid[i]
            outputs_.append(p.reshape(bs, -1, num_cls + 5))
        outputs = np.concatenate(outputs_, 1)

        # out = process_np(buffer, 0.001, 0.6)
        targets[:, 1:] *= [width, height, width, height]  # to pixels
        # 此时out(经过了nms)是在标准图片尺寸下(与target一致) x1 y1 x2 y2 conf cls_ind [[n,6],*bs]
        out = non_max_suppression(torch.from_numpy(outputs), 0.001, 0.6, multi_label=True)
        # Statistics per image 每张图片的统计数据
        for si, pred in enumerate(out):
            labels = torch.from_numpy(targets)  # target及labels是在标准尺寸下的,x y w h
            nl = len(labels)  # 该张图片的label数量
            tcls = labels[:, 0].tolist() if nl else []  # target class 该张图片的类list
            path, shape = Path(img_path), (height, width)  # 该张图片的路径
            seen += 1  # 已处理的图片数量
            if len(pred) == 0:
                if nl:  # 有label,但一个都没检测到
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()  # c,h,w   x1 y1 x2 y2  (h0, w0),      ((h / h0, w / w0), pad) clone不共享内存,但共享梯度
            scale_coords(img.shape[2:], predn[:, :4], img0.shape)  # 去padding-> 放缩回[w0,h0]-> clip
            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # 将target_box -> x y x y -> padding剥离 -> 放缩回[w0,h0] -> clip
                # scale_coords(img[si].shape[1:], tbox, shape, ratio_pad)  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # 这里tbox都是未经改变的所以不用处理
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            # 将每张图片的统计信息整合起来作为一个list添加进stats (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...]
            save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            # image end-------------------------------------------------------------------------------------------------
        # batch end-----------------------------------------------------------------------------------------------------
    # 根据这些统计信息计算相关性能指标 [pn,10] [pn,] [pn,] [tn] pn->测试集所有的检测结果的总数 tn->测试集所有的标注目标的总数
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():  # 如果IoU矩阵不为空(至少有一个预测目标),且存在一个与target_box的IoU值大于0.5的
        p, r, ap, f1, ap_class = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # 输出整体结果
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # 输出格式
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # 输出每类结果(验证集中出现过的类)
    for i, c in enumerate(ap_class):  # c是指该类在检测类别中的索引. 而i是指验证集类别索引,因为 p、r ap、f1都是基于此的
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    pred_json = str(f"dnn_predictions.json")  # predictions json
    print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)

    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        check_requirements(['pycocotools'])
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
    except Exception as e:
        print(f'pycocotools unable to run: {e}')


if __name__ == "__main__":
    bs, in_h, in_w, num_cls = 1, 640, 640, 80  # 该输入参数也必须固定
    grid = []
    anchor_grid = [[10, 13, 16, 30, 33, 23],
                   [30, 61, 62, 45, 59, 119],
                   [116, 90, 156, 198, 373, 326]]  # 该anchor必须与dnn模块读取的yolo模型保持一致
    stride = [8, 16, 32]  # stride同上
    for i, s in enumerate(stride):
        nx, ny = in_w // s, in_h // s
        xx, yy = np.meshgrid(np.arange(ny), np.arange(nx))
        grid.append(np.stack((xx, yy), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32))
        anchor_grid[i] = np.array(anchor_grid[i]).reshape((1, 3, 1, 1, 2))
    onnx_path = "/home/cmv/PycharmProjects/yolov5/yolov5s.onnx"
    val_txt = '/home/cmv/PycharmProjects/datasets/coco/val2017.txt'
    val_json = '/home/cmv/PycharmProjects/datasets/coco/annotations/instances_val2017.json'
    calc_ap(onnx_path, val_txt, val_json, coco_list)
