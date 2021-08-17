# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    # Model fitness as a weighted combination of metrics 将模型的各个性能指标加权求和,并作为模型整体优化方向
    w = [0.0, 0.0, 0.1, 0.9]  # [P, R, mAP@0.5, mAP@0.5:0.95]中各个指标的贡献度(权重)
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # 根据置信度排序 加个负号代表从大到小 这里是把测试集所有检测结果都揉到一起了,以图片顺序来排序,图片内以conf大小排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]  # tp conf pred_cls 根据conf从大到小进行重排序

    # 获取验证集中出现的所有类别
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # 验证集中出现的类别总数

    # 创建P-R曲线,并计算每类的AP
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):  # 按类计算P R AP等信息
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # target_box中 类别c 的个数
        n_p = i.sum()  # pred_box中为 类别c 的个数

        if n_p == 0 or n_l == 0:  # 如果没有合格的检测结果或标注物体 则跳过类别c 实际n_l不可能为0
            continue
        else:
            # 计算 FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall = TP / (TP + FN) = TP / num_target
            recall = tpc / (n_l + 1e-16)  # 召回曲线 上面的if已经对n_l=0进行处理了,这里还有必要+1e-16?
            # 这里是根据线性插值法获取[0,1]之间的多个conf值对应的recall值,至于为什么加负号,因为x与xp参数要求必须↑,而conf是↓,所以-conf
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases  ↓

            # Precision = TP / (TP + FP) = TP / num_pred
            precision = tpc / (tpc + fpc)  # 精确曲线 同上
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score  ↑

            # 根据P-R曲线计算AP(各个IoU阈值下的AP)
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:  # 仅绘出IoU阈值=0.5下的平均Precision与平均Recall
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5 通过线性差值的方法得到1000个[0,1]之间对应的p

    #  F1_score (P、R的调和平均值) P、R曲线是以conf与P、R为基准通过线性插值生成出的1000个点,可以说这1000个点来代表了conf与P、R的关系
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index 相对于[0,999]  最后返回F1_score最大处的点对应的 P、R、F1等值 AP除外
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ 根据P-R曲线计算AP
    Arguments
        recall:    (m,)  numpy型数据   ↑
        precision: (m,)  numpy型数据   ↓
    Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope 参考 https://www.it1352.com/240349.html
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # 计算曲线下的面积
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # 积分 https://www.osgeo.cn/numpy/reference/generated/numpy.trapz.html
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes 因为改变是相对值,排除第一个值,所以[1:]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve  因为i是基于[:1],所以要+1

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])  # 注! 这里没有在统一class下计算IoU,表现为在混淆矩阵上非对角线位置会有值

        x = torch.where(iou > self.iou_thres)  # 与 torch.nonzero(condition, as_tuple=True) 结果相同
        if x[0].shape[0]:  # 对matches重新整理 -> [c,3] iou_index(2) iou_value(1) c为满足iou_thresh值的c个iou
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                """
                1.根据IoU值对matches排序(从大到小)
                2.获取每个pred_box中最大IoU的matches
                3.根据IoU值matches重新排序(从大到小)
                4.获取每个gt_box中最大的IoU,这样得到的IoU就是gt_box与pred_box一对一且是最大IoU(对双方来说)
                这样做的目的是为了获取多个pred_box与多个gt_box中最大IoU的那些matches 此时matches -> [v,3] v<=c(大概率v<<c)
                """
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:  # 即所有iou都不满足iou_thresh要求
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)  # 此处m0,m1分别指matches在gt_boxes、pre_boxes中的索引
        for i, gc in enumerate(gt_classes):  # 针对所有的标记结果,只有TP与FP 总感觉应该和下面的倒过来(-> TP+FN)
            j = m0 == i  # 该gt_box是否在matches(TP)中
            if n and sum(j) == 1:  # 改成j.any() 是否会更合适?
                self.matrix[detection_classes[m1[j]], gc] += 1  # 注这里并非是TP,只是经过nms的与target有最佳匹配IoU的box(忽略了类一致性)
            else:
                self.matrix[self.nc, gc] += 1  # background FP   这里将background也视作一类,预测为bg但实际是gc

        if n:
            for i, dc in enumerate(detection_classes):  # 针对所有的预测结果,只有TP与FN 总感觉应该和下面的倒过来(-> TP+FP)
                if not any(m1 == i):  # 该pre_box是否在matches(TP)中
                    self.matrix[dc, self.nc] += 1  # background FN  预测为dc,但实际为bg

    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)  # tight_layout 自动调整图形元素,使其恰当的显示
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T  # 这里box1与box2都是4*n,4*m的格式

    # 将box转为统一格式
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # x y w h -> x1 y1 x2 y2
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # px 是[0,1]的1000个x坐标 py是对应的precision值(IoU=0.5时) [(与px同shape),*num_cls]
    # 上面的num_cls是理想状态,是在所有的识别类别都在测试集与预测结果中出现的情况
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)  # stack后 py.shape -> [1000,num_cls]

    if 0 < len(names) < 21:  # 如果类数量低于21,则显示各个类名称
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # 单独绘制每个类的P-曲线 以及每个类的AP@0.5
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # 一口气绘制多个类的P-曲线

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # 参考 http://www.deiniu.com/article/186659.htm
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
