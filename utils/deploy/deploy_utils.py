"""
该文件的方法主要是为了ONNX、TensorRT的inference准备的,大部分都是已有方法的numpy版本
"""
import numpy as np
import time
from utils.general import xywh2xyxy


coco_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
             'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
             'hair drier', 'toothbrush']


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def nms_np(boxes, scores, iou_thresh):
    boxes_tem = boxes.copy()
    sort_c = np.argsort(scores)[::-1]
    boxes = boxes[sort_c]
    keep_ind = []
    while boxes.shape[0]:
        large_overlap = box_iou(boxes[:1], boxes)[0] > iou_thresh
        keep_ind += [np.where(boxes_tem == boxes[0])[0][0]]
        boxes = boxes[~large_overlap]
    return np.stack(keep_ind)


def process_np(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=300):
    """对检测结果进行后处理 (numpy版本)
    prediction: [bs,c,nc+5]
    Returns:    [(n,6),*bs]  6 -> [x y x y, conf, cls]
    """
    xc = prediction[..., 4] > conf_thres  # candidates
    # nms的相关配置
    min_wh, max_wh = 2, 4096  # box的最小与最大宽高(px)
    max_nms = 30000  # 进入torchvision.ops.nms()的最大box数量
    time_limit = 10.0  # 每张图片进行nms的时间限制,超出则直接强制退出整个batch的nms

    t = time.time()
    output = [np.zeros(shape=(0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # 图片在batch_size中索引, 图片检测结果
        # 约束box宽高
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # 第一次根据 conf_thresh 过滤box
        # 如果x被conf_thresh全部过滤掉了,那么下一张
        if not x.shape[0]:
            continue
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (x, y, w, h) -> (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # x.shape [n,5+nc] (x y x y, conf, cls*nc)    下面是第二次根据 conf_thresh 过滤box
        conf = x[:, 5:].max(1)
        j = x[:, 5:].argmax(1)
        x = np.concatenate((box, conf[:, None], j[:, None].astype('float32')), 1)[conf > conf_thres]
        # 此时的x变更了数据格式 x.shape [n,6] (x y x y, cls_conf, cls_ind)

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence nms前限制box数量

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 是否在同类之间进行nms 因为agnostic=single_cls,所以取决于是否单类检测
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = nms_np(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections  nms后限制box数量
            i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # 超过单张图片的nms时间限制了

    return output
