import numpy as np
import cv2

from utils.datasets import letterbox
from utils.general import scale_coords
from utils.deploy.deploy_utils import process_np
from utils.deploy.deploy_utils import coco_list

"""
 转onnx之前需要修改yolo.py,sigmoid后直接放进一个list返回
"""


def detect_dnn(onnx_path, img_path):
    net = cv2.dnn.readNetFromONNX(onnx_path)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    img0 = cv2.imread(img_path)
    img = letterbox(img0, new_shape=(640, 640), auto=False, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, hwc -> chw
    img = np.ascontiguousarray(img)[None]  # add bs dim
    img = img.astype('float32')
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # blob = cv2.dnn.blobFromImage(img0,1 / 255, (640,640), [0, 0, 0], swapRB=True, crop=False)
    net.setInput(img)
    outputs = net.forward(net.getUnconnectedOutLayersNames())  # Inference
    pred = []
    for i, p in enumerate(outputs):
        p[..., :2] = (p[..., :2] * 2 - 0.5 + grid[i]) * stride[i]
        p[..., 2:4] = (p[..., 2:4] * 2) ** 2 * anchor_grid[i]
        pred.append(p.reshape(bs, -1, num_cls + 5))
    pred = np.concatenate(pred, 1)
    # Apply NMS
    pred = process_np(pred, 0.25, 0.45)
    for det in pred:
        # 处理单张图片检测结果
        print(det)
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # save results on image
            for x1, y1, x2, y2, conf, cls in det:
                cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (2, 128, 128), 5, cv2.LINE_AA)
                cv2.putText(img0, coco_list[int(cls)], (int(x1), int(y1) + 27), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (2, 25, 25))
            # cv2.imwrite(path.replace('123', 'mm'), img0)
            # cv2.namedWindow('result', 0)
            # cv2.resizeWindow('result', (img0.shape[1]//5, img0.shape[0]//5))  # winname, width, height
            cv2.imshow('result', img0)
            cv2.waitKey(0)


if __name__ == '__main__':
    bs, in_h, in_w, num_cls = 1, 640, 640, 80      # 该输入参数也必须固定
    grid = []
    anchor_grid = [[10, 13, 16, 30, 33, 23],
                   [30, 61, 62, 45, 59, 119],
                   [116, 90, 156, 198, 373, 326]]  # 该anchor必须与dnn模块读取的yolo模型保持一致
    stride = [8, 16, 32]                           # stride同上
    for i, s in enumerate(stride):
        nx, ny = in_w // s, in_h // s
        xx, yy = np.meshgrid(np.arange(ny), np.arange(nx))
        grid.append(np.stack((xx, yy), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32))
        anchor_grid[i] = np.array(anchor_grid[i]).reshape((1, 3, 1, 1, 2))
    onnx_file = '/home/cmv/PycharmProjects/yolov5/yolov5s.onnx'
    # onnx_file = '/home/cmv/PycharmProjects/yolov5/runs/train/exp66/weights/best.onnx'
    img_file = '/home/cmv/PycharmProjects/yolov5/data/images/zidane.jpg'
    detect_dnn(onnx_file, img_file)
