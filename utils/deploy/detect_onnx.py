import numpy as np
import cv2
import onnxruntime

from utils.datasets import letterbox
from utils.general import scale_coords
from deploy_utils import process_np
from deploy_utils import coco_list


def detect_onnx(onnx_path,img_path):
    session = onnxruntime.InferenceSession(onnx_path)
    print("ONNX模型期望输入尺寸为:", session.get_inputs()[0].shape)  # 非数字代表是动态范围
    img0 = cv2.imread(img_path)
    img = letterbox(img0, new_shape=(640, 640), stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, hwc -> chw
    img = np.ascontiguousarray(img)[None]  # add bs dim
    img = img.astype('float32')
    # Run inference
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    pred = session.run(None, {session.get_inputs()[0].name: img})[0]  # Inference
    # (torch.cat(z, 1), x) 其中z是pre_fm先view、permute再与grid与anchor结合, x则是pre_fm简单view与permute处理

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
                cv2.putText(img0, coco_list[int(cls)], (int(x1), int(y1) + 27), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 25, 25))
            # cv2.imwrite(path.replace('123', 'mm'), img0)
            # cv2.namedWindow('result', 0)
            # cv2.resizeWindow('result', (img0.shape[1]//5, img0.shape[0]//5))  # winname, width, height
            cv2.imshow('result', img0)
            cv2.waitKey(0)
if __name__ == '__main__':
    onnx_file = '/home/cmv/PycharmProjects/yolov5/yolov5s.onnx'
    img_file = '/home/cmv/PycharmProjects/yolov5/data/images/zidane.jpg'
    detect_onnx(onnx_file,img_file)