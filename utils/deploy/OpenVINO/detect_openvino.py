from openvino.inference_engine import IECore
import numpy as np
import cv2
import time

from utils.datasets import letterbox
from utils.general import scale_coords
from utils.deploy.deploy_utils import process_np
from utils.deploy.deploy_utils import coco_list


def detect_vino(xml_path, img_path):
    ie = IECore()
    # net = ie.read_network(model=xml_path, weights=xml_path.replace('xml', 'bin'))
    net = ie.read_network(xml_path.replace('xml', 'onnx'))  # 使用ONNX模型进行inference会慢约6ms
    exec_net = ie.load_network(network=net, device_name='CPU')
    input_name = next(iter(net.input_info))  # 'images'
    print("OpenVINO模型期望输入尺寸为:", net.input_info[input_name].input_data.shape)  # 非数字代表是动态范围
    cost = 0
    i = 0
    while 1:
        if i == 1000:
            exit()
        i+=1
        img0 = cv2.imread(img_path)
        img = letterbox(img0, new_shape=(640, 640),auto=False, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, hwc -> chw
        img = np.ascontiguousarray(img)[None]  # add bs dim
        img = img.astype('float32')  # openvino支持FP16格式(INT8需要POT),但速度好像没有提升.
        # Run inference
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        s = time.time()
        pred = exec_net.infer(inputs={input_name: img})['output']  # Inference
        # (torch.cat(z, 1), x) 其中z是pre_fm先view、permute再与grid与anchor结合, x则是pre_fm简单view与permute处理
        # Apply NMS
        pred = process_np(pred, 0.25, 0.45)
        cost += time.time()-s
        print(cost/i)
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
    xml_file = '/home/cmv/res/yolov5s.xml'
    img_file = '/home/cmv/PycharmProjects/yolov5/data/images/zidane.jpg'
    detect_vino(xml_file, img_file)
