"""
显存不足 或者 申请显存大小与trt模型输出维度不一致
PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
cuMemFree failed: an illegal memory access was encountered
"""

import numpy as np
import tensorrt as trt
import cv2
import pycuda.driver as cuda
import pycuda.autoinit  # 这个不能删

TRT_LOGGER = trt.Logger()

from utils.datasets import letterbox
from deploy_utils import process_np
from utils.general import scale_coords
from deploy_utils import coco_list


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def inference_static(context,image_txt,input_size):
    b, c, h, w = context.get_binding_shape(0)
    # 分配输入的内存. image.nbytes=prod(shape)*4 当image为float32类型时 image.dtype.itemsize=4
    d_input = cuda.mem_alloc(b * c * h * w * 4)
    # 0.(bs, 3, 640, 640)  123.(bs, 3, 80/2/4, 80/2/4, 85)  4.(bs, 25200, 85)  img_size:640*640
    output_shape = context.get_binding_shape(1)
    # 创建输出占位符
    buffer = np.empty(output_shape, dtype=np.float32)
    # 分配输出内存
    d_output = cuda.mem_alloc(buffer.nbytes)
    bindings = [d_input, d_output]
    img_paths = open(image_txt,'r').readlines()
    for img_path in img_paths:
        img0 = cv2.imread(img_path.rstrip())
        img = letterbox(img0, new_shape=(h, w), stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, hwc -> chw
        img = np.ascontiguousarray(img)[None]  # add bs dim
        img = img.astype('float32')
        img /= 255.0
        # img:cpu->gpu
        cuda.memcpy_htod_async(d_input, img)
        # Forward
        context.execute_v2(bindings)
        # output:gpu->cpu 只需(bs, 25200, 85)即可
        cuda.memcpy_dtoh(buffer, d_output)
        pred = process_np(buffer, 0.25, 0.45)  # default
        for det in pred:
            # 处理单张图片检测结果
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # save results on image
                for x1, y1, x2, y2, conf, cls in det:
                    cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (2, 128, 128), 5, cv2.LINE_AA)
                    cv2.putText(img0, coco_list[int(cls)], (int(x1), int(y1) + 27), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 25, 25))
                # cv2.namedWindow('result', 0)
                # cv2.resizeWindow('result', (img0.shape[1]//5, img0.shape[0]//5))  # winname, width, height
                cv2.imshow('result', img0)
                cv2.waitKey(0)


def inference_dynamic(context,image_txt,input_size):
    img_paths = open(image_txt,'r').readlines()
    for img_path in img_paths:
        img0 = cv2.imread(img_path.rstrip())
        img = letterbox(img0, new_shape=input_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, hwc -> chw
        img = np.ascontiguousarray(img)[None]  # add bs dim
        img = img.astype('float32')
        # Run inference
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # img = np.repeat(img,2,0)
        # 设置输入维度 img_in 代表输入数据,(1, 3, 416, 416) 或 (2, 3, 320, 416)
        context.set_binding_shape(0, img.shape)
        # 分配输入的内存. nbytes=prod(shape)*4
        d_input = cuda.mem_alloc(img.nbytes)
        output_shape = context.get_binding_shape(1)
        # 创建三个输出占位符
        buffer = np.empty(output_shape, dtype=np.float32)
        # 分配输出内存
        d_output = cuda.mem_alloc(buffer.nbytes)
        # 将输入数据拷贝到GPU
        cuda.memcpy_htod(d_input, img)
        bindings = [d_input, d_output]
        # Forward
        context.execute_v2(bindings)
        # 将输出数据拷贝到CPU
        cuda.memcpy_dtoh(buffer, d_output)
        pred = process_np(buffer, 0.25, 0.45)
        # for det in pred:
        #     # 处理单张图片检测结果
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        #         # save results on image
        #         for x1, y1, x2, y2, conf, cls in det:
        #             cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (2, 128, 128), 5, cv2.LINE_AA)
        #             cv2.putText(img0, coco_list[int(cls)], (int(x1), int(y1) + 27), cv2.FONT_HERSHEY_SIMPLEX, 1, (2, 25, 25))
        #         # cv2.namedWindow('result', 0)
        #         # cv2.resizeWindow('result', (img0.shape[1]//5, img0.shape[0]//5))  # winname, width, height
        #         cv2.imshow('result', img0)
        #         cv2.waitKey(0)

if __name__ == '__main__':
    trt_path = "/home/cmv/PycharmProjects/yolov5/yolov5l_-1_3_-1_-1_int8.trt"
    img_txt = '/home/cmv/PycharmProjects/datasets/coco/val2017.txt'
    input_size = (640,640)
    engine = load_engine(trt_path)
    context = engine.create_execution_context()
    inference_dynamic(context,img_txt,input_size)
    # inference_static(context,img_txt,input_size)
# 动态 scaleFill
# s 5.5 fp16          5.1 int8
# m 7.3 fp16          6.5 int8
# l 9.3 fp16          7.8 int8
# x 15.5 fp16         11.2 int8
