ONNX->TensorRT

由于torch->onnx已由yolov5自身提供接口(export.py)
其本身提供的接口默认是静态转换.即默认1x3x640x640.如需动态转换需要自行修改

由于v5Detect层输出时会有两个值,第一个为合并数据,第二个为未处理数据,在实际检测时不需要第二个数据 所以转onnx时需要将返回值做如下修改

(torch.cat(z, 1), x) -> torch.cat(z, 1)

更新时间 2021-07-26

硬件环境:RTX-2060 i7-10700 SSD 1T 内存-2666赫兹-16Gx1

软件环境:Ubuntu 18.04 python 3.8 pytorch 1.7.1 CUDA11.0 cuDNN8.0.5

测试数据:instances_val2017.json,mAP 通过COCO API 计算(注!mAP与Speed是分开计算的 因为二者的conf与iou阈值不一致)

动态输入 [1,3,640*]

|Model |FP16<br><sup>Speed-ms (mAP) |TRT-FP16<br>Speed (mAP) |TRT-INT8<br>Speed (mAP)|
|---          |---   |---           |---                        |
|YOLOv5s      |8.9(36.6)   |5.0(36.4)     |4.6(28.7)                  |
|YOLOv5m      |11.5(44.4)  |6.6(44.0)     |5.9(32.9)                  |  
|YOLOv5l      |14.7(48.1)  |9.0(47.5)     |7.3(38.0)                  |
|YOLOv5x      |26.5(50.3)  |13.6(49.7)    |9.8(42.3)                  |

静态输入 以[1,3,640,640]为例(不足640的边默认padding(value=114))

|Model |TRT-FP16<br>Speed (mAP)|TRT-INT8<br>Speed (mAP)|
|---          |---      |---                     |
|YOLOv5s      |4.6 (36.5)     |4.3 (28.8)        |
|YOLOv5m      |6.5 (44.1)     |5.7 (33.1)        |
|YOLOv5l      |8.5 (47.7)     |7.1 (38.1)        |
|YOLOv5x      |14.4 (49.7)    |10.4 (42.5)        |

以下是可能会出现的错误或警告


1.[TensorRT] ERROR: FAILED_EXECUTION: std::exception

 [TensorRT] ERROR: safeContext.cpp (184) - Cudnn Error in configure: 7 (CUDNN_STATUS_MAPPING_ERROR)

可能的原因:代码中存在torch.cuda.synchronize()

2.ImportError: libnvinfer.so.7: cannot open shared object file: No such file or directory

环境变量后添加LD_LIBRARY_PATH=你的TensorRT存放路径/lib (其中包括pycharm运行配置参数) 具体自行百度

3.出现含有Mindims 或 Maxdims  或 < 或 > 等关键词的警告或错误信息

则大概率是你的输入图片尺寸超出模型可接受范围了

4.WARNING: Missing dynamic range for tensor

将目录中的calib_yolo.bin删除即可

5.PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
   cuMemFree failed: an illegal memory access was encountered

实际inference图片尺寸大小与trt模型输出维度不一致