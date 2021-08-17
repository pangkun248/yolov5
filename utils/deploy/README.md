ONNX->TensorRT

由于torch->onnx已由yolov5自身提供接口(export.py)
其本身提供的接口默认是静态转换.即默认1x3x640x640.如需动态转换需要自行修改

由于v5Detect层输出时会有两个值,第一个为合并数据,第二个为未处理数据,在实际检测时不需要第二个数据

所以转onnx时需要将返回值做如下修改 (torch.cat(z, 1), x) -> torch.cat(z, 1) (这个对性能影响很小,只是为了inference时少写一些代码)

硬件环境:RTX-2060 i7-10700 SSD 1T 内存-2666赫兹-16Gx1

软件环境:Ubuntu 18.04 python 3.8 pytorch 1.7.1 CUDA11.0 cuDNN8.0.5

测试数据:instances_val2017.json,mAP 通过COCO API 计算(注!mAP与Speed是分开计算的 因为二者的conf与iou阈值不一致)

动态输入 [1,3,640*]

|Model |Torch<br><sup>Speed-FP16 |Torch<br><sup>mAP-FP32 |TensorRT<br><sup>Speed-FP16 |TensorRT<br><sup>mAP-FP16 |TensorRT<br><sup>Speed-INT8|TensorRT<br><sup>mAP-INT8|
|---          |---   |---   |---           |---           |--- |---                        |
|YOLOv5s      |8.9ms   |36.6   |5.0ms     |36.4     |4.6ms|28.7                 |
|YOLOv5m      |11.5ms  |44.4  |6.6ms     |44.0     |5.9ms|32.9                 |  
|YOLOv5l      |14.7ms  |48.1  |9.0ms     |47.5     |7.3ms|38.0                  |
|YOLOv5x      |26.5ms  |50.3  |13.6ms    |49.7    |9.8ms|42.3                |

静态输入 以[1,3,640,640]为例(不足640的边默认padding(value=114))

|Model |TensorRT<br><sup>Speed-FP16|TensorRT<br><sup>mAP-FP16|TensorRT<br><sup>Speed-INT8|TensorRT<br><sup>mAP-INT8|
|---          |--- |---      |---  |---                     |
|YOLOv5s      |4.6ms|36.5     |4.3ms        |28.8        |
|YOLOv5m      |6.5ms|44.1     |5.7ms        |33.1        |
|YOLOv5l      |8.5ms|47.7     |7.1ms        |38.1        |
|YOLOv5x      |14.4ms|49.7    |10.4ms       |42.5        |

[1,3,640,640] i7-10700 **无GPU** forward+post_process 计算方式: 1000次循环取平均 测试图片 /data/images/zidane.jpg

|Model |PyTorch|ONNX|OpenVINO|OpenCV-DNN|
|---          |--- |---      |---  |---                     |
|YOLOv5s      |105ms|54ms     |86ms        |395ms        |

注!ONNX支持动态输入尺寸,可以利用yolov5的rect inference,在zidane.jpg这样的图片上可达29ms,而PyTorch则为58ms
