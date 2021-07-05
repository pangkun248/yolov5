<div align="center">
<p>
<a align="left" href="https://ultralytics.com/yolov5" target="_blank">
<img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>
</p>
<br>
<div>
<a href="https://github.com/ultralytics/yolov5/actions"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
<a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="Open In Kaggle"></a>
<br>  
<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
<a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
</div>
  <br>
  <div align="center">
    <a href="https://github.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://twitter.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://youtube.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="2%"/>
    </a>
    <img width="2%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="2%"/>
    </a>
</div>

<br>
<p>
YOLOv5 🚀 是在 COCO 数据集上预训练的一系列对象检测框架和模型, 代表了 <a href="https://ultralytics.com">Ultralytics</a> 
对未来视觉AI方法的开源研究,结合了在数千小时的研究和开发过程中总结的经验教训和最佳实践.
</p>

<!-- 
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>


## <div align="center">文档</div>

参见 [YOLOv5 Docs](https://docs.ultralytics.com) 相关训练、测试和部署的完整文档。.


## <div align="center">快速入门示例</div>


<details open>
<summary>Install</summary>

Python >= 3.6.0 需要安装所有 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) 中的依赖项:
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->
```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```
</details>

<details open>
<summary>Inference</summary>

使用 YOLOv5 和 [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) 进行检测. 模型会自动从 [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) 下载.

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, PIL, OpenCV, numpy, multiple

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>



<details>
<summary>Inference with detect.py</summary>

`detect.py` 可以在各种图片源上进行检测, 并且自动从 [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) 下载模型,也可以将检测结果保存到 `runs/detect`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

执行以下命令可以复现在 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) 上的结果 (数据集在第一次使用时会自动下载). 在 YOLOv5s/m/l/x 上的训练时间为 2/4/6/8 天(单卡V100) (多卡自然会快). 使用你的GPU允许的最大的 `--batch-size` (下面展示的batch size为16GB显存的卡).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>  

<details open>
<summary>Tutorials</summary>

* [训练自己的数据](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [获得最佳训练效果的技巧](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️ RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Supervisely Ecosystem](https://github.com/ultralytics/yolov5/issues/2518)&nbsp; 🌟 NEW
* [多卡训练](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TorchScript, ONNX, CoreML 模型转换](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [测试增强 (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [模型集成](https://github.com/ultralytics/yolov5/issues/318)
* [模型剪枝](https://github.com/ultralytics/yolov5/issues/304)
* [超参数演化](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT 部署](https://github.com/wang-xinyu/tensorrtx)

</details>


## <div align="center">Environments and Integrations</div>

在几秒钟内即可使用经过我们验证的环境和集成(?), 包括用于自动记录 YOLOv5 实验的 [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme). 点击下面的每个图标以了解详细信息.

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
    <a href="https://www.kaggle.com/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="15%"/>
    </a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="15%"/>
    </a>
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_readme">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-wb-small.png" width="15%"/>
    </a>
</div>  


## <div align="center">Compete and Win</div>

我们对我们首次举办的 Ultralytics YOLOv5 🚀 EXPORT 比赛感到非常激动,奖金 **$10,000**!  

<p align="center">
  <a href="https://github.com/ultralytics/yolov5/discussions/3213">
  <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-export-competition.png"></a>
</p>


## <div align="center">Why YOLOv5</div>

<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313216-f0a5e100-9af5-11eb-8445-c682b60da2e3.png"></p>
<details>
  <summary>YOLOv5-P5 640 性能指标 (点击展开)</summary>
  
<p align="center"><img width="800" src="https://user-images.githubusercontent.com/26833433/114313219-f1d70e00-9af5-11eb-9973-52b1f98d321a.png"></p>
</details>
<details>
  <summary>图片注释 (点击展开)</summary>
  
  * GPU 速度为使用batch size为 32 的 V100 GPU,并计算每张图像end-to-end的平均时间(超过5000张 COCO val2017 图像), 包括图像预处理、PyTorch FP16 inference、NMS和后处理. 
  * EfficientDet 的数据来自 [google/automl](https://github.com/google/automl) batch size为8.
  * **复现** `python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`
</details>


### 预训练权重

[assets]: https://github.com/ultralytics/yolov5/releases

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPs<br><sup>640 (B)
|---                    |---  |---      |---      |---      |---     |---|---   |---
|[YOLOv5s][assets]      |640  |36.7     |36.7     |55.4     |**2.0** |   |7.3   |17.0
|[YOLOv5m][assets]      |640  |44.5     |44.5     |63.1     |2.7     |   |21.4  |51.3
|[YOLOv5l][assets]      |640  |48.2     |48.2     |66.9     |3.8     |   |47.0  |115.4
|[YOLOv5x][assets]      |640  |**50.4** |**50.4** |**68.8** |6.1     |   |87.7  |218.8
|                       |     |         |         |         |        |   |      |
|[YOLOv5s6][assets]     |1280 |43.3     |43.3     |61.9     |**4.3** |   |12.7  |17.4
|[YOLOv5m6][assets]     |1280 |50.5     |50.5     |68.7     |8.4     |   |35.9  |52.4
|[YOLOv5l6][assets]     |1280 |53.4     |53.4     |71.1     |12.3    |   |77.2  |117.7
|[YOLOv5x6][assets]     |1280 |**54.4** |**54.4** |**72.0** |22.4    |   |141.8 |222.9
|                       |     |         |         |         |        |   |      |
|[YOLOv5x6][assets] TTA |1280 |**55.0** |**55.0** |**72.0** |70.8    |   |-     |-

<details>
  <summary>表格说明 (点击展开)</summary>
  
  * AP<sup>test</sup> 表示 COCO [test-dev2017](http://cocodataset.org/#upload) 服务器上的结果, 所有其他 AP 结果是在 val2017 上的准确性.  
  * 除非另有说明,否则AP 值仅适用于单一模型单一尺度. **复现 mAP** `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`  
  * Speed<sub>GPU</sub> 超过5000张COCO val2017图像上的平均值,使用 GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100, 其中包括 FP16 inference, NMS和后处理. **复现速度** `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`  
  * 所有权重都是在上用默认配置和超参数训练300个epoch (没有自动增强?). 
  * 测试增强 ([TTA](https://github.com/ultralytics/yolov5/issues/303)) 包括翻转和放缩. **复现 TTA** `python test.py --data coco.yaml --img 1536 --iou 0.7 --augment`
</details>


## <div align="center">贡献</div>

我们欢迎您的意见! 我们想让对 YOLOv5 的贡献尽可能简单和透明. 请参阅我们的 [贡献指南](CONTRIBUTING.md) 开始使用. 


## <div align="center">联系</div>

有关运行 YOLOv5 的问题,请访问 [GitHub Issues](https://github.com/ultralytics/yolov5/issues). 如需业务或专业支持请求,请访问
[https://ultralytics.com/contact](https://ultralytics.com/contact).

<br>

<div align="center">
    <a href="https://github.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://twitter.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://youtube.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="3%"/>
    </a>
</div>
