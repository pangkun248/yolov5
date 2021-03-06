# YOLOv5 ๐ by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box
from utils.metrics import bbox_ioa


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3', hard=True)  # version requirement

            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0)],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains ่ฟ้็ไธๆไธบไปไน่ฆ่ฟๆ ทๆไฝ
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))  # ๅฏๅ่ https://zhuanlan.zhihu.com/p/67930839
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # cv2.LUT ๅฎ็ฐ่ฟไธชๆ ๅฐ็จ็ๆฏOpenCV็ๆฅ่กจๅฝๆฐ 0-255 -> lut_hue/lut_sat/lut_val
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels ๆฐๆฎๅขๅผบ็ไธ็งๆนๅผ.ๅฐๅทฒๆ็่พๅฐ็ฉไฝ(ไปbox)ๆฐดๅนณๅ็ด็ฟป่ฝฌๅ่ฆ็ๅฐๅๅงๅพๅไธ,labelไนๆฏ  ๆณจๆๆฏbox่้ๅ็ด ็น็ฟป่ฝฌ๏ผ๏ผ๏ผ
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    img: ๅๅงๅพ็ inference้ถๆฎตไธบ img = cv.imread(path)
    new_shape: 1.train: new_shape->(int) 2.test:new_shapeๆฏๆๅคง่พน้ฟไธบimg_size็่ฝๅผๅฎนไธไธชbatchๅพๅ็ๆนๅฝขbatch_shape(h,w)
    color: ่ฟ่กpaddingๆถๅกซๅ็RGBๅผ
    auto:ๆฏๅฆ่ฟ่กๆๅฐๅคๆฅ็ฉๅฝขๅผ็padding
    scaleFill: ๆฏๅฆ็ดๆฅresizeๅฐ็ฎๆ ๅฐบๅฏธ,ๆ padding
    scaleup:ๆฏๅฆๅ่ฎธๅพๅๅไธ็ผฉๆพ,Falseๆถ,ๅช่ฝๅไธ็ผฉๆพ.
    stride:ๆจกๅๆๅคงไธ้ๆ ทๅๆฐ,ๅพๅ่ฟ่กpaddingๆถ้่ฆๅฐๅไธช่พนๅกซๅๅฐstride็ๅๆฐ
    ๆณจ!ๆ็ปๅพๅๅฝขๆๆไธ็ง
    1.ๅฎๅจๅกซๅ่ณnew_shape(auto=False,scaleFill=False)
    2.ๆๅฐๅคๆฅ็ฉๅฝขๅกซๅ(auto=True,scaleFill=True/False)
    3.ๆ ๅกซๅ,ๅ่พน็ดๆฅresizeๅฐ็ฎๆ ๅฐบๅฏธ(auto=False,scaleFill=True)
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) ๅฝๆ ๅๅฐบๅฏธๅคงไบๅๅงๅฐบๅฏธๅฎฝ้ซๆถ,ไธscaleupใautoใautoไธบFalse,้ฃไนไธๆพ็ผฉๅพๅๅฐบๅฏธ,ๅพๅๅฐบๅฏธๅคๆ ๅๅบๅๅกซๅ114
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP) ๅช็ผฉไธๆพ,้พ้ๆพๅคงไผไฝฟๅพๅๅคฑ็่ๅฏผ่ดmAPไธ้?
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios  imๆ็ปresizeๅฐnew_shape็ๅไธช่พน็ๆพ็ผฉๆฏไพ
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # ๅ ไธบcv2.resize็็ฎๆ sizeๅๆฐๆฏ w,hๆ ผๅผ,ๆไปฅ่ฟ้่ฆๆดๆข้กบๅบ
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle  ๆๅฐๅคๆฅ็ฉๅฝข.ๅณๅชๅจๆ็ญ่พน้ฟๅค้ขๅไธ็นpad,ไปฅๆปก่ถณๆๅฐ่พน้ฟไธบ32ๅๆฐ็ๆกไปถ
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch  ไธๅฏน็ญ่พนๅpadding,img็ดๆฅresizeๅฐnew_shape
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # (width, height) ratios  ้ๆฐ่ฎก็ฎๆพ็ผฉๆฏไพ

    dw /= 2  # divide padding into 2 sides ๅฝdwๆdhไธบๅฅๆฐๆถ,้คไปฅ2ไนๅๅฐฑไผๅๆx.5 ็ถๅไธ้ขๆไธไธช+-0.1ๅนถๅ่ไบๅฅ็ๆไฝๆฅ้ฒๆญข่ฟ็งๆๅต
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, x y x y]
    # ไปฟๅฐๅๆขๅๅถ็ฉ้ต่กจ็คบๅ่: https://www.zhihu.com/question/20666664/answer/157400568
    # https://www.zhihu.com/question/20666664/answer/15790507  https://www.cnblogs.com/shine-lee/p/10950963.html
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective ้่ง ้ป่ฎคไธบ0,ๅ ไธบๅผๅฏไนๅไผ้ ๆๅพ็ๆไธๅฎๅพๆ่งๅบฆ,่boxไนๅฟ้กปๅๆญฅ.ไฝๆฏ็ฎๅYOLOv5ไธๆฏๆๆๆ่ฝฌ่งๅบฆ็box
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale  ๆ่ฝฌไธๆพ็ผฉ cv2็ธๅณๅฝๆฐ https://blog.csdn.net/qq_39507748/article/details/104448953
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix  ็ปๅๆ่ฝฌ็ฉ้ต cv2.warpPerspectiveไธcv2.warpAffine็็ธๅณๅทฎๅซ ๅ่งโ
    # https://blog.csdn.net/qq_27261889/article/details/80720359 https://zhuanlan.zhihu.com/p/37023649
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:  # ้่ง  ๆณจ! ่ฟ้ๆฏไปimgไธญ้ๆบๆฝๅ (width, height)ไฝไธบ่ฟๅๅพๅ. ๅฐบๅฏธไธๆฅ่ฏด == 4*img -> 1*img ไธๅ
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine ไปฟๅฐ
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates ๅฏนๆ ็ญพๅๆ ่ฟ่กๅๆข
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)  # ๅฎ้ไธ่ฟ้ๅฏไปฅๅฟฝ็ฅ,็ดๆฅ้ป่ฎคไธบFalseๅณๅฏ(ๅฆๆๆฒกๆๅๅฒๅฝขๅผ็ๆฐๆฎ้็่ฏ)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    # ่ฏฅๆนๆณๅถๅฎๅฐฑๆฏๅฐ็ฎๆ (ๅๅฒ)ๅบๅ ๆฐดๅนณ็ฟป่ฝฌไธไธ็ถๅ่ฆ็ๅฐๅๅงๅพๅไธ,ๅชไธ่ฟๅๅงๅๅฒๅบๅไธ่ฝ่ถ่ฟๅพๅไธญ้ดๅคชๅค(่ถๅบ่ช่บซ0.3ๅๅบๅ้ข็งฏ)
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]  # ๆฐดๅนณๆนๅ็ฟป่ฝฌๅ็l
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area ่ฎก็ฎๆฐดๅนณ็ฟป่ฝฌๅ็boxไธๅๅงbox็IOA
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels ๅฆๆioaๅฐไบ0.3,ๅๅคๅถๅถๅๅฒๅบๅๅฐๆฐดๅนณๅฆไธไพง
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))  # ๅฐ็ฟป่ฝฌๅ็segๅๆ ๆทปๅ ่ฟๆฅ
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)  # ๅๅงๅๅฒๅบๅ่ตๅผ

        result = cv2.bitwise_and(src1=im, src2=im_new)  # ่ทๅๅๅฒๅบๅๅ็ด 
        result = cv2.flip(result, 1)  # augment segments (flip left-right) ๅฐๅๅงๅๅฒๅบๅๆฐดๅนณ็ฟป่ฝฌ
        i = result > 0  # pixels to replace  ่ทๅ็ฟป่ฝฌๅ็ๅๅฒๅบๅๅ็ด ็ดขๅผ
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug  ๅฐ็ฟป่ฝฌๅ็ๅบๅ่ฆ็ๅฐๅๅงๅพๅไธ  ๆณจๆๆฏๅ็ด ็น็ฟป่ฝฌ๏ผ๏ผ๏ผ

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        # create random masks ็ๆ31ไธชไธๅๅฎฝ้ซ็maskๅบๅ,ๅนถๅกซๅไธๅ็ๅผ.ๅๅฆmaskๅบๅไธbox็iou่ถ่ฟ0.6ๅๆboxไธขๅผ
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # ็ๆ้ๆบmaskๅบๅ
            mask_w = random.randint(1, int(w * s))
    
            # box้ๅถๅบๅ่ๅด
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
    
            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
    
            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0 ๅฏๅ่ https://zhuanlan.zhihu.com/p/24555092
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]  # ๅๆขๅ็box็ๅฎฝ้ซใ้ข็งฏๅฟ้กปๅๅซๅคงไบwh_thrใarea_thr้ๅผ
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio ๅๆขๅ็boxไธ่ฝ่ฟไบ็ป้ฟ
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
