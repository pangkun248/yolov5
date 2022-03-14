# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-anchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import LOGGER, colorstr, emojis

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    # 根据 YOLOv5 Detect() 模块 m 的stride顺序检查anchor顺序,并在必要时更正
    a = m.anchors.prod(-1).view(-1)  # anchor面积  anchors.shape -> (nl, 1, na, 1, 1, 2)
    da = a[-1] - a[0]  # anchor面积的差值
    ds = m.stride[-1] - m.stride[0]  # stride的差值  stride -> tensor([ 8., 16., 32.])
    if da.sign() != ds.sign():  # stride大小顺序必须与anchor_grid面积大小顺序一致 否则就与stride保持一致
        LOGGER.info(f'{PREFIX}反转模型内部anchor顺序')
        m.anchors[:] = m.anchors.flip(0)  # 注torch.flip是反序地复制一份新数据,NumPy是返回一个view,所以torch.flip耗时更久


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # 检查anchor是否适合训练数据, 必要时重新生成anchor
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    # dataset.shapes是指原始图像的shapes(w,h) 并以最大边为基准将shape同比例放缩到img_size
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        """
        这里关于为何要如此计算bpr  注!以下gt_box与anchor的公共区域面积将以S来称呼 短边占长边以L来称
        0. 这里的wh受限于img_size,即不管原始图像为多大尺寸,(最大边)都要放缩到img_size尺寸(另一边等比例).显然anchor也受限于img_size尺寸
        1. 通常我们比较gt_box与anchor的相似度都是按照IoU来计算,那么为了方便计算都是把两边的box以左上角对齐,坐标(0,0)
        2. r.shape-> n,9,2 然后torch.min(r, 1. / r)这步操作的意义是获取gt_box与anchor的公共部分占最大边长的比例.
        注意是相对值,所以一定小于等于1,所以这里需要用到1/r,同时也是短边占长边的比例,无所谓是gt_box占anchor还是anchor占gt_box
        即无论是谁的边长(这里拿宽举例,高同理)更长,要获取的都是短边占长边的几分之几
        3. .min(2)[0]->n,9 取第3个维度上最小的值,因为要保守估计,然后此时的x^2实际上就是最小S值了,即gt_box与anchor的S大于等于x^2
        4. .max(1)[0]->n,  取每个gt_box与所有anchor的L(wh中的最小值)的最大值,即计算和每个gt_box最匹配的anchor的最小S为多少
        5. best(n) > 1. / thr, 代表n个 gt_box与所有anchor的最大L 这里1/thr可以理解为L阈值即0.25 超过该值即可认为这些anchor合格
        6. 计算所有(gt_box是否与最佳匹配anchor的L超过阈值)的均值(非L均值),如超过0.98则认为这些anchor符合要求否则需用k-means重新计算
        """
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold  每个gt_box与anchor的IoU超过阈值的数量,再取平均
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    # 这里m.anchor_grid与m.anchors是一致的,只不过shape不同.同时它们的值取决于hyp.*.yaml中anchors(记为na)是否注释.
    # 如果没注释则使用[range(6),*na]来作为基础值,如果注释则取决于*.yaml 或*.pt中内置的anchors -> *.yaml.get('anchors',*.pt.anchors)
    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # 重新生成anchor的阈值
        LOGGER.info(emojis(f'{s}Current anchors are a good fit to dataset ✅'))
    else:
        LOGGER.info(emojis(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...'))
        na = m.anchors.numel() // 2  # anchor总数量
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f'{PREFIX}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # 更新anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(emojis(s))


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ 从训练数据中利用k-means生成anchors

        Arguments:
            dataset:    data.yaml的路径或者是加载了该路径的dict
            n:          生成anchor的个数
            img_size:   训练的输入尺寸
            thr:        训练阶段的anchor与target的同边差异阈值,小于该阈值意为差距过大 默认=4.0
            gen:        使用遗传算法来改进anchor的轮数
            verbose:    是否每轮都输出进化结果k

        Return:
            k: 经过kmeans改善了的anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):
        """
        返回值大概可以这么理解 torch.where(max(min((w,anchor_w),(h,anchor_h)),*n)>thr,1,0).mean()
        n:anchor数量
        w:标注物体的w >2px
        h:标注物体的h >2px
        """
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness  所有gt_box与最佳匹配anchor的S均值

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large 根据面积从小到大进行排序
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # wh最小->k最大L值中大于thr的均值,每个wh平均有几个合格anchor
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for i, x in enumerate(k):
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # 获取标注物体的wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # 原始宽高

    # 过滤
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}警告: 发现非常小的标注物体. {i}个标注物体的宽高小于3px(共{len(wh0)}).')
    wh = wh0[(wh0 >= 2.0).any(1)]  # 过滤掉长度小于2px之后的宽高
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # k-means生成
    LOGGER.info(f'{PREFIX}在宽高大于2px的 {len(wh)} 个点上通过k-means计算出合适的 {n} 个anchors...')
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance 进入k-means之前先除以标准差.出来之后再乘以标准差 why?
    assert len(k) == n, f'{PREFIX}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # 过滤后的wh
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # 过滤前的wh
    k = print_results(k, verbose=False)  # 进化之前输出一下anchors

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # 开始进化
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{PREFIX}利用遗传算法改进anchors:')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # 进化直到发生变化(防止重复)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1) .clip(0.3, 3.0)  # 最后限制进化范围
        kg = (k.copy() * v).clip(min=2.0)  # 限制anchor的wh最小值
        fg = anchor_fitness(kg)  # 所有gt_box与最佳匹配anchor的S均值
        if fg > f:
            f, k = fg, kg.copy()  # 这里进行阈值与anchor的更新.即只会保存最好的S均值与anchor
            pbar.desc = f'{PREFIX}利用遗传算法改进anchors: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k)  # 进化之后输出一下anchor
