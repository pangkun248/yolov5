"""Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from threading import Thread

import math
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import test  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.metrics import fitness

# DDP相关变量的设置  WORLD_SIZE为你的机器数量  RANK为当前机器在所有机器中的index(0,1,2..),-1为非DDP模式
# 参考 https://zhuanlan.zhihu.com/p/86441879
logger = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, notest, nosave, workers, = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.notest, opt.nosave, opt.workers

    # 目录相关
    save_dir = Path(save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # 新建权重文件夹包括父级目录
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'  # 这个是保存mAP指标相关的数据

    # 超参数
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # 保存本次运行时的配置(参数与超参数)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not evolve  # create plots  不太懂为什么plots与evolve冲突
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)  # 这里是根据每台机器的id来固定seed
    with open(data) as f:
        data_dict = yaml.safe_load(f)  # 训练集参数

    # Loggers 日志
    loggers = {'wandb': None, 'tb': None}  # loggers dict
    if RANK in [-1, 0]:
        # TensorBoard 可视化
        if not evolve:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            loggers['tb'] = SummaryWriter(str(save_dir))

        # W&B
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        run_id = run_id if opt.resume else None  # start fresh run if transfer learning
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        if loggers['wandb']:
            data_dict = wandb_logger.data_dict
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # 如果是是继续训练的话,Logger可能会更新权重与epochs

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # 如果提供的是pt模型,则基于此模型继续训练
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(RANK):  # 让从机进程等待主机进程的读取数据完成之后再进行其他操作
            weights = attempt_download(weights)  # 如果本地找不到权重文件,则尝试下载
        ckpt = torch.load(weights, map_location=device)  # 加载权重
        # 如果opt.cfg中指定了model.yaml则加载此配置,否则加载权重中内置的(包括anchors)  以及hyp文件中的anchors(默认注释)是int而非list
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # 如果resume=False并且cfg指定某个model.yaml文件的话,那么就不加载opt.weights中的anchors直接使用model.yaml中的anchors(初始化)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # 所要加载的权重(加载权重与训练权重的交集)
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('训练模型加载比例 %g/%g 来自 %s' % (len(state_dict), len(model.state_dict()), weights))
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # 不依赖官方权重从零开始训练,但opt.cfg必须指定
    with torch_distributed_zero_first(RANK):
        check_dataset(data_dict)  # 检查数据是否存在
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze  冻结某些层 但这里作者不推荐使用.因为无论如何都不会取得更好的性能(精度),除非显存受限
    freeze = []  # 冻结的参数名称 (全部或部分)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # 理论bs,也即batch_size累计到该数量时才进行梯度更新
    accumulate = max(round(nbs / batch_size), 1)  # 优化前的累积损失次数  total_batch_size -> 实际bs
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # 改变权重衰减系数,当实际bs<64时不变,否则x(实际bs/理论bs)
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # 简而言之就是conv的weight有hyp['weight_decay'],其他权重参数没有.其他方面都没区别 但是不清楚为什么要这样做,以及怎么想到这种方法的
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases  conv.bias + bn.bias
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay  bn.weight
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay  conv.weight

    if opt.adam:  # 默认adam=False optimizer.param_groups -> [bn.w, conv.w, bias]
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases) 这里不明白为什么要把pg2单独放在一个group.明明和pg0一样的优化配置
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g bn.weight' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # 相对:cosine 1->hyp['lrf']  绝对:lr0->lr0*lrf  T=epochs lr趋势cos[0,π]
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # from utils.plots import plot_lr_scheduler
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA 提高测试指标使模型健壮性更强 TODO 暂时没搞明白 参考 https://zhuanlan.zhihu.com/p/68748778
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer  加载预训练模型的优化参数等信息 如果基于官方s m l x四个模型的话 以下四个属性是默认None None None None -1的
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results 如果预训练中有训练相关信息的话则把它复制进将要保存的result.txt中去,
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs  当一个模型训练到最后一个epoch结束时会将其赋值为-1,即训练结束
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s 训练已结束(%g轮),已无法继续.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s 已训练 %g 轮. 现额外微调 %g 轮.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # 更新训练的总轮数

        del ckpt, state_dict  # 用完就丢 工具模

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride) 除非魔改网络否则最大步长就是32
    nl = model.model[-1].nl  # 检测层的数量 (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 验证img_size是否是gs的整数倍

    # DP mode 单机多卡(不建议这使用该种模式,性能几乎没有提升.建议DDP)
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 可参考 https://zh.mxnet.io/blog/syncbn#batch-normalization%E5%A6%82%E4%BD%95%E5%B7%A5%E4%BD%9C
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # 训练集  hyp为dict 来自 opt.hyp
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=RANK,
                                            workers=workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, data, nc - 1)

    # Process 0
    if RANK in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=opt.cache_images and not notest, rect=True, rank=-1,
                                       workers=workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency 每个class出现的频率
            # model._initialize_biases(cf.to(device))  # 根据cls的频率初始化检测层的bias对应的各个cls的值
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if loggers['tb']:
                    loggers['tb'].add_histogram('classes', c, 0)  # TensorBoard

            # Anchors  检查模型中的anchor是否与数据集想适应,否则利用k-means与遗传算法优化anchor
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)  iou占obj_conf的比例,具体参见obj_loss计算部分
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # 根据每个cls出现频率大小赋予反向的权重
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training 同时将iter限制在剩余iter/2以内
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)  根据img_weight重新生成n个dataset的图片索引
        if opt.image_weights:
            # Generate indices
            if RANK in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if RANK != -1:
                indices = (torch.tensor(dataset.indices) if RANK == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if RANK != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if RANK != -1:
            dataloader.sampler.set_epoch(epoch)  # 和shuffle有关 https://www.zhihu.com/question/67209417/answer/1017851899
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar 只在主机上显示训练进度相关信息
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # 已训练batch数 non_blocking参考 https://www.zhihu.com/question/274635237/answer/756144739
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                # 不太明白为什么要在Warmup前期设置loss每batch更新一次,后期变慢至 (nbs/total_batch_size)个batch才更新一次,有利于收敛?
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr 0.1 -> lr0, other lrs 0.0 -> lr0  注! bias是慢慢减小 其余参数是慢慢增大
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 默认关闭
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size  这里+gs是为了保证各个size概率一致
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward(默认开启混合精度训练)
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:  # 四张图片所以要*4,但其实有一定概率(0.5)是一张图像边长放缩两倍,所以这里loss*4不太"合适"
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Print 当且仅当单机或主机上才显示训练信息
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    if loggers['tb'] and ni == 0:  # TensorBoard
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')  # suppress jit trace warning
                            loggers['tb'].add_graph(torch.jit.trace(de_parallel(model), imgs[0:1], strict=False), [])
                elif plots and ni == 10 and loggers['wandb']:
                    wandb_logger.log({'Mosaics': [loggers['wandb'].Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # DDP process 0 or single-GPU
        if RANK in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs  # 判断是否是最后一轮
            if not notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, _ = test.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz_test,
                                            model=ema.ema,
                                            single_cls=single_cls,
                                            dataloader=testloader,
                                            save_dir=save_dir,
                                            save_json=is_coco and final_epoch,
                                            verbose=nc < 50 and final_epoch,
                                            plots=plots and final_epoch,
                                            wandb_logger=wandb_logger,
                                            compute_loss=compute_loss)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if loggers['tb']:
                    loggers['tb'].add_scalar(tag, x, epoch)  # TensorBoard
                if loggers['wandb']:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # [P, R, mAP@.5, mAP@.5-.95]的加权组合
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # 保存模型 如果no_save=True 则只有最后一个epoch才保存, 如果no_save=False 则每个epoch都保存,
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(de_parallel(model)).half(),  # 默认以FP16格式储存
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if loggers['wandb']:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png 将results.txt中的内容绘制成png图像
            if loggers['wandb']:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [loggers['wandb'].Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        if not evolve:
            if is_coco:  # COCO dataset
                for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                    results, _, _ = test.run(data_dict,
                                             batch_size=batch_size // WORLD_SIZE * 2,
                                             imgsz=imgsz_test,
                                             conf_thres=0.001,
                                             iou_thres=0.7,
                                             model=attempt_load(m, device).half(),
                                             single_cls=single_cls,
                                             dataloader=testloader,
                                             save_dir=save_dir,
                                             save_json=True,
                                             plots=False)

            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # 剥去不必要的模型信息
            if loggers['wandb']:  # Log the stripped model
                loggers['wandb'].log_artifact(str(best if best.exists() else last), type='model',
                                              name='run_' + wandb_logger.wandb_run.id + '_model',
                                              aliases=['latest', 'best', 'stripped'])
        wandb_logger.finish_run()

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    set_logging(RANK)
    if RANK in [-1, 0]:  # 非DDP 或者 当前机器为训练主机时(RANK=0)
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
        check_git_status()      # 检查git状态 是否非docker镜像 是否联网等,以及最终拉取最新版本的代码到本地
        check_requirements(exclude=['thop'])  # 检测相应的py版本及requirements.txt中依赖的库,如果缺少则安装,但exclude除外
    # 恢复训练,TODO 我不知道以下check_wandb_resume的意义何在,似乎与DDP(我没实际使用过)有关.以及其他机器上加载wandb上的权重与数据？
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # 继续上一次中断了的训练
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # 指定具体的路径 或 最近一次训练保存的权重
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # 将上一次训练权重的超参数替换进现有的超参数
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # 恢复一些参数设置
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # 检查文件是否指定或唯一
        # 模型与配置文件至少存在一个,否则网络无法定义
        # weights为空时,即从零开始训练此时需根据cfg即model.yaml来定义网络.weights不为空时,根据weights中内置的配置文件来定义网络
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))

    # DDP模式 相关的参数配置(涉及到多卡时) 参考 https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-ddp-instead
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=60))
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:  # 如果有多台机器,并且当前代码环境在主机上
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

    # Evolve hyperparameters (optional)
    else:
        # 超参数进化 原始数据 (进化尺度 0-1, 最小值, 最大值)
        meta = {'lr0': (1, 1e-5, 1e-1),  # 初始化学习率 (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # 最终OneCycleLR学习率 = lr0*lrf
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # 优化器的权重衰减值
                'warmup_epochs': (1, 0.0, 5.0),  # warmup的轮数 (小数也可以)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup的初始动量
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup的初始偏差学习率
                'box': (1, 0.02, 0.2),  # box loss gain  box损失的权重
                'cls': (1, 0.2, 4.0),  # cls loss gain   cls损失的权重
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss  分类损失中的正样本权重
                'obj': (1, 0.2, 4.0),  # obj loss gain   obj损失的权重(scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss  目标损失中的正样本权重
                'iou_t': (0, 0.1, 0.7),  # 训练时IoU阈值 在build_targets阶段anchor与target的IoU超过此值即认为该anchor为正样本
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold  target与anchor的同边比阈值,生成anchor与过滤target的时候使用
                'anchors': (2, 2.0, 10.0),  # 每个grid的anchor数量 (0代指忽略)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # 图片 HSV-色调 增强幅度 (小数)
                'hsv_s': (1, 0.0, 0.9),  # 图片 HSV-饱和度 增强幅度 (小数)
                'hsv_v': (1, 0.0, 0.9),  # 图片 HSV-亮度 增强幅度 (小数)
                'degrees': (1, 0.0, 45.0),  # 图片旋转角度  (+/- 度数)
                'translate': (1, 0.0, 0.9),  # 图片水平或垂直平移的幅度 (+/- 小数)
                'scale': (1, 0.0, 0.9),  # 图片缩放尺寸 (+/- gain)
                'shear': (1, 0.0, 10.0),  # 图片剪切 (+/- deg)
                'perspective': (0, 0.0, 0.001),  # 透视变换参数  (+/- 小数), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # 图片上下翻转 (概率)
                'fliplr': (0, 0.0, 1.0),  # 图片左右翻转 (概率)
                'mosaic': (1, 0.0, 1.0),  # 图片 mosaic (概率)
                'mixup': (1, 0.0, 1.0),  # 图片 mixup (概率)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
        assert LOCAL_RANK == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # 从谷歌云盘下载evolve.txt 如果存在的话

        for _ in range(opt.evolve):  # 这里进化300次,但是每次都训练300epoch?
            if Path('evolve.txt').exists():  # 如果evolve.txt存在,选择最好的超参数并进化
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)  # 其实evolve.txt内的参数在每epoch更新的时候已经根据"mAP"从高到低排好序了
                n = min(5, len(x))  # 兼容evolve.txt行数小于5的情况
                x = x[np.argsort(-fitness(x))][:n]  # 根据evolve每行结果计算出"mAP",并从高到底重排序,最后截取前n(<5)个
                w = fitness(x) - fitness(x).min() + 1E-6  # weights  计算出前n个"mAP"与最低"mAP"的差值 以及防止作为分母为0
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # 随机选择
                    x = x[random.choices(range(n), weights=w)[0]]  # 根据最高"mAP"与最低"mAP"的差值权重随机选取x
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # 加权组合

                # 演化
                mp, s = 0.8, 0.2  # 每个超参数的演化概率, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # 演化尺度(单位)
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # 当v不全等于1时才开始演化 (防止原地不动)
                    # 演化尺度 * 是否演化 * 正太分布 * 随机小数 * sigma +1 再限制范围
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # 超参数正式演化!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # 通过meta中设置的值来限制hyp中的超参数范围,
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # 最小值限制
                hyp[k] = min(hyp[k], v[2])  # 最大值限制
                hyp[k] = round(hyp[k], 5)  # 设置值的有效范围-小数点后五位

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # 将每次演化之后的最优结果(hyp和"mAP")更新到yaml及"evolve.txt"文件中去
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # 绘制进化结果
        plot_evolution(yaml_file)
        print(f'超参数进化完成. 最佳结果另存为: {yaml_file}\n'
              f'使用这些超参数训练新模型的命令: $ python train.py --hyp {yaml_file}')


def run(**kwargs):
    # Usage: import train; train.run(imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
