# encoding: utf-8
import logging
import os
import sys
from multiprocessing import Pool, Queue, Process, Semaphore
import multiprocessing

from rich.progress import track
from rich import print
import joblib
import pynvml
import wandb

import torch
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

sys.path.append('.')
from engine import clustering
from engine import deep_feature_clustersing as dfc
from engine import deep_feature_clustersing_1 as dfc_origin
from engine.miou import compute_IoU
from PIL import Image
import time
import numpy as np
import random
import math

from utils.reid_metric import R1_mAP, R1_mAP_arm
from utils.vis import Visualizer
from collections import OrderedDict


def create_supervised_trainer_with_center(model, center_criterion_part, center_criterion_global, center_criterion_fore, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, cls_target, parsing_target = batch
        img = img.cuda()
        cls_target = cls_target.cuda()
        parsing_target = parsing_target.cuda()
        cls_score_part, cls_score_global, cls_score_fore, y_part, y_full, y_fore, part_pd_score = model(img)
        total_loss, g_loss, f_loss, p_loss, parsing_loss = loss_fn(cls_score_part, cls_score_global, cls_score_fore,
                                                                   y_part, y_full, y_fore, part_pd_score,
                                                                   cls_target, parsing_target)
        total_loss.backward()
        optimizer.step()
        for param in center_criterion_part.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        for param in center_criterion_global.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        for param in center_criterion_fore.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = ((cls_score_global.max(1)[1] == cls_target).float().mean()+(cls_score_part.max(1)[1] == cls_target).float().mean()+(cls_score_fore.max(1)[1] == cls_target).float().mean())/3.0
        return total_loss.item(), acc.item(), g_loss.item(), f_loss.item(), p_loss.item(), parsing_loss.item()

    return Engine(_update)


def create_supervised_trainer(model, optimizer,  loss_fn, cfg,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, cls_target, parsing_target = batch
        img = img.cuda()
        cls_target = cls_target.cuda()
        parsing_target = parsing_target.cuda()
        cls_score_part, cls_score_global, cls_score_fore, y_part, y_full, y_fore, part_pd_score \
            = model(img)
        total_loss, g_loss, f_loss, p_loss, parsing_loss = loss_fn(cls_score_part, cls_score_global, cls_score_fore,
                                                                   y_part, y_full, y_fore, part_pd_score,
                                                                   cls_target, parsing_target)
        total_loss.backward()
        optimizer.step()

        # compute acc
        acc = ((cls_score_global.max(1)[1] == cls_target).float().mean()+(cls_score_part.max(1)[1] == cls_target).
               float().mean()+(cls_score_fore.max(1)[1] == cls_target).float().mean())/3.0
        return total_loss.item(), acc.item(), g_loss.item(), f_loss.item(), p_loss.item(), parsing_loss.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None, with_arm=False):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            if with_arm:
                g_f_feat, part_feat, part_visible, _ = model(data)
                return g_f_feat, part_feat, part_visible, pids, camids
            else:
                feat, _ = model(data)
                return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def compute_features(clustering_loader, model, device, with_arm=False):
    model.to(device)
    model.eval()
    mask_target_paths = []
    feats = []
    pids = []

    with torch.no_grad():
        for batch_img, batch_mask_target_path, batch_pid in track(clustering_loader):
            batch_img = batch_img.to(device)
            # print(batch_img.shape)
            batch_feats = model(batch_img, compute_futures=True)
            batch_feats = batch_feats.detach().cpu()
            feats.append(batch_feats)
            pids.extend(batch_pid)
            mask_target_paths.extend(batch_mask_target_path)
            # torch.cuda.empty_cache()

    feats = torch.cat(feats, 0)
    shape = feats.shape  # [15618, 256, 64, 32]
    # feats = feats.view(shape[0], shape[1], -1)  # [15618, 256, 64*32]
    # feats = feats.permute(0, 2, 1)  # [15618, 64*32, 256]
    feats = feats.numpy()
    return feats, mask_target_paths, pids, shape


def cluster_for_each_identity(cfg, feats, mask_paths, shape):
    _, C, H, W = shape
    N = feats.shape[0]
    cluster_feats = feats.reshape(N * H * W, C)
    # print(cluster_feats.shape)

    # foreground/background clustering
    deepcluster = clustering.__dict__[cfg.CLUSTERING.AL](2, norm=False)

    fore_back_feats = np.linalg.norm(feats, axis=2)
    fore_back_feats = fore_back_feats / np.max(fore_back_feats, axis=1).reshape(N, 1)
    fore_back_feats = fore_back_feats.reshape(N * H * W, 1)
    if cfg.CLUSTERING.ENHANCED:
        fore_back_feats = 1.0 / (1.0 + np.exp(-5.0 * (2.0 * fore_back_feats - 1) - 3.0))

    clustering_loss = deepcluster.cluster(fore_back_feats, device=0, min_num=int(N * H * W / 10), max_num=N * H * W,
                                          verbose=False)

    # Determining which category is the foreground
    mean_len = []
    for i in range(deepcluster.k):
        length = []
        for pixel in deepcluster.pixels_lists[i]:
            length.append(fore_back_feats[pixel])
        mean_len.append(np.mean(length))

    if mean_len[0] > mean_len[1]:
        fore_list = deepcluster.pixels_lists[0]
        fore_list = list(filter(lambda x: fore_back_feats[x] != 0, fore_list))
        fore_feats = cluster_feats[fore_list]
    else:
        fore_list = deepcluster.pixels_lists[1]
        fore_list = list(filter(lambda x: fore_back_feats[x] != 0, fore_list))
        fore_feats = cluster_feats[fore_list]

    # human body parts clustering
    deepcluster = clustering.__dict__[cfg.CLUSTERING.AL](cfg.CLUSTERING.PART_NUM - 1, norm=True)
    clustering_loss = deepcluster.cluster(fore_feats, device=0, min_num=50, max_num=int(len(fore_list) / 2),
                                          verbose=False)

    mean_h = []
    for i in range(deepcluster.k):
        pos_h = []
        for pixel in deepcluster.pixels_lists[i]:
            pos_h.append(float(int(fore_list[pixel] % (H * W) / W)))
        mean_h.append(np.mean(pos_h))
    cluster2label = np.argsort(np.argsort(mean_h)) + 1
    # print(cluster2label)

    # save human parsing labels
    pgt = np.zeros((N, H, W)).astype('uint8')
    for i in range(deepcluster.k):
        for pixel in deepcluster.pixels_lists[i]:
            img_idx = int(fore_list[pixel] / (H * W))
            pos_h = int(fore_list[pixel] % (H * W) / W)
            pos_w = int(fore_list[pixel] % (H * W) % W)
            pgt[img_idx, pos_h, pos_w] = cluster2label[i]
    # print(pgt[0])

    for img_idx in range(N):
        pgt_img = Image.fromarray(pgt[img_idx].astype('uint8'))
        pgt_img.save(mask_paths[img_idx])
        

def do_train_with_center(
        cfg,
        model,
        center_criterion_part,
        center_criterion_global,
        center_criterion_fore,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,      # modify for using self trained model
        loss_fn,
        num_query,
        start_epoch,     # add for using self trained model
        clustering_loader
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    clustering_period = cfg.CLUSTERING.PERIOD
    clustering_stop = cfg.CLUSTERING.STOP
    with_arm = cfg.TEST.WITH_ARM

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion_part, center_criterion_global,
                                                    center_criterion_fore, optimizer, optimizer_center, loss_fn,
                                                    cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    if with_arm:
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP_arm(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device,
                                                with_arm=with_arm)
    else:
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device, with_arm=with_arm)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)  #
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer
                                                                     })

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'total_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'g_loss')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'f_loss')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'p_loss')
    RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'parsing_loss')

    wandb.init(project="HCGA", entity="cesaredou", name=cfg.OUTPUT_DIR.split("/")[-1])

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()
    
    @trainer.on(Events.EPOCH_STARTED)
    def adjust_mask_pseudo_labels(engine):

        if engine.state.epoch <= clustering_stop:
            # engine.state.epoch % clustering_period == 1 and
            torch.cuda.empty_cache()
            compute_f_begin = time.time()
            feats, pseudo_labels_paths, pids, shape = compute_features(clustering_loader, model, device, with_arm)
            logger.info('compute feature time: {0:.0f} s'.format(time.time() - compute_f_begin))
            torch.cuda.empty_cache()

            cluster_begin = time.time()
            logger.info('clustering and adjust pseudo-labels begin...')

            pid_label = set(pids)
            # for label in pid_label:
            #     indexs = [i for i in range(len(pids)) if pids[i] == label]
            #     feats_I = feats[indexs]
            #     pseudo_labels_paths_I = [pseudo_labels_paths[i] for i in indexs]
            #     cluster_for_each_identity(cfg, feats_I, pseudo_labels_paths_I, shape)

            visual_ret = OrderedDict()
            for label in pid_label:
                indexs = [i for i in range(len(pids)) if pids[i] == label]
                feats_I = feats[indexs]
                # torch.save(feats_I, 'feats/%d.pt' % label)
                pseudo_labels_paths_I = [pseudo_labels_paths[i] for i in indexs]
                vis = deep_feature_cluster_for_each_identity(feats_I, pseudo_labels_paths_I, label, cfg)
                if label < 10:
                    visual_ret[str(label)] = vis

            logger.info('mask adjust use time: {0:.0f} s'.format(time.time() - cluster_begin))

            # evaluate the pseudo-part-labels
            if engine.state.epoch > 200:
                if cfg.DATASETS.NAMES == 'market1501':  # /home/ubuntu/reid/reid-data/market1501/Market-1501-v15.09.15
                    pred_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'market1501/Market-1501-v15.09.15',
                                            cfg.DATASETS.PSEUDO_LABEL_SUBDIR)
                    gt_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'market1501//Market-1501-v15.09.15',
                                          cfg.DATASETS.PREDICTED_GT_SUBDIR)
                    compute_IoU(pred_dir, gt_dir, cfg.CLUSTERING.PART_NUM)

                elif cfg.DATASETS.NAMES == 'dukemtmc':
                    pred_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'dukemtmc-reid/DukeMTMC-reID', cfg.DATASETS.PSEUDO_LABEL_SUBDIR)
                    gt_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'dukemtmc-reid/DukeMTMC-reID', cfg.DATASETS.PREDICTED_GT_SUBDIR)
                    compute_IoU(pred_dir, gt_dir, cfg.CLUSTERING.PART_NUM)

                elif cfg.DATASETS.NAMES == 'cuhk03_np_labeled':
                    pred_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'cuhk03-np/labeled', cfg.DATASETS.PSEUDO_LABEL_SUBDIR)
                    gt_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'cuhk03-np/labeled', cfg.DATASETS.PREDICTED_GT_SUBDIR)
                    compute_IoU(pred_dir, gt_dir, cfg.CLUSTERING.PART_NUM)

                elif cfg.DATASETS.NAMES == 'cuhk03_np_detected':
                    pred_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'cuhk03-np/detected', cfg.DATASETS.PSEUDO_LABEL_SUBDIR)
                    gt_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'cuhk03-np/detected', cfg.DATASETS.PREDICTED_GT_SUBDIR)
                    compute_IoU(pred_dir, gt_dir, cfg.CLUSTERING.PART_NUM)

            torch.cuda.empty_cache()
            # visualizer.display_current_results(visual_ret, engine.state.epoch)
            wandb.log({""})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['total_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        data = OrderedDict()
        data['total_loss'] = engine.state.metrics['total_loss']
        data['g_loss'] = engine.state.metrics['g_loss']
        data['f_loss'] = engine.state.metrics['f_loss']
        data['p_loss'] = engine.state.metrics['p_loss']
        data['parsing_loss'] = engine.state.metrics['parsing_loss']
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0 or engine.state.epoch > 110:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()



    trainer.run(train_loader, max_epochs=epochs)


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,      # modify for using self trained model
        loss_fn,
        num_query,
        start_epoch,     # add for using self trained model
        clustering_loader
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    clustering_period = cfg.CLUSTERING.PERIOD
    clustering_stop = cfg.CLUSTERING.STOP
    with_arm = cfg.TEST.WITH_ARM

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, cfg, device=device)
    if with_arm:
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_arm(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device, with_arm=with_arm)
    else:
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device, with_arm=with_arm)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)  #
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer
                                                                     })

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'total_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'g_loss')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'f_loss')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'p_loss')
    RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'parsing_loss')

    wandb.init(project="HCGA", entity="cesaredou", name=cfg.OUTPUT_DIR.split("/")[-1])

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_mask_pseudo_labels(engine):
        if engine.state.epoch <= clustering_stop: 
            # engine.state.epoch % clustering_period == 1 and
            torch.cuda.empty_cache()
            compute_f_begin = time.time()
            feats, pseudo_labels_paths, pids, shape = compute_features(clustering_loader, model, device, with_arm)
            logger.info('compute feature time: {0:.0f} s'.format(time.time() - compute_f_begin))
            torch.cuda.empty_cache()

            cluster_begin = time.time()
            logger.info('clustering and adjust pseudo-labels begin...')

            pid_label = set(pids)
            # sigle id
            if cfg.DFC.MODE =='k-means':
                print(feats.shape)
                feats = feats.reshape(shape[0], shape[1], -1)  # [15618, 256, 64*32]
                feats = feats.transpose(0, 2, 1)  # [15618, 64*32, 256]
                print(feats.shape)
                for label in pid_label:
                    indexs = [i for i in range(len(pids)) if pids[i] == label]
                    feats_I = feats[indexs]
                    # torch.save(feats_I, 'feats/%d.pt' % label)
                    pseudo_labels_paths_I = [pseudo_labels_paths[i] for i in indexs]
                    cluster_for_each_identity(cfg, feats_I, pseudo_labels_paths_I, shape)

            # single process
            elif cfg.DFC.MODE =='sp':
                visual_ret = {}
                for label in pid_label:
                    indexs = [i for i in range(len(pids)) if pids[i] == label]
                    feats_I = feats[indexs]
                    # torch.save(feats_I, 'feats/%d.pt' % label)
                    pseudo_labels_paths_I = [pseudo_labels_paths[i] for i in indexs]
                    vis = deep_feature_cluster_for_each_identity(feats_I, pseudo_labels_paths_I, label, cfg)
                    if label < 10:
                        visual_ret[str(label)] = wandb.Image(vis)
                del feats
                wandb.log(visual_ret, step=engine.state.epoch)

            # multi process
            elif cfg.DFC.MODE == 'mp':
                # queries = [[0, 350], [350, 705]]
                # if cfg.DATASETS.NAMES == "cuhk03_np_detected":
                #      queries = [[0, 370], [370, 765]]
                # joblib.Parallel(n_jobs=-1)(
                #     [joblib.delayed(deep_feature_cluster_for_each_identity_multi_process)(i) for i in queries]
                # )

                p = Pool(3)
                data = []
                for label in track(pid_label):
                    indexs = [i for i in range(len(pids)) if pids[i] == label]
                    feats_I = feats[indexs]
                    pseudo_labels_paths_I = [pseudo_labels_paths[i] for i in indexs]
                    if cfg.DATASETS.NAMES =='cuhk03_np_detected':
                        data.append([feats_I[0:5], pseudo_labels_paths_I[0:5], label, cfg])
                        data.append([feats_I[-5:], pseudo_labels_paths_I[-5:], label, cfg])
                    else:
                        data.append([feats_I, pseudo_labels_paths_I, label, cfg])
                    # 统计显存q
                    # pynvml.nvmlInit()
                    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
                    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    # free_mem = meminfo.free / 1024 / 1024 / 1024
                    # # print(free_mem)
                    # if free_mem < 5:
                    #     time.sleep(5)

                p.map(deep_feature_cluster_for_each_identity_multi_process, data)
                p.close()
                p.join()
                del feats

            else:
                print("Error DFC Model!")
                raise SystemError

            logger.info('mask adjust use time: {0:.0f} s'.format(time.time() - cluster_begin))
            
            # visualizer.display_current_results(visual_ret, engine.state.epoch)
            
            #  evaluate the pseudo-part-labels
            if engine.state.epoch > 200:  # <= clustering_stop
                if cfg.DATASETS.NAMES == 'market1501':
                    pred_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'market1501/Market-1501-v15.09.15',
                                            cfg.DATASETS.PSEUDO_LABEL_SUBDIR)
                    gt_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'market1501//Market-1501-v15.09.15',
                                          cfg.DATASETS.PREDICTED_GT_SUBDIR)
                    compute_IoU(pred_dir, gt_dir, cfg.CLUSTERING.PART_NUM)

                elif cfg.DATASETS.NAMES == 'dukemtmc':
                    pred_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'dukemtmc-reid/DukeMTMC-reID',
                                            cfg.DATASETS.PSEUDO_LABEL_SUBDIR)
                    gt_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'dukemtmc-reid/DukeMTMC-reID',
                                          cfg.DATASETS.PREDICTED_GT_SUBDIR)
                    compute_IoU(pred_dir, gt_dir, cfg.CLUSTERING.PART_NUM)

                elif cfg.DATASETS.NAMES == 'cuhk03_np_labeled':
                    pred_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'cuhk03-np/labeled',
                                            cfg.DATASETS.PSEUDO_LABEL_SUBDIR)
                    gt_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'cuhk03-np/labeled', cfg.DATASETS.PREDICTED_GT_SUBDIR)
                    compute_IoU(pred_dir, gt_dir, cfg.CLUSTERING.PART_NUM)

                elif cfg.DATASETS.NAMES == 'cuhk03_np_detected':
                    pred_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'cuhk03-np/detected',
                                            cfg.DATASETS.PSEUDO_LABEL_SUBDIR)
                    gt_dir = os.path.join(cfg.DATASETS.ROOT_DIR, 'cuhk03-np/detected', cfg.DATASETS.PREDICTED_GT_SUBDIR)
                    compute_IoU(pred_dir, gt_dir, cfg.CLUSTERING.PART_NUM)

            torch.cuda.empty_cache()   
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['total_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))


    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()
        data = OrderedDict()
        data['total_loss'] = engine.state.metrics['total_loss']
        data['g_loss'] = engine.state.metrics['g_loss']
        data['f_loss'] = engine.state.metrics['f_loss']
        data['p_loss'] = engine.state.metrics['p_loss']
        data['parsing_loss'] = engine.state.metrics['parsing_loss']
        # visualizer.plot_current_errors(engine.state.epoch, data)
        wandb.log(data, step=engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0 or engine.state.epoch > 110:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
            wandb.log({"mAP": mAP, "Rank-1": cmc[0]}, step=engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)


def deep_feature_cluster_for_each_identity(feats_I, pseudo_labels_paths_I, label, cfg):
    results, vis = dfc.clustering(cfg, feats_I, label)
    # print(pseudo_labels_paths_I[0])
    for i in range(cfg.CLUSTERING.PART_NUM):
        results = np.where(results == 100+i, i, results)

    N = feats_I.shape[0]
    for img_idx in range(N):
        pgt_img = Image.fromarray(results[img_idx].astype('uint8'))
        pgt_img.save(pseudo_labels_paths_I[img_idx])
    return vis


def deep_feature_cluster_for_each_identity_multi_process(arg):
    # sem.acquire()
    feats_I, pseudo_labels_paths_I, label, cfg = arg

    results, vis = dfc.clustering(cfg, feats_I, label)
    # print(pseudo_labels_paths_I[0])
    for i in range(cfg.CLUSTERING.PART_NUM):
        results = np.where(results == 100+i, i, results)

    N = feats_I.shape[0]
    for img_idx in range(N):

        pgt_img = Image.fromarray(results[img_idx].astype('uint8'))
        pgt_img.save(pseudo_labels_paths_I[img_idx])
    # sem.release()


def random_erase_human_part(img, p_target, cfg):
    mean = [0.4914, 0.4822, 0.4465]
    ratio = 1

    # expand to image size
    mask = p_target.reshape(p_target.shape[0], 1, p_target.shape[1], p_target.shape[2])
    mask = torch.nn.functional.interpolate(mask.float(), scale_factor=4, mode='nearest')
    mask = mask.reshape(mask.shape[0], mask.shape[2], mask.shape[3])

    # random erase one human part
    erase_part = random.randint(1, cfg.CLUSTERING.PART_NUM-1)
    torch.where(mask == erase_part, torch.tensor([0]), torch.tensor([1]))

    img[:, 0, :, :] = img[:, 0, :, :] * mask.float()
    img[:, 1, :, :] = img[:, 1, :, :] * mask.float()
    img[:, 2, :, :] = img[:, 2, :, :] * mask.float()

    torch.where(p_target ==erase_part, torch.tensor([0]), p_target)

    # torch.where(img[:, 0, :, :] == erase_part, mean[0], img[:, 0, ::])
    # torch.where(img[:, 1, :, :] == erase_part, mean[0], img[:, 0, ::])
    # torch.where(img[:, 2, :, :] == erase_part, mean[0], img[:, 0, ::])

    return img, p_target


def batch_erasing(x, target, epoch=-1):
    r1 = 0.5
    r2 = 1
    epochm = 70
    p = 1 / 3  # Oculusionreid p=1/3

    h, w = x.size()[-2:]
    area = h * w

    if epoch <= epochm:
        target_area = area * (0.1 + 0.4 * epoch / 120)  # easy-to-hard
    else:
        target_area = area * p

    for attempt in range(1000000):
        if math.sqrt(target_area) <= w:
            aspect_ratio = random.uniform(r1, r2)
            rh = int(round(math.sqrt(target_area * aspect_ratio)))
            rw = int(round(math.sqrt(target_area / aspect_ratio)))
        else:
            rw = w
            rh = int(round(target_area/rw))

        if rw <= w and rh <= h:
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            target[:, int(sx / 4):int((sx + rh) / 4), int(sy / 4):int((sy + rh) / 4)] = 0
            x[:, 0, sx:sx + rh, sy:sy + rw] = random.uniform(0, 1)
            x[:, 1, sx:sx + rh, sy:sy + rw] = random.uniform(0, 1)
            x[:, 2, sx:sx + rh, sy:sy + rw] = random.uniform(0, 1)
            break
    return x, target
