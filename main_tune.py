# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import pdb
import os
from functools import partial
from pickletools import optimize
from shutil import SpecialFileError
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from config import get_config
from models import build_model
from data import build_loader_tune
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (load_checkpoint, load_finetunable_base, load_pretrained,
                    save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor)
from ray.tune.integration.torch import DistributedTrainableCreator

try:
    # noinspection PyUnresolvedReferences
    # from apex import amp
    import torch.cuda.amp as amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader_tune(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    if config.TUNE_PARAMETERS.APPLY_TUNE:
        tune_scheduler = ASHAScheduler(
            metric='loss',
            mode='min',
            max_t=10,
            grace_period=1,
            reduction_factor=2
        )
        tune_reporter = CLIReporter(
            metric_columns=["Val Loss", "Val Accuracy", "Training Epoch"]
        )
        # pdb.set_trace()
        tune_config = {
            'warmup_epochs': tune.choice(config.TUNE_PARAMETERS.WARMUP_EPOCHS),
            'weight_decay': tune.choice(config.TUNE_PARAMETERS.WEIGHT_DECAY),
            'base_lr': tune.choice(config.TUNE_PARAMETERS.BASE_LR),
            'warmup_lr': tune.choice(config.TUNE_PARAMETERS.WARMUP_LR),
            'min_lr': tune.choice(config.TUNE_PARAMETERS.MIN_LR),
        }

    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model)#, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)#, find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_valid_accuracy = 0.0
    max_valid_top_five = 0.0
    scaler = torch.cuda.amp.GradScaler()
    # Remove any pretraining/model loading flags since we are using tune environment
    config.defrost()
    config.TRAIN.AUTO_RESUME = False
    config.MODEL.RESUME = False
    config.MODEL.PRETRAINED = False
    config.freeze()

    def train_model(tune_config, checkpoint_dir=None):
        global max_valid_accuracy, max_valid_top_five, config
        whitelisted_param_names = None
        if config.FINETUNING.APPLY_FINETUNING:

            whitelisted_param_names = load_finetunable_base(config, model_without_ddp, logger)

            if config.FINETUNING.FREEZE_BASE:
                whitelisted_param_names = []
                for layer_idx, layer in enumerate(model_without_ddp.layers):
                    if layer_idx in config.MODEL.SWIN.REVERSE_ATTENTION_LOCATIONS:
                        whitelisted_param_names += [f'module.layers.{layer_idx}.' + i[0] for i in layer.named_parameters()]
                whitelisted_param_names = set(whitelisted_param_names)
    
        optimizer = build_optimizer(config, model, whitelisted_params=whitelisted_param_names, tune_config=tune_config)
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train), tune_config=tune_config)

        if config.TRAIN.AUTO_RESUME:
            resume_file = auto_resume_helper(config.OUTPUT)
            if resume_file:
                if config.MODEL.RESUME:
                    logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
                config.defrost()
                config.MODEL.RESUME = resume_file
                config.freeze()
                logger.info(f'auto resuming from {resume_file}')
            else:
                logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

        if config.MODEL.RESUME:
            max_valid_accuracy, missing_tuple = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, scaler)
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            if config.EVAL_MODE:
                return

        if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
            load_pretrained(config, model_without_ddp, logger)
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

        if config.THROUGHPUT_MODE:
            throughput(data_loader_val, model, logger)
            return
    
        logger.info("Start training")
        start_time = time.time()
        for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
            data_loader_train.sampler.set_epoch(epoch)

            train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, scaler=scaler)
            if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_valid_accuracy, optimizer, lr_scheduler, logger, scaler, use_tune=True, checkpoint_dir=checkpoint_dir)

            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: Top1: {acc1:.2f}% | Top5: {acc5:.2f}")
            max_valid_top_five = max_valid_top_five if max_valid_accuracy > acc1 else acc5
            max_valid_accuracy = max(max_valid_accuracy, acc1)
            
            logger.info(f'Max valid accuracy | Top1: {max_valid_accuracy:.2f}% | Top5: {max_valid_top_five:.2f}')
            tune.report(loss=(loss), accuracy=acc1)
        # if data_loader_test is not None:
        #     test_acc1, test_acc5, test_loss = validate(config, data_loader_test, model)
        #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: Top1: {test_acc1:.2f}% | Top5: {test_acc5:.2f}")
        #     update_test = max_valid_accuracy == acc1
        #     test_accuracy_at_best_valid = test_acc1 if update_test else test_accuracy_at_best_valid
        #     test_top5_at_best_valid = test_acc5 if update_test else test_top5_at_best_valid
        #     logger.info(f'Max test accuracy: Top1: {test_accuracy_at_best_valid:.2f}% | Top5: {test_top5_at_best_valid: .2f}')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))
    
    distributed_trainer = DistributedTrainableCreator(
        partial(train_model),
        # use_gpu=True,
        num_workers=4,
        num_gpus_per_worker=1,
        num_cpus_per_worker=6
    )

    result = tune.run(
        distributed_trainer,
        resources_per_trial={'cpu': 24, 'gpu': 4},
        config=tune_config,
        num_samples=1,
        scheduler=tune_scheduler,
        progress_reporter=tune_reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if config.AMP_OPT_LEVEL != 'O0':
            with amp.autocast():
                outputs = model(samples, use_amp=True)
        else:
            outputs = model(samples, use_amp=False)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if config.AMP_OPT_LEVEL != 'O0':
                with amp.autocast():
                    loss = criterion(outputs, targets)
                    loss = loss / config.TRAIN.ACCUMULATION_STEPS
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaler.scale(loss).backward()
                    # scaled_loss.backward()
                    if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                        if config.TRAIN.CLIP_GRAD:
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                        else:
                            grad_norm = get_grad_norm(model.parameters())
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        lr_scheduler.step_update(epoch * num_steps * idx)
       
                        loss_meter.update(loss.item(), targets.size(0))
                        norm_meter.update(grad_norm)
                        batch_time.update(time.time() - end)
                # if config.TRAIN.CLIP_GRAD:
                #     grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                # else:
                #     grad_norm = get_grad_norm(amp.master_params(optimizer))
            
            else:
                #  with torch.autograd.detect_anomaly():
                # pdb.set_trace()
                loss = criterion(outputs, targets)
                if loss.isnan():
                    pdb.set_trace()
                    del loss
                    continue
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    # pdb.set_trace()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step_update(epoch * num_steps + idx)

                    loss_meter.update(loss.item(), targets.size(0))
                    norm_meter.update(grad_norm)
                    batch_time.update(time.time() - end)
        else:
            if config.AMP_OPT_LEVEL != "O0":
                with amp.autocast():
                    loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                    
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step_update(epoch * num_steps * idx)
            else:
                loss = criterion(outputs, targets)
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                lr_scheduler.step_update(epoch * num_steps * idx)

            loss_meter.update(loss.item(), targets.size(0))
            norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            optimizer.zero_grad()

            # optimizer.zero_grad()
            # if config.AMP_OPT_LEVEL != "O0":
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            #     if config.TRAIN.CLIP_GRAD:
            #         grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
            #     else:
            #         grad_norm = get_grad_norm(amp.master_params(optimizer))
            # else:
            #     loss.backward()
            #     if config.TRAIN.CLIP_GRAD:
            #         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            #     else:
            #         grad_norm = get_grad_norm(model.parameters())
            # optimizer.step()
            # lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        # loss_meter.update(loss.item(), targets.size(0))
        # norm_meter.update(grad_norm)
        # batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images, use_amp=False)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images, use_amp=False)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images, use_amp=False)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    # else:
    #     rank = -1
    #     world_size = -1
    # torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

    seed = config.SEED #+ dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR #* config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR #* config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR #* config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    config.defrost()
    config.LOCAL_RANK = 0
    config.freeze()
    main(config)
