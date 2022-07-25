# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from multiprocessing.sharedctypes import Value
import pdb
import os
from pickletools import optimize
from shutil import SpecialFileError
import time
import random
import argparse
import datetime
import numpy as np
from operator import methodcaller
import kornia as K
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data.cross_build import build_loader
# from data.augmentation_selector import choose_augmentation, perform_augmentation
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_finetunable_base, load_pretrained, \
save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, setup_finetuning_regime, do_stop_finetuning

try:
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
    parser.add_argument('--num-workers', type=int, help="Number of data loading threads")
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--ffcv', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('secondary_eval_aug', default=None, type=str, help='secondary image augmentation')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    return parser


def main(config, logger):
    # pdb.set_trace()
    data_loader_train, dataset_val, _, data_loader_val, mixup_fn, after_ffcv_transform = build_loader(config)

    if config.FFCV:
        print("Using ffcv")

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True
    )
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
    prev_max_accuracy = 0.0
    max_valid_accuracy = 0.0
    max_valid_top_five = 0.0
    scaler = GradScaler(enabled=config.NATIVE_AMP)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

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

    load_pretrained(config, model_without_ddp, logger)
    squential_results, parallel_results = evaluate_configuration(config, data_loader_val, model, model_without_ddp, logger)
    # logger.info(f"Accuracy of the network on the {len(data_loader_val) * args.batch_size} test images: {acc1:.1f}%")


def update_meters(acc1, acc5, loss, batch_time, size, meter_dict):
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)
            meter_dict['loss'].update(loss.item(), size)
            meter_dict['acc1'].update(acc1.item(), size)
            meter_dict['acc5'].update(acc5.item(), size)
            meter_dict['batch_time'].update(batch_time)


@torch.no_grad()
def evaluate_configuration(config, data_loader, model, model_without_ddp, logger):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    # batch_time = AverageMeter()
    # loss_meter = AverageMeter()
    # acc1_meter = AverageMeter()
    # acc5_meter = AverageMeter()

    sequential_meters = {
        'batch_time': AverageMeter(),
        'loss': AverageMeter(),
        'acc1': AverageMeter(),
        'acc5': AverageMeter(),
    }
    parallel_meters = {
        'batch_time': AverageMeter(),
        'loss': AverageMeter(),
        'acc1': AverageMeter(),
        'acc5': AverageMeter(),
    }
    baseline_meters = {
        'batch_time': AverageMeter(),
        'loss': AverageMeter(),
        'acc1': AverageMeter(),
        'acc5': AverageMeter(),
    }

    # secondary_augmentation = choose_augmentation(config.SECOND_EVAL_AUG)

    end = time.time()
    for idx, (images, target, _, secondary_images) in tqdm(enumerate(data_loader)):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        secondary_images = secondary_images.cuda(non_blocking=True)
        
        # pdb.set_trace()
        # pil_images = transforms.ToPILImage()(raw_images).convert("RGB")
        # secondary_images = perform_augmentation(raw_images).cuda(non_blocking=True)

        # compute output
        with autocast(enabled=config.NATIVE_AMP):
            B, C, H, W = images.shape
            # Sequential Batch
            sequential_input = torch.cat((images, secondary_images), dim=0)
            model_without_ddp.cross_attention_locations = []
            aggregated_output = model(sequential_input, use_amp=False, combine_streams=False, target=target)
            orig_output, secondary_output = torch.split(aggregated_output, split_size_or_sections=B, dim=0)
            sequential_output = (orig_output + secondary_output) / 2
            sequential_loss = criterion(sequential_output, target)
            # Parallel Batch
            model_without_ddp.cross_attention_locations = [0, 1, 2, 3]
            parallel_output = model(images, secondary_images, use_amp=False, combine_streams=True, target=target)
            parallel_loss = criterion(parallel_output, target)
            # Baseline Batch
            model_without_ddp.cross_attention_locations = []
            baseline_output = model(images, use_amp=False, combine_streams=False, target=target)
            baseline_loss = criterion(baseline_output, target)

        # pdb.set_trace()
        sequential_acc1, sequential_acc5 = accuracy(sequential_output, target, topk=(1, 5))
        parallel_acc1, parallel_acc5 = accuracy(parallel_output, target, topk=(1, 5))
        baseline_acc1, baseline_acc5 = accuracy(baseline_output, target, topk=(1, 5))
        batch_time = time.time() - end
        end = time.time()

        update_meters(
            acc1=sequential_acc1, acc5=sequential_acc5, loss=sequential_loss, 
            batch_time=batch_time, meter_dict=sequential_meters, size=target.shape[0]
        )
        update_meters(
            acc1=parallel_acc1, acc5=parallel_acc5, loss=parallel_loss, 
            batch_time=batch_time, meter_dict=parallel_meters, size=target.shape[0]
        )
        update_meters(
            acc1=baseline_acc1, acc5=baseline_acc5, loss=baseline_loss, 
            batch_time=batch_time, meter_dict=baseline_meters, size=target.shape[0]
        )
    # pdb.set_trace()
    logger.info(f' Baseline Accuracy | * Acc@1 {baseline_meters["acc1"].avg:.3f} Acc@5 {baseline_meters["acc5"].avg:.3f}')
    logger.info(f' Sequential Accuracy | * Acc@1 {sequential_meters["acc1"].avg:.3f} Acc@5 {sequential_meters["acc5"].avg:.3f}')
    logger.info(f' Parallel Accuracy | * Acc@1 {parallel_meters["acc1"].avg:.3f} Acc@5 {parallel_meters["acc5"].avg:.3f}')
    
    lambda_params = {}
    for name, params in model.named_parameters():
        if 'lambda' in name:
            lambda_params[name] = (params.detach().cpu().numpy(), params.requires_grad)
    if len(lambda_params) > 0:
        logger.info('======== Lambdas ========')
        for name, param in lambda_params.items():
            logger.info(f'{name}: {param}')
        logger.info('=========================')
    
    return sequential_meters, parallel_meters

def run(args):
    config = get_config(args)

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = config.LOCAL_RANK
        dist_url = 'env://'
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    elif 'SLURM_PROCID' in os.environ:
        
        # rank = int(os.environ['SLURM_PROCID'])
        # gpu = args.rank % torch.cuda.device_count()
        
        rank = args.rank
        gpu = args.gpu
        world_size = args.world_size
        dist_url = args.dist_url
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE  * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
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
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config, logger)


if __name__ == '__main__':
    parser = parse_option()
    args, _ = parser.parse_known_args()
    run(args)
