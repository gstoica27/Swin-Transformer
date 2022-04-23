# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
# import ffcv
# from ffcv.fields import IntField, RGBImageField
# from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
# from ffcv.loader import Loader, OrderOption
# from ffcv.transforms import RandomHorizontalFlip, Cutout, NormalizeImage, \
#     RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage, RandAugment
# from ffcv.transforms.common import Squeeze
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# CIFAR100_MEAN = IMAGENET_DEFAULT_MEAN
# CIFAR100_STD = IMAGENET_DEFAULT_STD


def build_loader(config):
    if config.FFCV:
        ffcv_data_loader_train, after_ffcv_transform = build_ffcv_dataloader(True, config)
        ffcv_data_loader_val, _ = build_ffcv_dataloader(False, config)
    else:
        config.defrost()
        dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
        config.freeze()
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
        dataset_val, _ = build_dataset(is_train=False, config=config)
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
            indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        if config.TEST.SEQUENTIAL:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False
            )

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    if config.FFCV:
        return None, None, ffcv_data_loader_train, ffcv_data_loader_val, mixup_fn, after_ffcv_transform
    else:
        return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn, None

def build_ffcv_dataloader(is_train, config):
    device = torch.device(f'cuda:{config.LOCAL_RANK}')
    label_pipeline = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(device, non_blocking=True)]
    image_pipeline = [RandomResizedCropRGBImageDecoder((224, 224))]

    is_imagenet = 'imagenet' in config.DATA.DATASET

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    CIFAR100_MEAN = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
    CIFAR100_STD = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    mean = IMAGENET_MEAN if is_imagenet else CIFAR100_MEAN
    std = IMAGENET_STD if is_imagenet else CIFAR100_STD

    # Add image transforms and normalization
    if is_train:
        image_pipeline.extend([
            RandomHorizontalFlip(flip_prob=0.5),
            RandAugment()
        ])
    image_pipeline.extend([
        ToTensor(),
        ToTorchImage(),
        ToDevice(device, non_blocking=True),
        NormalizeImage(mean, std, np.float32),
    ])

    if is_train:
        # hacky way to use a mixture of timm and ffcv transformations
        timm_transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        #rand_augment = timm_transform.transforms[2]
        rand_erasing = timm_transform.transforms[5]

        after_ffcv_transform = transforms.Compose([
            rand_erasing
        ])
    else:
        after_ffcv_transform = None

    prefix = 'train' if is_train else 'val'
    if is_imagenet:
        ffcv_dir = '/srv/share4/thearn6/datasets/ffcv_imagenet'
        ffcv_path = os.path.join(ffcv_dir, f'{prefix}_224_0.5_90.ffcv')
    else:
        ffcv_dir = '/srv/share4/thearn6/datasets/ffcv_cifar100'
        ffcv_path = os.path.join(ffcv_dir, f'{prefix}_224_0_raw.ffcv')

    is_distributed = True
    return Loader(ffcv_path,
        batch_size=config.DATA.BATCH_SIZE,    
        num_workers=config.DATA.NUM_WORKERS,
        order=ffcv.loader.OrderOption.RANDOM if is_distributed else ffcv.loader.OrderOption.QUASI_RANDOM,
        os_cache=is_distributed or not is_imagenet,
        drop_last=is_train,
        pipelines= {
            'image': image_pipeline,
            'label': label_pipeline
        },
        distributed=is_distributed
    ), after_ffcv_transform

def build_loader_tune(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"successfully build val dataset")

    num_tasks = 4 #dist.get_world_size()
    global_rank = 0#dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'cifar100':
        root = os.path.join(config.DATA.DATA_PATH)
        dataset = datasets.CIFAR100(root=root, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif config.DATA.DATASET == 'imagenet22K':
        raise NotImplementedError("Imagenet-22K will come soon.")
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    if 'imagenet' in config.DATA.DATASET:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    else:
        mean = CIFAR100_MEAN
        std = CIFAR100_STD

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
