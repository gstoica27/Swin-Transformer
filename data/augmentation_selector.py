import torch
import numpy as np
import pdb

# from torchvision import datasets, transforms
from torchvision.transforms import (
    ToPILImage, 
    RandomHorizontalFlip, 
    RandomVerticalFlip, 
    ColorJitter, 
    ToTensor, 
    Normalize, 
    Compose,
    CenterCrop,
    Resize,
)
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.data import create_transform
from timm.data.auto_augment import _RAND_TRANSFORMS, AugmentOp
from timm.data.transforms import (
    RandomResizedCropAndInterpolation, 
    _pil_interp, 
)
from timm.data.random_erasing import RandomErasing


DEFAULT_AUGMENTATIONS = {
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'Posterize',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    # 'Cutout'  # NOTE I've implement this as random erasing separately
}

OTHER_AUGMENTATIONS = {
    'resize': RandomResizedCropAndInterpolation(224, scale=(.08, 1.0), ratio=(3./4., 4./3.), interpolation='bicubic'),
    'hflip': RandomHorizontalFlip(p=1.0),
    'vflip': RandomVerticalFlip(p=1.0),
    'jitter': ColorJitter(0.4),
}

POST_TENSOR_AUGMENTATIONS = {
    'erase': RandomErasing(1.0, mode='pixel', max_count=1, num_splits=0, device='cpu')
}

CIFAR100_MEAN = np.array([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
CIFAR100_STD = np.array([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])


DEFAULT_AUG_PARAMS = dict(
    magnitude_std=0.5,
    magnitude=9,
    increase_magnitude_severity=True,
    translate_const=224,
    img_mean=tuple([min(255, round(255 * x)) for x in CIFAR100_MEAN]),
    interpolation=_pil_interp('bicubic')
)

def choose_augmentation(chosen_transform):
    transforms = []
    # pdb.set_trace()
    if chosen_transform in DEFAULT_AUGMENTATIONS:
        # We will always augment image with chosen function
        transform = AugmentOp(
            name=chosen_transform, 
            magnitude=DEFAULT_AUG_PARAMS['magnitude'], 
            hparams=DEFAULT_AUG_PARAMS,
            prob=1.0
        )
        transforms.append(transform)
    elif chosen_transform in OTHER_AUGMENTATIONS:
        transform = OTHER_AUGMENTATIONS[chosen_transform]
        transforms.append(transform)
    
    transforms += [
        Resize(int((256 / 224) * 224), interpolation=_pil_interp('bicubic')),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=torch.tensor(CIFAR100_MEAN), std=torch.tensor(CIFAR100_STD))
    ]

    if chosen_transform in POST_TENSOR_AUGMENTATIONS:
        transforms.append(
            POST_TENSOR_AUGMENTATIONS[chosen_transform]
        )
    
    return Compose(transforms)
    
def perform_augmentation(images, aug):
    return aug(images)
