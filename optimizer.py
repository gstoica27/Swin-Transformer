# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim
import pdb

def build_optimizer(config, model, whitelisted_params=None, tune_config=None):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}

    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords, whitelisted_params)
    # pdb.set_trace()
    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None

    weight_decay = config.TRAIN.WEIGHT_DECAY
    base_lr = config.TRAIN.BASE_LR
    if tune_config is not None:
        if 'weight_decay' in tune_config:
            weight_decay = tune_config['weight_decay']
        if 'base_lr' in tune_config:
            base_lr = tune_config['base_lr']
    # pdb.set_trace()

    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=base_lr, weight_decay=weight_decay)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=base_lr, weight_decay=weight_decay)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=(), whitelisted_params=None):
    has_decay = []
    no_decay = []

    optimized_param_names = []
    for name, param in model.named_parameters():
        if whitelisted_params is not None and name not in whitelisted_params:
            continue # parameters are not whitelisted
        elif not param.requires_grad:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            # print(f'Name: {name} | shape: {param.shape}')
            # pdb.set_trace()
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
        optimized_param_names.append(name)
    # pdb.set_trace()
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
