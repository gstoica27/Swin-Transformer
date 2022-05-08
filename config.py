# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
from numpy import True_
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.SUMMARY_TYPE = 'none'
_C.MODEL.SWIN.SUMMARY_LAYERS = []
_C.MODEL.SWIN.BIDIRECTIONAL_ATTENTION_LOCATIONS = []

# _C.MODEL.SWIN.ALTERED_ATTENTION = CN()
# _C.MODEL.SWIN.ALTERED_ATTENTION.TYPE = 'forward'
# _C.MODEL.SWIN.ALTERED_ATTENTION.REDUCE_REVERSE = False
# _C.MODEL.SWIN.ALTERED_ATTENTION.REVERSE_ACTIVATION = 'none'
# _C.MODEL.SWIN.ALTERED_ATTENTION.HYPERNETWORK_BIAS = True
# _C.MODEL.SWIN.ALTERED_ATTENTION.PROJECT_VALUES = True
# _C.MODEL.SWIN.ALTERED_ATTENTION.PROJECT_INPUT = True
# _C.MODEL.SWIN.ALTERED_ATTENTION.VALUE_IS_INPUT = False
# _C.MODEL.SWIN.ALTERED_ATTENTION.TRANSPOSE_SOFTMAX = True
# _C.MODEL.SWIN.ALTERED_ATTENTION.ACTIVATE_HYPER_WEIGHTS = False
# _C.MODEL.SWIN.ALTERED_ATTENTION.GEN_INDIV_HYPER_WEIGHTS = False
# _C.MODEL.SWIN.ALTERED_ATTENTION.ACTIVATE_INPUT = True
# _C.MODEL.SWIN.ALTERED_ATTENTION.SINGLE_WEIGHT_MATRIX = False
# _C.MODEL.SWIN.ALTERED_ATTENTION.ENFORCE_ORTHONOGALITY = False
# _C.MODEL.SWIN.ALTERED_ATTENTION.WEIGH_DIRECTIONS = False
# _C.MODEL.SWIN.ALTERED_ATTENTION.SELECTION_LAMBDA_FORM = 'scalar'

# Finetuning instructions
_C.FINETUNING = CN()
_C.FINETUNING.APPLY_FINETUNING = False
_C.FINETUNING.BASE_MODEL = None
_C.FINETUNING.FREEZE_BASE = False
_C.FINETUNING.STOP_EPOCH = None
_C.FINETUNING.DIFF_BOUND = None
_C.FINETUNING.TRAINABLES = 'head'
_C.FINETUNING.COMP_ACCURACY = None
_C.FINETUNING.BASELINE_USES_SWIN_CFG = False

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# Below are specifical for CSAM
_C.MODEL.CSAM = CN()
_C.MODEL.CSAM.APPROACH_NAME = "Three_unmasked"
_C.MODEL.CSAM.POS_EMB_DIM = 0
_C.MODEL.CSAM.SOFTMAX_TEMP = 1
_C.MODEL.CSAM.PADDING = 'same'
_C.MODEL.CSAM.STRIDE = 1
_C.MODEL.CSAM.APPLY_STOCHASTIC_STRIDE = False
_C.MODEL.CSAM.USE_RESIDUAL_CONNECTION = False
_C.MODEL.CSAM.SUFFIX = ''
_C.MODEL.CSAM.VALUE_DIM = -1
_C.MODEL.CSAM.RANDOM_K = -1
_C.MODEL.CSAM.FORGET_GATE_TYPE = "sigmoid"
_C.MODEL.CSAM.SIMILARITY_METRIC = "cosine_similarity"
_C.MODEL.CSAM.INJECTION_LOCATIONS = [0]
_C.MODEL.CSAM.OUTPUT_SCALAR = 2.
_C.MODEL.CSAM.FILTER_SIZE = 3
_C.MODEL.CSAM.OUTPUT_PROJECTION = False
_C.MODEL.CSAM.ADD_BATCH_NORM = False
_C.MODEL.CSAM.ADD_LAYER_NORM = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4 #(0.00015) / 8
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.ENFORCE_ATTENTION_ORTHONOGALITY = False
_C.TRAIN.ORTHOGONALITY_LAMBDA = 1e-5
_C.TRAIN.ALL_ORTHOGONALITY_LAMBDA = 1e-5

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

_C.TUNE_PARAMETERS = CN()
_C.TUNE_PARAMETERS.APPLY_TUNE = False
_C.TUNE_PARAMETERS.WARMUP_EPOCHS = None
_C.TUNE_PARAMETERS.WEIGHT_DECAY = None
_C.TUNE_PARAMETERS.BASE_LR = None
_C.TUNE_PARAMETERS.WARMUP_LR = None
_C.TUNE_PARAMETERS.MIN_LR = None

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.num_workers:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    config.NATIVE_AMP = args.native_amp
    config.FFCV = args.ffcv

    # set local rank for distributed training
    if hasattr(args, "submitit_run"):
        config.LOCAL_RANK = args.gpu 
    else:
        config.LOCAL_RANK = args.local_rank 

    if config.MODEL.TYPE == 'csam':
        config.TAG = name_model(config=config.MODEL.CSAM)
    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


# def name_model(config):
#     model_name = 'CSAM_Approach{}_AddPosEmb{}_Temp{}_StochStride{}_Stride{}_Residual{}'.format(
#         config.APPROACH_NAME, 
#         config.ADD_POS_EMB, 
#         config.SOFTMAX_TEMP, 
#         config.APPLY_STOCHASTIC_STRIDE, 
#         config.STRIDE,
#         config.USE_RESIDUAL_CONNECTION
#     )
#     return model_name

def name_model(config):
   
    model_name = 'CSAM_Approach{}_AddPosEmb{}_Temp{}_StochStride{}_Stride{}_Residual{}_Forget{}_SimMetr{}'.format(
        config.APPROACH_NAME, 
        config.POS_EMB_DIM, 
        config.SOFTMAX_TEMP, 
        config.APPLY_STOCHASTIC_STRIDE, 
        config.STRIDE,
        config.USE_RESIDUAL_CONNECTION,
        config.FORGET_GATE_NONLINEARITY,
        config.SIMILARITY_METRIC,
    )
    return model_name
