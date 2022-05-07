# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
from tabnanny import check
import torch
import torch.nn as nn
import torch.distributed as dist
import pdb
# from ray import tune

# from ray.tune.integration.torch import (DistributedTrainableCreator,
#                                         distributed_checkpoint_dir)
try:
    # noinspection PyUnresolvedReferences
    # from apex import amp
    import torch.cuda.amp as amp
except ImportError:
    amp = None



def load_checkpoint(config, model, optimizer, lr_scheduler, logger, scaler):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    # pdb.set_trace()
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        # pdb.set_trace()
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            scaler.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, msg

def update_model_parameter_names(param_names):
    updated_params = []
    for param_name in param_names:
        if 'module' not in param_name:
            prefix = 'module.layers.'
        else:
            prefix = ''
        updated_params.append(prefix + param_name)
    return updated_params

def set_requires_grad(m, requires_grad):
    # from https://discuss.pytorch.org/t/requires-grad-doesnt-propagate-to-the-parameters-of-the-module/9979/6
    if hasattr(m, 'weight') and m.weight is not None:
        m.weight.requires_grad_(requires_grad)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.requires_grad_(requires_grad)
    else:
        m.requires_grad_(requires_grad)

def setup_finetuning_regime(config, model):
    if config.FINETUNING.FREEZE_BASE:
        training_params = []
        for name, param in model.named_parameters():
            param.requires_grad = False
        for layer_idx, layer in enumerate(model.layers):
            if layer_idx in config.MODEL.SWIN.REVERSE_ATTENTION_LOCATIONS:
                for block_idx, block in enumerate(layer.blocks):
                    training_params += block.attn.attention_parameters + block.attn.reverse_parameters + block.attn.output_parameters
                    if config.FINETUNING.TRAINABLES == 'block':
                        training_params += block.non_attention_parameters
                    if config.MODEL.SWIN.ALTERED_ATTENTION.TYPE == 'shared_forward_and_reverse':
                            training_params += block.attn.forward_parameters
        # pdb.set_trace()
        for param in training_params:
            set_requires_grad(param, True)
        # pdb.set_trace()
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
    
def do_stop_finetuning(config, curr_accuracy, curr_epoch):
    # pdb.set_trace()
    are_finetuning = config.FINETUNING.APPLY_FINETUNING
    diff_bound = config.FINETUNING.DIFF_BOUND
    stop_epoch = config.FINETUNING.STOP_EPOCH
    compare_accuracy = config.FINETUNING.COMP_ACCURACY
    # pdb.set_trace()
    if are_finetuning and diff_bound is not None and curr_accuracy - compare_accuracy >= diff_bound:
        return True
    elif are_finetuning and stop_epoch is not None and curr_epoch >= stop_epoch:
        return True
    return False

def load_finetunable_base(config, model, logger):#, optimizer, lr_scheduler, scaler):
    logger.info(f"==============> Resuming form {config.FINETUNING.BASE_MODEL}....................")
    if config.FINETUNING.BASE_MODEL.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.FINETUNING.BASE_MODEL, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.FINETUNING.BASE_MODEL, map_location='cpu')

    # if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #     # pdb.set_trace()
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
    #     if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
    #         scaler.load_state_dict(checkpoint['amp'])
    # checkpoint_params = update_model_parameters(checkpoint=checkpoint)
    # pdb.set_trace()
    if config.FINETUNING.BASELINE_USES_SWIN_CFG:
        new_dict = convert_swin_to_biswin(checkpoint['model'])
    else:
        new_dict = checkpoint['model']
    # aug_dict = {}
    # for k in new_dict.keys():
    #     if 'selection_lambda' not in k: 
    #         aug_dict[k] = new_dict[k]
    # new_dict = aug_dict
    missing_tuple = model.load_state_dict(new_dict, strict=False)
    logger.info(missing_tuple)

    if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
            logger.info('ACCURACY OF BASE MODEL: {:3f}'.format(max_accuracy))
    return max_accuracy


def convert_swin_to_biswin(checkpoint_base):
    checkpoint = {}
    for name, param in checkpoint_base.items():
        if 'qkv' in name:
            enc3 = param.shape[0]
            split_loc = int(2 * enc3 // 3)
            qk = param[:split_loc] 
            v = param[split_loc:]
            new_qk_name = name.replace('qkv', 'qk')
            new_v_name = name.replace('qkv', 'forward_v')
            checkpoint[new_qk_name] = qk
            checkpoint[new_v_name] = v
            # del checkpoint[name]
        elif 'attn.proj' in name:
            new_name = name.replace('attn.proj', 'attn.output_proj')
            checkpoint[new_name] = param
            # del checkpoint[name]
        else:
            checkpoint[name] = param
    return checkpoint

           


def load_pretrained(config, model, logger=None):
    if logger is not None:
        logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            if logger is not None:
                logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            if logger is not None:
                logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            if logger is not None:
                logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            if logger is not None:
                logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    if logger is not None:
        logger.warning(msg)
        logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, scaler, use_tune=False, checkpoint_dir=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config
                  }
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = scaler.state_dict()
        # save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if use_tune:
        logger.info(f"Saving to tune...")
        with distributed_checkpoint_dir(epoch) as checkpoint_dir:
            tune_save_path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimizer.state_dict()), tune_save_path)
            logger.info(f"Saved to {tune_save_path}")
            


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
