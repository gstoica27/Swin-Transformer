import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_checkpoint, load_finetunable_base
from operator import methodcaller


class FineTuningWrapper(object):
    def __init__(self, config, model, logger) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.logger = logger
        self.load_finetunable_base()
        self.stop_flag = False
        self.tuning_regime = self.config.FINETUNING.REGIME
    
    def load_finetunable_base(self):
        self.base_accuracy = load_finetunable_base(self.config, self.model, self.logger)

    def initialize_finetune_parameters(self):
        if self.tuning_regime == 'freeze_base':
            self.stop_epoch = self.config.FINETUNING.REGIME_PARAMETERS['stop_epoch']
            self.trainables = self.config.FINETUNING.REGIME_PARAMETERS['trainables']
            self.in_last_stage = True
        elif self.tuning_regime == 'train_all':
            self.stop_epoch = self.config.FINETUNING.REGIME_PARAMETERS['stop_epoch']
            self.in_last_stage = True
        elif self.tuning_regime == 'two_stage':
            self.unfreeze_epoch = self.config.FINETUNING.REGIME_PARAMETERS['unfreeze_epoch']
            self.unfreeze_differences = self.config.FINETUNING.REGIME_PARAMETERS['unfreeze_difference']
            self.stage_one_trainables = self.config.FINETUNING.REGIME_PARAMETERS['stage_one_trainables']
            self.stop_epoch = self.config.FINETUNING.REGIME_PARAMETERS['stop_epoch']
            self.in_last_stage = False
        
    def determine_trainable_search_fns(self):
        if self.tuning_regime == 'train_all':
            search_fns = [
                lambda x: True
            ]
        elif self.tuning_regime == 'freeze_base':
            if self.trainables == 'block':
                search_fns = [
                    lambda x: 'attention_parameters' in x, 
                    lambda x: 'reverse_parameters' in x
                ]
            elif self.trainables == 'head':
                search_fns = [
                    lambda x: 'reverse_parameters' in x
                ]
        elif self.tuning_regime == 'two_stage':
            if self.in_last_stage:
                search_fns = [lambda x: True]
            elif self.stage_one_trainables == 'block':
                search_fns = [
                    lambda x: 'attention_parameters' in x, 
                    lambda x: 'reverse_parameters' in x
                ]
            elif self.stage_one_trainables == 'head':
                search_fns = [
                    lambda x: 'reverse_parameters' in x
                ]
        else:
            raise ValueError('Unknown finetuning regime')
        return search_fns
        
    def determine_trainable_parameters(self, search_fns):
        trainable_params = []
        for name, _ in self.model.named_parameters():
            param_is_valid = list(
                map(methodcaller('__call__', name), search_fns)
            )
            if param_is_valid:
                trainable_params.append(name)
        return trainable_params
    
    def determine_optimization_parameters(self):
        search_fns = self.determine_trainable_search_fns()
        return self.determine_trainable_parameters(search_fns)
    
    def check_tuning_completeness(self, accuracy, epoch):
        if self.tuning_regime == 'train_all' and epoch >= self.stop_epoch:
            return True
        elif self.tuning_regime == 'freeze_base' and epoch >= self.stop_epoch:
            return True
        elif ((self.tuning_regime == 'two_stage' and not self.in_last_stage) and 
        (abs(accuracy - self.base_accuracy) <= self.))
            