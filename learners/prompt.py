from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function
import clip

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        self.temperature = learner_config['temperature']
        self.memory = learner_config['memory']
        super(Prompt, self).__init__(learner_config)


    def update_model(self, inputs, targets, text_encoder=None, trans_dict=None):

        # logits
        logits, prompt_loss = self.model(inputs, train=True, text_encoder=text_encoder) 

        if text_encoder is None:
            logits = logits[:,:self.valid_out_dim] 
            # ce with heuristic
            if self.memory == 0:
                logits[:,:self.last_valid_out_dim] = -float('inf')
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            total_loss = self.criterion(logits, targets.long(), dw_cls)

            # ce loss
            total_loss = total_loss + prompt_loss.sum()
        

        else:

            text_label = trans_dict
            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text_label]).cuda()
                if text_encoder.dp:
                    text_features = text_encoder.model.module.encode_text(text_inputs)
                else:
                    text_features = text_encoder.model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features  = logits / logits.norm(dim=-1, keepdim=True)
            


            # Calculating the Loss
            logits = (image_features.half() @ text_features.T.half()) / self.temperature 


            logits = logits[:,:self.valid_out_dim] 
            # ce with heuristic
            logits[:,:self.last_valid_out_dim] = -float('inf') 
            # print("self.last_valid_out_dim:",self.last_valid_out_dim)
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
            total_loss = self.criterion(logits, targets.long(), dw_cls)
            # print("self.dw_k:", self.dw_k)

            # ce loss
            total_loss = total_loss + prompt_loss.sum()






        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers 
    def init_optimizer(self):

        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


class MSPrompt(Prompt):

    def __init__(self, learner_config):
        super(MSPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'msp', prompt_param=self.prompt_param, clip_encoder=cfg['use_clip_encoder'], cfg=self.config) # 从 model 文件夹下找到 zoo 文件，获取对应模型vit_pt_imnet，返回vitzoo，返回vit中的visiontransformer
        return model    
