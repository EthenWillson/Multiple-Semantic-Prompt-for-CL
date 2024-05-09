import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners
import clip
import time

# class CLIPEncoder:
#     def __init__(self, args):
#         if args.txt_encoder == 'CLIP':
#             self.dataset_name = args.dataset
#             self.model_name = args.txt_encoder
#             # self.model , self.preprocess = clip.load('ViT-L/14', "cuda") # ViT-B/32
#             self.model , self.preprocess = clip.load('ViT-B/32', "cuda") # ViT-B/32
#             self.dp = False
    
#     def cuda(self, args):
#         torch.cuda.set_device(args.gpuid[0])
#         self.model = self.model.cuda()
#         # Multi-GPU
#         if len(args.gpuid) > 1:
#             self.model = torch.nn.DataParallel(self.model, device_ids=args.gpuid, output_device=args.gpuid[0])
#             self.dp = True
        
#         return self


class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.args = args
        # 是否进行语义实验：初始化
        self.semantic_flag = False
        
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'DomainNet':
            Dataset = dataloaders.iDOMAIN_NET
            num_classes = 345
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'CUB200':
            Dataset = dataloaders.myCUB200
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'MNIST':
            Dataset = dataloaders.iMNIST
            num_classes = 10
            self.dataset_size = [28,28,1]
            self.top_k = 1
        elif args.dataset == 'CIFAR100_semantic':
            Dataset = dataloaders.CIFAR100_semantic
            num_classes = 10
            self.dataset_size = [32,32,3]
            args.other_split_size = 6
            args.first_split_size = 10
            self.semantic_flag = True
            
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task): # 计算多少个tasks
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet) # transform有问题
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, rand_split=args.rand_split, validation=args.validation) # 实例化
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        # 获取label序号和文本标签对应关系
        self.trans_dict = self.train_dataset.get_trans_dict()
        if self.semantic_flag:
            self.semantic_trans_dict = self.train_dataset.get_trans_dict(semantic_exp=True)
        else:
            self.semantic_trans_dict = None

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param],
                        'temperature': args.temperature,
                        'use_clip_encoder': args.use_clip_encoder,
                        'clip_label_dict': self.trans_dict,
                        'semantic_label_dict':self.semantic_trans_dict,
                        'args':args,
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name

        # CLIP 编码器
        # if args.dont_use_clip_encoder:
        #     self.clip_encoder = CLIPEncoder(args)
        #     # Multi-GPU
        #     if len(self.learner_config['gpuid']) > 1:
        #         self.clip_encoder.cuda(args)
        # else:
        #     self.clip_encoder = None

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config) # 这里创建模型，在learner下的prompt.py下的CODAPrompt

        

    # 每个任务推理
    def task_eval(self, t_index, local=False, task='acc', text_encoder=None, trans_dict=None, semantic_train=None):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval 
        if semantic_train == 'train': # 当进行语义实验时
            self.test_dataset.load_dataset(t_index, train=True, semantic_exp=True) # semantic_exp=True, train=True时只加载学过的类
        elif semantic_train == 'test':
            self.test_dataset.load_dataset(t_index, train=False, semantic_exp=True) # semantic_exp=True, train=True时只加载没学过的类
        else:
            self.test_dataset.load_dataset(t_index, train=True) # train=True时只加载当前任务的数据

            # self.train_dataset.load_dataset(t_index, train=True) # train=True时只加载当前任务的数据
        
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers) 
        # test_loader  = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers) 



        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task, text_encoder=text_encoder, trans_dict=trans_dict, semantic_train=semantic_train) # 在learner下的prompt，转default.py的NormalNN
        else:
            return self.learner.validation(test_loader, task_metric=task, text_encoder=text_encoder, trans_dict=trans_dict, semantic_train=semantic_train)

    def train(self, avg_metrics):
    
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index 
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task 
            task = self.tasks_logits[i]
            if self.oracle_flag: 
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config) 
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False) 
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time, total_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir=model_save_dir, val_loader=test_loader, text_encoder=None, trans_dict=self.trans_dict) # 模型训练，在learner/defualt.py中，转prompt.py的update_model
            # avg_train_time = self.learner.learn_batch(test_loader, self.train_dataset, model_save_dir=model_save_dir, val_loader=test_loader, text_encoder=None, trans_dict=self.trans_dict) # 模型训练，在learner/defualt.py中，转prompt.py的update_model

            # save model
            self.learner.save_model(model_save_dir)
            
            # evaluate acc
            acc_table = []
            acc_table_ssl = []
            self.reset_cluster_labels = True
            if self.semantic_flag: 
                for j in range(i+1):
                    acc_table.append(self.task_eval(j, text_encoder=None, trans_dict=self.trans_dict, semantic_train='train'))
            else:
                for j in range(i+1):
                    acc_table.append(self.task_eval(j, text_encoder=None, trans_dict=self.trans_dict))
            temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # save temporary acc results
            for mkey in ['acc']:
                save_file = temp_dir + mkey + '.csv'
                np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')  

            if total_train_time is not None: avg_metrics['time']['global'][i] = total_train_time

        return avg_metrics 
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1): 
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name] 
            avg_acc_history[i] = cls_acc_sum / (i + 1) 

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task): 

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model 加载模型
            if not self.args.semantic_exp_pure_clip:
                model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
                self.learner.task_count = i 
                self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
                self.learner.pre_steps()
                self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True

            if self.semantic_flag: 
                # for j in range(i+1):
                val_name = self.task_names[i]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(i, text_encoder=None, trans_dict=self.trans_dict, semantic_train = 'test')
            else:
                for j in range(i+1): 
                    val_name = self.task_names[j]
                    metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j, text_encoder=None, trans_dict=self.trans_dict)
            
        

        if not self.semantic_flag:
            # summarize metrics
            avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])  
            try:
                forgetting = 0
                for i in range(self.max_task-1):  
                    forgetting += metric_table['acc'][self.task_names[i]][self.task_names[i]] - metric_table['acc'][self.task_names[i]][self.task_names[self.max_task-1]]
                forgetting = forgetting / (self.max_task-1)
                print("Forgetting: ",forgetting)
            except:
                print("forgetting error")
                pass 

        return avg_metrics