import logging
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm
from einops import rearrange
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import wandb
from sklearn.cluster import KMeans
import json
import os
import ast
import time
from torch.cuda.amp import GradScaler, autocast
import sys

from methods.base_learner import BaseLearner
# from methods.net_trans import MMIL_Net
# from methods.AVQA import MMIL_Net
# from methods.PC import MMIL_Net
# from methods.ease import MMIL_Net
# from methods.more_prompt import MMIL_Net
from methods.tri_prompts import MMIL_Net
# from methods.avs_version import MMIL_Net

from .nets.utils import get_mix_lambda, compute_accuracy_supervised, AverageMeter, mask_iou
from utils.toolkit import count_parameters, load_partial_state_dict, load_partial_state_dict_with_fallback
from data_manager import DataManager
from methods.eval import eval_task, AVQA_eval_task, AVS_eval_task
from methods.loss import IouSemanticAwareLoss



class AVLearner(BaseLearner):
    def __init__(self, args, logfilename):
        super().__init__(args)

        self.args = args
        self.is_task_incremental = args['is_task_incremental']


        self.task_order = args['task_order']
        # self.EPSILON = args["EPSILON"]

        # TODO: 先没改，之后根据不同任务有不同的epoch和学习率
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        # self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        # self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]

        self.logfilename = logfilename
        
        self._total_classes = []

        self._cur_task_id = -1

        self.is_use_deep_prompts = args['use_deep_prompt']
        self.is_use_mid_prompts = args['use_mid_prompt']

        if self.is_use_deep_prompts:
            self.deep_prompt_list = []

        if self.is_use_mid_prompts:
            self.mid_prompts_list = []
    
    def update_net(self, args, task, is_weak):
        if not args['train']:
            self._cur_task_id = len(args["task_order"]) - 1
        else: 
            self._cur_task_id += 1
        self._known_classes = []
        self._network = MMIL_Net(args, task, is_weak)
        self.is_weak = is_weak

        # 更新参数
        logging.info(f"<<<<<<< incremental stage {self._cur_task_id} >>>>>>>>")
        logging.info("<<<<<<< update model parameters >>>>>>>>")
        

        if self._cur_task_id > 0:
            saved_model_path = os.path.join(self.logfilename, "task_{}.pth".format(self._cur_task_id-1))

            if os.path.exists(saved_model_path):
                    load_partial_state_dict(self._network, saved_model_path)
                    logging.info(f"Loaded model parameters from {saved_model_path}")
            else:
                logging.info(f"No saved model parameters found for {saved_model_path}")


    def after_task(self):
        # self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

        logfilename = os.path.abspath(self.logfilename)
        if not os.path.exists(logfilename):
            os.makedirs(logfilename)
        torch.save(self._network.state_dict(), os.path.join(logfilename, "task_{}.pth".format(int(self._cur_task_id))))
        
        if self.is_use_deep_prompts:
            self.deep_prompt_list.append(self._network.deep_visual_e_prompt)
        if self.is_use_mid_prompts:
            self.mid_prompts_list.append(self._network.mid_e_prompt) 
        torch.cuda.empty_cache()
        
    def get_dataset_name(self, dataset_name, args):
        self.dataset_name = dataset_name
        self._network.get_dataset_name(dataset_name)


    def incremental_train(self, data_manager):
        
        # self._cur_task_id += 1
        # print(data_manager.get_task_order(self._cur_domain))
        self._cur_class = data_manager.get_task_order(self._cur_domain)

        print(self._known_classes)
        print(self._cur_class)
        self._total_classes = self._known_classes + self._cur_class

        if not self.is_task_incremental:
            self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))
        # wandb.log('Learning on {}-{}'.format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(self._cur_class, source='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                    num_workers=self.num_workers, pin_memory=True)
        test_dataset = data_manager.get_dataset(self._total_classes, source='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                    num_workers=self.num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if not self.is_task_incremental:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module



    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        with open('parameter_names.txt', 'w') as f:
            for name, param in self._network.named_parameters():
                f.write(name + '\n')


        for name, param in self._network.named_parameters():
            param.requires_grad = False
			### ---> compute params
            tmp = 1
            for num in param.shape:
                tmp *= num

            if 'ViT' in name:
                param.requires_grad = False if self.args["fine_tune"] == 0 else True
                # total_params += tmp
            elif 'htsat' in name:
                param.requires_grad = False if self.args["fine_tune"] == 0 else True
                # total_params += tmp
            elif 'text_encoder' in name:
                param.requires_grad = False if self.args["fine_tune"] == 0 else True
                # total_params += tmp
            elif 'token_embedding' in name:
                param.requires_grad = False if self.args["fine_tune"] == 0 else True
                # total_params += tmp
            elif 'adapter_blocks' in name:
                param.requires_grad = True
                # train_params += tmp
                # additional_params += tmp
                # total_params += tmp
                # print('########### train layer:', name, param.shape, tmp)
            elif 'clip_adapter' in name:
                param.requires_grad = True
                # train_params += tmp
                # total_params += tmp
                # additional_params += tmp
            elif 'audio_adapter' in name:
                param.requires_grad = True
                # train_params += tmp
                # total_params += tmp
                # additional_params += tmp
            elif 'prompt_learner' in name:
                param.requires_grad = True
                # train_params += tmp
                # total_params += tmp
                # additional_params += tmp
            elif 'CMBS' in name:
                param.requires_grad = True
                # train_params += tmp
                # additional_params += tmp
                # total_params += tmp
            elif 'audio_visual_contrastive_learner' in name:
                param.requires_grad = True

            elif 'audio_projection' in name:
                param.requires_grad = False if self.args["fine_tune"] == 0 else True
                # train_params += tmp
                # additional_params += tmp
                # total_params += tmp
            elif "visual_e_prompt" in name:
                param.requires_grad = True
            # elif "g_prompt" in name:
            #     param.requires_grad = True
            elif "audio_e_prompt" in name:
                param.requires_grad = True
            elif "cls_token" in name:
                param.requires_grad = True
            # elif "audio_prompt_pool" + "." + str(self._network.numtask) in name:
            #     param.requires_grad = True
            # elif "audio_prompt_pool_lr" + "." + str(self._network.numtask) in name:
            #     param.requires_grad = True
            # elif "audio_prompt_pool_tb" + "." + str(self._network.numtask) in name:
            #     param.requires_grad = True
            # elif "vis_prompt_proj" + "." + str(self._network.numtask) in name:
            # # elif "vis_prompt_proj" in name:
            #     param.requires_grad = True
            # elif "aud_prompt_proj" + "." + str(self._network.numtask) in name:
            # # elif "aud_prompt_proj" in name:
            #     param.requires_grad = True
            elif "handler" in name:
                param.requires_grad = True
            elif "mid_e_prompt" in name:
                param.requires_grad = True
            elif "deep_visual_e_prompt" in name:
                param.requires_grad = True
            
            

        # with open('parameter_names_learnable.txt', 'w') as f:
        #     for name, param in self._network.named_parameters():
        #         if not param.requires_grad: 
        #             f.write(name + '\n')

        # Double check
        enabled = set()
        with open("enabled.txt", "w") as f:	
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
                    f.writelines(name + '\n')
                

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        wandb.log({'All params': count_parameters(self._network)})
        wandb.log({'Trainable params:': count_parameters(self._network, True)})


        if self._cur_domain==0:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), self.init_lr, weight_decay=1e-5)
            # optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.init_lr,weight_decay=self.init_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch, eta_min=1e-6)
            # scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
            self.run_epoch = self.init_epoch
            
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            # optimizer = optim.SGD(self._network.parameters(), momentum=0.9,lr=self.lrate,weight_decay=self.weight_decay)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()), self.lrate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs, eta_min=1e-6)
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)


    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        for _, epoch in enumerate(range(self.run_epoch)):
            # self._network.train()

            # init losses
            mean_loss = torch.zeros(1).to(self._device)
            mean_loss_audio_image = torch.zeros(1).to(self._device)
            mean_loss_image_audio = torch.zeros(1).to(self._device)
            mean_acc = torch.zeros(1).to(self._device)

            mean_iou = AverageMeter('miou')

            prog_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=True)
            for batch_idx, sample in enumerate(prog_bar):

                gt = sample['GT']
                # print("gt", gt.shape)
                image = sample['image']
                wave = sample['wave']
                bs = image.size(0)
                
                # 半精�?                # with autocast(dtype=torch.float16):
                
                if self.dataset_name == 'AVQA':
                    question = sample['question']
                    event_scores, out_match_posi, loss_vis_reduce_sim, loss_aud_reduce_sim = self._network(wave.to(self._device), image.to(self._device), question=question.to(self._device), train=True)
                elif self.dataset_name == 'AVS':
                    mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*5)).to('cuda')
                    output_mask, v_map_list, a_fea_list = self._network(wave.to(self._device), image.to(self._device), train=True, mixup_lambda=mixup_lambda)
                else:
                    start_time = time.time()
                    event_scores, logits_audio_image, logits_image_audio, loss_vis_reduce_sim, loss_aud_reduce_sim = self._network(wave.to(self._device), image.to(self._device), train=True)
                    end_time = time.time()



                labels = gt.to(self._device)

                if self.dataset_name == 'AVE':
                    loss = F.cross_entropy(event_scores, rearrange(labels, 'b t class -> (b t) class'))  # criterion_event
                    audio_visual_labels = torch.eye(bs).to(self._device)
                    loss_audio_image = F.cross_entropy(logits_audio_image, audio_visual_labels)    # criterion   
                    loss_image_audio = F.cross_entropy(logits_image_audio, audio_visual_labels)    # criterion
            
                elif self.dataset_name == 'LLP':

                    event_scores = event_scores.view(bs, 10, -1)
                    event_scores = torch.mean(event_scores, dim=1)
                    loss = F.cross_entropy(event_scores, labels)
                    audio_visual_labels = torch.eye(bs).to(self._device)
                    loss_audio_image = F.cross_entropy(logits_audio_image, audio_visual_labels)    # criterion   
                    loss_image_audio = F.cross_entropy(logits_image_audio, audio_visual_labels)    # criterion
            
                elif self.dataset_name == "AVQA":
                    loss = F.cross_entropy(event_scores, labels)
                    loss_audio_image = torch.tensor(0)
                    loss_image_audio = torch.tensor(0)
                
                elif self.dataset_name == "AVS":
                    B, frame, C, H, W = image.shape
                    mask = labels.view(B, H, W)
                    loss, loss_dict = IouSemanticAwareLoss(output_mask, labels, \
												a_fea_list, v_map_list, \
												lambda_1=50, \
												count_stages=[], \
												sa_loss_flag=False, \
												mask_pooling_type='avg')
                    loss_audio_image = torch.tensor(0).to(self._device)
                    loss_image_audio = torch.tensor(0).to(self._device)
                    loss_vis_reduce_sim = torch.tensor(0).to(self._device)
                    loss_aud_reduce_sim = torch.tensor(0).to(self._device)

        
                    indices = torch.tensor(list(range(0, len(output_mask), 5)))
                    indices = indices.cuda()
                    first_pred = torch.index_select(output_mask, dim=0, index=indices) # [bs, 1, 224, 224]
                    miou = mask_iou(first_pred.squeeze(1), mask)
                    mean_iou.add({'miou': miou})
                    # print(mean_iou.pop('miou'))


                # audio_visual_labels = torch.eye(bs).to(self._device)
                # loss_audio_image = F.cross_entropy(logits_audio_image, audio_visual_labels)    # criterion   
                # loss_image_audio = F.cross_entropy(logits_image_audio, audio_visual_labels)    # criterion
            
                if self.dataset_name == 'AVE' or self.dataset_name == 'LLP':
                    loss = 5 * loss + 2 * loss_audio_image + 1.5 * loss_image_audio + 2 * loss_vis_reduce_sim + 2 * loss_aud_reduce_sim
                elif self.dataset_name == 'AVQA':
                    loss = loss  # AVQA只需要分类loss
                elif self.dataset_name == 'AVS':
                    loss = loss  # AVS已经在IouSemanticAwareLoss中处理了loss组合
                    


                # 半精�?                # scaler.scale(loss).backward(retain_graph=True)
                loss.backward(retain_graph=True)


                # FOR DISTRIBUTION
                # loss_audio_image = reduce_value(loss_audio_image, average=True)
                # loss_image_audio = reduce_value(loss_image_audio, average=True)
                mean_loss = (mean_loss * batch_idx + loss.detach()) / (batch_idx + 1)
                mean_loss_image_audio = (mean_loss_image_audio * batch_idx + loss_image_audio.detach()) / (batch_idx + 1)
                mean_loss_audio_image = (mean_loss_audio_image * batch_idx + loss_audio_image.detach()) / (batch_idx + 1)
                # mean_loss_av_prompt_sim = (loss_av_prompt_sim * batch_idx + loss_av_prompt_sim.detach()) / (batch_idx + 1)


                '''Compute Accuracy'''
                if self.dataset_name == "AVS":
                    mean_acc = torch.tensor(-1)
                else:
                    if self.is_weak or self.dataset_name == "AVQA":
                        val = (event_scores.argmax(dim=-1) == labels.argmax(dim=-1)).sum()
                    else:
                        val = (event_scores.argmax(dim=-1) == rearrange(labels, 'b t class -> (b t) class').argmax(dim=-1)).sum()
                    num = event_scores.size(0)
                    acc = val / num * 100
                    mean_acc = (mean_acc * batch_idx + acc) / (batch_idx + 1)
                    
                    
                '''Clip Gradient'''
                # if args.clip_gradient is not None:
                total_norm = clip_grad_norm_(self._network.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            info = 'Task {}, Epoch {}/{} => Loss_total {:.6f}, train acc: {:.3f}, loss_is_event: {:.6f} loss_audio_image: {:.6f} loss_image_audio: {:.6f}'.format(
                self._cur_task_id, epoch + 1, self.run_epoch, mean_loss.item(), mean_acc.item(),
                (mean_loss-mean_loss_image_audio-mean_loss_audio_image).item(), 
                mean_loss_audio_image.item(), mean_loss_image_audio.item())
            prog_bar.set_description(info)

        wandb.log({
        "Task": self._cur_task_id,
        "Epoch": epoch + 1,
        "Loss_total": mean_loss.item(),
        "train_acc" : mean_acc.item(),
        "loss_is_event": (mean_loss-mean_loss_image_audio-mean_loss_audio_image).item(),
        "loss_audio_image": mean_loss_audio_image.item(),
        "loss_image_audio": mean_loss_image_audio.item(),
        # "loss_av_sim": mean_loss_av_prompt_sim.item()
        })

            
        logging.info(info)



    def task_eval_step(self, args, cur_data_manager):
        print("======================================== task eval ========================================")

        result_cur = []
        forgetting_step = []
        

        data_manager = cur_data_manager
        # test on current task
        # print(self._cur_task_id)
        if self.task_order[self._cur_task_id] == "AVQA":
            print("use AVQA")
            # F_event_cur = self.AVQA_eval_task(self._cur_task_id, data_manager)
            F_event_cur = AVQA_eval_task(self._cur_task_id, self._network, data_manager, self._device, args)
        elif self.task_order[self._cur_task_id] == "AVS":
            F_event_cur = AVS_eval_task(self._cur_task_id, self.task_order[self._cur_task_id], self._network, data_manager, self._device, args=args)
        else:
            print("Not use AVQA")
            # F_event_cur = self.eval_task(self._cur_task_id, self.task_order[self._cur_task_id], data_manager)
            F_event_cur = eval_task(self._cur_task_id, self.task_order[self._cur_task_id], self._network, data_manager, self._device, is_weak=self.is_weak, args=args)
        
        # print(self._cur_task_id)
        if len(self.task_order) > 1 and self._cur_task_id > 0:
            for task_idx, task_name in enumerate(self.task_order[0:self._cur_task_id]):
                # if task_idx < self._cur_task_id:
                data_manager = DataManager(task_name, args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)

                if task_name == "LLP":
                    is_weak = 1
                else:
                    is_weak = 0
                
                specific_model = MMIL_Net(args, task_name, is_weak)
                specific_model.get_dataset_name(task_name)
                specific_model.get_numtask(task_idx)

                logging.info('test on {}'.format(task_name))
                logfilename = os.path.abspath(self.logfilename)
                saved_model_path = os.path.join(logfilename, "task_{}.pth".format(task_idx))
                if os.path.exists(saved_model_path):
                    # load_partial_state_dict(model._network, saved_model_path)
                    load_partial_state_dict_with_fallback(specific_model, self._network.state_dict(), saved_model_path)
                    logging.info(f"Loaded model parameters from {saved_model_path}")
                else:
                    logging.info(f"No saved model parameters found for {saved_model_path}")
                
                if self.is_use_deep_prompts:
                    specific_model.deep_visual_e_prompt = self.deep_prompt_list[task_idx]
                if self.is_use_mid_prompts:
                    specific_model.mid_e_prompt = self.mid_prompts_list[task_idx]
                if task_name == "AVQA":
                    print("use AVQA")
                    # F_event = specific_model.AVQA_eval_task(task_idx, data_manager)
                    F_event = AVQA_eval_task(task_idx, specific_model, data_manager, self._device, args)
                elif task_name == "AVS":
                    print("use AVS")
                    F_event = AVS_eval_task(task_idx, task_name, specific_model, data_manager, self._device, args)
                else:
                    print("Not use AVQA")
                    # F_event = specific_model.eval_task(task_idx, task_name, data_manager)
                    F_event = eval_task(task_idx, task_name, specific_model, data_manager, self._device, is_weak, args)

                result_cur.append(F_event)
        result_cur.append(F_event_cur)

        log_dir = {}
    
        for i in range(self._cur_task_id+1):
            if i < len(result_cur):
                log_dir[args["task_order"][i]] = result_cur[i]
            else:
                log_dir[args["task_order"][i]] = 0
                # if i < len(forgetting_step):
                #     log_dir["forgetting_"+str(i)] = forgetting_step[i]
                # else:
                #     log_dir["forgetting_"+str(i)] = 0

            # log_dir["task_step"] = len(result_cur)

        wandb.log(log_dir)

            # forgetting_step = []
            
        print("======================================== task eval over ========================================")

