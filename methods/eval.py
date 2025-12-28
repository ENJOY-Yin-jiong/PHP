import torch
from einops import rearrange
import torch.nn as nn
import logging
import wandb
import json
from torch.utils.data import DataLoader
import numpy as np
import ast
from tqdm import tqdm
import os


from .nets.utils import get_mix_lambda, mask_iou, AverageMeter

def top_1_acc(logits, target):
    # if len()logits.shape
    # print(logits)
    top1_res = logits.argmax(dim=1)
    # print(top1_res.shape, '-------------', target.shape)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()


# def eval_class(step, task_best_acc_list, step_result_list, args):
#     # model.eval()
    
#     # mean_acc = torch.zeros(1).to(device)
    
#     all_test_out_logits  = torch.Tensor([])# .to(device)
#     all_test_labels = torch.Tensor([])# .to(device)
#     with torch.no_grad():
#         for batch_idx, sample in enumerate(self.test_loader):
#             gt= sample['GT']# .to(device)
#             image = sample['image'].to(device)
#             wave = sample['wave'].to(device)
#             bs = image.size(0)
#             labels = gt.to(device)

#             if isinstance(model, nn.DataParallel):
#                 event_scores, _, _  = model.module.interface(image, wave)
#             else:
#                 event_scores, _, _  = model.interface(image, wave)


#             # event_scores, _, _ = model(wave.to(device), image.to(device))
#             if not self.is_weak:
#                 event_scores_logits = event_scores.detach().cpu()
#                 all_test_out_logits = torch.cat((all_test_out_logits, event_scores_logits),dim=0)   
#                 gt_label = rearrange(gt, 'b t class -> (b t) class')
#             else:
#                 event_scores = event_scores.view(bs, 10, -1)
#                 event_scores = event_scores.mean(dim=1)
#                 event_scores_logits = event_scores.detach().cpu()
#                 all_test_out_logits = torch.cat((all_test_out_logits, event_scores_logits),dim=0)   
#                 gt_label = gt
            
#             all_test_labels = torch.cat((all_test_labels, gt_label.argmax(dim=-1)), dim=0)


#     test_top1 = top_1_acc(all_test_out_logits, all_test_labels)
#     logging.info("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
#     # wandb.log("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
#     wandb.log({"incremental step": step,
#             "F1 score": test_top1})

    
#     old_task_acc_list = []
#     for i in range(step+1):
#         # step_class_list = range(i*args['increment'], (i+1)*args['increment'])
#         step_class_list = self._cur_class
#         step_class_idxs = []
#         for c in step_class_list:
#             idxs = np.where(all_test_labels.numpy() == c)[0].tolist()
#             step_class_idxs += idxs
#         step_class_idxs = np.array(step_class_idxs)
#         i_labels = torch.Tensor(all_test_labels.numpy()[step_class_idxs])
#         i_logits = torch.Tensor(all_test_out_logits.numpy()[step_class_idxs])
#         i_acc = top_1_acc(i_logits, i_labels)
#         if i == step:
#             curren_step_acc = i_acc
#         else:
#             old_task_acc_list.append(i_acc)
#     if step > 0:
#         forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
#         print('forgetting: {:.6f}'.format(forgetting))
#         for i in range(len(task_best_acc_list)):
#             task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
#     else:
#         forgetting = None
#     task_best_acc_list.append(curren_step_acc)
    
#     if step > 0: 
#         wandb.log({
#             "step_forgetting": forgetting
#         })
#     # forgetting

#     # print('val acc: %.3f'%(mean_acc.item()))
#     # return mean_acc.item()

#     return test_top1, forgetting, task_best_acc_list, step_result_list


def eval_task(step, task_name, model, data_manager, device, is_weak, args):
    # model.eval()
    model.to(device)
    # mean_acc = torch.zeros(1).to(device)_total_classes
    test_dataset = data_manager.get_dataset(data_manager.get_task_order(0), source='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                num_workers=args['num_workers'])
    
    all_test_out_logits  = torch.Tensor([])# .to(device)
    all_test_labels = torch.Tensor([])# .to(device)
    # all_for_tsne_labels = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_loader)):
            gt = sample['GT']# .to(device)
            image = sample['image'].to(device)
            wave = sample['wave'].to(device)
            bs = image.size(0)

            

            if isinstance(model, nn.DataParallel):
                event_scores, _, _, _, _ = model.module.interface(image, wave)
            else:
                event_scores, _, _, _, _ = model.interface(image, wave)


            if not is_weak:
                event_scores_logits = event_scores.detach().cpu()
                all_test_out_logits = torch.cat((all_test_out_logits, event_scores_logits),dim=0)   
                gt_label = rearrange(gt, 'b t class -> (b t) class')
            else:
                event_scores = event_scores.view(bs, 10, -1)
                event_scores = event_scores.mean(dim=1)
                event_scores_logits = event_scores.detach().cpu()
                all_test_out_logits = torch.cat((all_test_out_logits, event_scores_logits),dim=0)   
                gt_label = gt
                # print(gt.shape)
            all_test_labels = torch.cat((all_test_labels, gt_label.argmax(dim=-1)), dim=0)
            # all_for_tsne_labels.append(gt_label)
            # print('111111111111111111111111111111111111111')
            # print(task_name)
            # print('111111111111111111111111111111111111111')
            # with open('output_'+ task_name +'.txt', 'a', encoding='utf-8') as file:
            #     file.write(str(event_scores_logits.argmax(dim=-1)) + '\n')
# xxx
    test_top1 = top_1_acc(all_test_out_logits, all_test_labels)
    logging.info("Incremental step {}-{} Testing res: {:.6f}".format(step, task_name, test_top1))
    # wandb.log("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
    wandb.log({"incremental step": step,
                "F1 score": test_top1})


    return test_top1


# def AVQA_eval_class(step, task_best_acc_list, step_result_list, args):
    
#     all_result = []
    
#     total_qa = 0
#     total_match=0
#     correct_qa = 0
#     correct_match=0
#     A_count = []
#     A_cmp = []
#     V_count = []
#     V_loc = []
#     AV_ext = []
#     AV_count = []
#     AV_loc = []
#     AV_cmp = []
#     AV_temp = []
#     samples = json.load(open('/opt/data/private/dataset/AVQA/labels/json/avqa-test.json', 'r'))

    
#     # samples = json.load(open('/opt/data/private/dataset/AVQA/labels/json/avqa-test.json', 'r'))
#     with torch.no_grad():
#         for batch_idx, sample in enumerate(self.test_loader):
#             target = sample['GT'].to(device)
#             image = sample['image'].to(device)
#             wave = sample['wave'].to(device)
#             question = sample['question'].to(device)

        
#             if isinstance(model, nn.DataParallel):
#                 preds_qa, out_match_posi = model.module.interface(image, wave, question=question)
#             else:
#                 preds_qa, out_match_posi = model.interface(image, wave, question=question)

#             _, predicted = torch.max(preds_qa.data, 1)
#             total_qa += preds_qa.size(0)
#             correct_qa += (predicted == target).sum().item()
            
            

#             x = samples[batch_idx]
#             type =ast.literal_eval(x['type'])
#             if type[0] == 'Audio':
#                 if type[1] == 'Counting':
#                     A_count.append((predicted == target).sum().item())
#                 elif type[1] == 'Comparative':
#                     A_cmp.append((predicted == target).sum().item())
#             elif type[0] == 'Visual':
#                 if type[1] == 'Counting':
#                     V_count.append((predicted == target).sum().item())
#                 elif type[1] == 'Location':
#                     V_loc.append((predicted == target).sum().item())
#             elif type[0] == 'Audio-Visual':
#                 if type[1] == 'Existential':
#                     AV_ext.append((predicted == target).sum().item())
#                 elif type[1] == 'Counting':
#                     AV_count.append((predicted == target).sum().item())
#                 elif type[1] == 'Location':
#                     AV_loc.append((predicted == target).sum().item())
#                 elif type[1] == 'Comparative':
#                     AV_cmp.append((predicted == target).sum().item())
#                 elif type[1] == 'Temporal':
#                     AV_temp.append((predicted == target).sum().item())
    
#     all_result.append(correct_qa/total_qa)

        

#     logging.info("Incremental step {} Testing res: {:.6f}".format(step, correct_qa/total_qa))
#     # wandb.log("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
#     wandb.log({"incremental step": step,
#                 "F1 score": correct_qa/total_qa})
#     logging.info('Audio Counting Accuracy: %.2f %%' % (100 * sum(A_count)/len(A_count)))
#     logging.info('Audio Cmp Accuracy: %.2f %%' % (
#             100 * sum(A_cmp) / len(A_cmp)))
#     logging.info('Audio Accuracy: %.2f %%' % (
#             100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
#     logging.info('Visual Counting Accuracy: %.2f %%' % (
#             100 * sum(V_count) / len(V_count)))
#     logging.info('Visual Loc Accuracy: %.2f %%' % (
#             100 * sum(V_loc) / len(V_loc)))
#     logging.info('Visual Accuracy: %.2f %%' % (
#             100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
#     logging.info('AV Ext Accuracy: %.2f %%' % (
#             100 * sum(AV_ext) / len(AV_ext)))
#     logging.info('AV counting Accuracy: %.2f %%' % (
#             100 * sum(AV_count) / len(AV_count)))
#     logging.info('AV Loc Accuracy: %.2f %%' % (
#             100 * sum(AV_loc) / len(AV_loc)))
#     logging.info('AV Cmp Accuracy: %.2f %%' % (
#             100 * sum(AV_cmp) / len(AV_cmp)))
#     logging.info('AV Temporal Accuracy: %.2f %%' % (
#             100 * sum(AV_temp) / len(AV_temp)))
#     logging.info('AV Accuracy: %.2f %%' % (
#             100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
#                 +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))
    
#     wandb.log({
#         "Audio Counting Accuracy": 100 * sum(A_count)/len(A_count),
#         "Audio Cmp Accuracy": 100 * sum(A_cmp) / len(A_cmp),
#         "Audio Accuracy": 100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp)),
#         "Visual Counting Accuracy": 100 * sum(V_count) / len(V_count),
#         "Visual Loc Accuracy": 100 * sum(V_loc) / len(V_loc), 
#         "Visual Accuracy": 100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc)),
#         "AV Ext Accuracy": 100 * sum(AV_ext) / len(AV_ext),
#         "AV counting Accuracy": 100 * sum(AV_count) / len(AV_count),
#         "AV Loc Accuracy": 100 * sum(AV_loc) / len(AV_loc),
#         "AV Cmp Accuracy": 100 * sum(AV_cmp) / len(AV_cmp),
#         "AV Temporal Accuracy": 100 * sum(AV_temp) / len(AV_temp),
#         "AV Accuracy": 100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
#         +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))
#     })

    

#     step_result_list.append(correct_qa/total_qa)

#     if step > 0:
#         forgetting = step_result_list[-1] - step_result_list[-2]
#         logging.info('forgetting: {:.6f}'.format(forgetting))
#         wandb.log({
#             "step_forgetting": forgetting
#         })
#     else:
#         forgetting = None 
        
#     return correct_qa/total_qa, forgetting, task_best_acc_list, step_result_list


    
def AVQA_eval_task(step, model, data_manager, device, args):
    
    all_result = []
    
    total_qa = 0
    total_match=0
    correct_qa = 0
    correct_match=0
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    samples = json.load(open('/opt/data/private/dataset/AVQA/labels/json/avqa-test.json', 'r'))
    # samples = json.load(open('/opt/data/private/dataset/AVQA/labels/json/avqa-test.json', 'r'))
    
    # model   self.num_workers    device    


    model.to(device)
    test_dataset = data_manager.get_dataset(data_manager.get_task_order(0), source='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                num_workers=args['num_workers'])

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_loader)):
            target = sample['GT'].to(device)
            image = sample['image'].to(device)
            wave = sample['wave'].to(device)
            question = sample['question'].to('cuda')
            bs = image.shape[0]

            if isinstance(model, nn.DataParallel):
                preds_qa, out_match_posi, loss_vis_reduce_sim, loss_aud_reduce_sim  = model.module.interface(image, wave, question=question)
            else:
                preds_qa, out_match_posi, loss_vis_reduce_sim, loss_aud_reduce_sim = model.interface(image, wave, question=question)



            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

            with open('output_AVQA.txt', 'a', encoding='utf-8') as file:
                file.write(str(preds_qa.argmax(dim=-1)) + '\n')

            x = samples[batch_idx]
            type =ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())
    
    all_result.append(correct_qa/total_qa)
    

    logging.info("Incremental step {} Testing res: {:.6f}".format(step, correct_qa/total_qa))
    # wandb.log("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
    wandb.log({"incremental step": step,
               "F1 score": correct_qa/total_qa})
    logging.info('Audio Counting Accuracy: %.2f %%' % (100 * sum(A_count)/len(A_count)))
    logging.info('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    logging.info('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    logging.info('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    logging.info('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    logging.info('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    logging.info('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    logging.info('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    logging.info('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    logging.info('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    logging.info('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))
    logging.info('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))
    
    wandb.log({
        "Audio Counting Accuracy": 100 * sum(A_count)/len(A_count),
        "Audio Cmp Accuracy": 100 * sum(A_cmp) / len(A_cmp),
        "Audio Accuracy": 100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp)),
        "Visual Counting Accuracy": 100 * sum(V_count) / len(V_count),
        "Visual Loc Accuracy": 100 * sum(V_loc) / len(V_loc), 
        "Visual Accuracy": 100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc)),
        "AV Ext Accuracy": 100 * sum(AV_ext) / len(AV_ext),
        "AV counting Accuracy": 100 * sum(AV_count) / len(AV_count),
        "AV Loc Accuracy": 100 * sum(AV_loc) / len(AV_loc),
        "AV Cmp Accuracy": 100 * sum(AV_cmp) / len(AV_cmp),
        "AV Temporal Accuracy": 100 * sum(AV_temp) / len(AV_temp),
        "AV Accuracy": 100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
        +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))
    })

    

    # step_result_list.append(correct_qa/total_qa)

    # if step > 0:
    #     forgetting = step_result_list[-1] - step_result_list[-2]
    #     logging.info('forgetting: {:.6f}'.format(forgetting))
    #     wandb.log({
    #         "step_forgetting": forgetting
    #     })
    # else:
    #     forgetting = None 
        
    # forgetting

    # print('val acc: %.3f'%(mean_acc.item()))
    # return mean_acc.item()

    return correct_qa/total_qa# , task_best_acc_list, step_result_list



def AVS_eval_task(step, task_name, model, data_manager, device, args):
    # model.eval()
    model.to(device)
    # mean_acc = torch.zeros(1).to(device)_total_classes
    test_dataset = data_manager.get_dataset(data_manager.get_task_order(0), source='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                num_workers=args['num_workers'])
    
    all_test_out_logits  = torch.Tensor([])# .to(device)
    all_test_labels = torch.Tensor([])# .to(device)
    # all_for_tsne_labels = []

    avg_meter_miou = AverageMeter('miou')

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_loader)):
            gt = sample['GT']# .to(device)
            image = sample['image'].to(device)
            wave = sample['wave'].to(device)

            B, frame, C, H, W = image.shape
            mask = gt.to(device)
            mask = mask.view(B*frame, H, W)             
            

            mixup_lambda = torch.from_numpy(get_mix_lambda(0.5, len(wave)*5)).to('cuda')
            if isinstance(model, nn.DataParallel):
                output, _, _ = model.module.interface(image, wave, mixup_lambda=mixup_lambda)
            else:
                output, _, _ = model.interface(image, wave, mixup_lambda=mixup_lambda)

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})
        
        miou = (avg_meter_miou.pop('miou'))
        

    logging.info("Incremental step {}-{} Testing res: {:.6f}".format(step, task_name, miou))
    # wandb.log("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
    wandb.log({"incremental step": step,
                "mIOU": miou})

    return miou