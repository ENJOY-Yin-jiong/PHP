import torch
import time
import logging
import copy
import sys
import os 
import numpy as np
import wandb


from data_manager import DataManager
from utils import factory




# def load_partial_state_dict(model, state_dict_path):
#     state_dict = torch.load(state_dict_path)
#     model_state_dict = model.state_dict()

#     # 创建一个新的state_dict，只包含匹配的键
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k in model_state_dict and v.size() == model_state_dict[k].size():
#             new_state_dict[k] = v
#         else:
#             print(f"Skipping {k} due to size mismatch or missing key")
        
#     model.load_state_dict(new_state_dict, strict=False)


torch.autograd.set_detect_anomaly(True)




def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)


# def class_eval(args, model, cur_step, task_best_acc_list, step_result_list):
#     if args["dataset"] != "AVQA":
#         print("Not use AVQA")
#         F_event, step_forgetting, task_best_acc_list, step_result_list = model.eval_class(cur_step, task_best_acc_list, step_result_list, args)
#     else:
#         print("use AVQA")
#         F_event, step_forgetting, task_best_acc_list, step_result_list = model.AVQA_eval_task_class(cur_step, task_best_acc_list, step_result_list, args)

#     return F_event, step_forgetting, task_best_acc_list, step_result_list


# def task_eval(args, model, task_list, task_best_acc_list, step_result_list, result_old, logfilename):
    
#     print("======================================== task eval ========================================")

#     result_cur = []
#     forgetting_step = []
    
#     data_manager = DataManager(task_list[-1], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
#     # test on current task
#     if args["dataset"] != "AVQA":
#         print("Not use AVQA")
#         F_event_cur, task_best_acc_list, step_result_list = model.eval_task(len(task_list)-1, task_list[-1], task_best_acc_list, step_result_list, data_manager)
#     else:
#         print("use AVQA")
#         F_event_cur, task_best_acc_list, step_result_list = model.AVQA_eval_task(len(task_list)-1, task_list[-1], task_best_acc_list, step_result_list, data_manager)
    
#     if len(task_list) > 1:
#         for task_idx, task_name in enumerate(task_list[0:-1]):
#             args["dataset"] = task_name
#             data_manager = DataManager(task_name, args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)

#             if task_name == "LLP":
#                 is_weak = 1
#             else:
#                 is_weak = 0
#             specific_model = factory.get_model(args['model_name'], args, task_name, is_weak)
#             specific_model.get_dataset_name(task_name, args)
#             specific_model._network.get_numtask(task_idx)

#             logging.info('test on {}'.format(task_name))
#             logfilename = os.path.abspath(logfilename)
#             saved_model_path = os.path.join(logfilename, "task_{}.pth".format(task_idx))
#             if os.path.exists(saved_model_path):
#                 # load_partial_state_dict(model._network, saved_model_path)
#                 load_partial_state_dict_with_fallback(specific_model._network, model._network.state_dict(), saved_model_path)
#                 logging.info(f"Loaded model parameters from {saved_model_path}")
#             else:
#                 logging.info(f"No saved model parameters found for {saved_model_path}")

#             if task_name != "AVQA":
#                 print("Not use AVQA")
#                 F_event, task_best_acc_list, step_result_list = specific_model.eval_task(task_idx, task_name, task_best_acc_list, step_result_list, data_manager)
#             else:
#                 print("use AVQA")
#                 F_event, task_best_acc_list, step_result_list = specific_model.AVQA_eval_task(task_idx, task_name, task_best_acc_list, step_result_list, data_manager)

#             result_cur.append(F_event)
#     result_cur.append(F_event_cur)

    # if len(task_list) == 1:
    #     forgetting_step = np.array([0])
    # else:
    #     forgetting_step = np.array(result_old) - np.array(result_cur[0:-1])
    

    # log_dir = {}
    
    # for i in range(len(args["task_order"])):
    #     if i < len(result_cur):
    #         log_dir[args["task_order"][i]] = result_cur[i]
    #     else:
    #         log_dir[args["task_order"][i]] = 0
    #     # if i < len(forgetting_step):
    #     #     log_dir["forgetting_"+str(i)] = forgetting_step[i]
    #     # else:
    #     #     log_dir["forgetting_"+str(i)] = 0

    # # log_dir["task_step"] = len(result_cur)

    # wandb.log(log_dir)

    # forgetting_step = []
    
    # print("======================================== task eval over ========================================")

    # return result_cur, forgetting_step


def _train(args):
    if not args["is_task_incremental"]:
        logfilename = 'logs/{}_{}_{}_{}_{}_{}_'.format(args['prefix'], args['model_name'], args['net_type'],
                                                   args['dataset'], args['init_cls'], args['increment'])+ time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    else:
        logfilename = 'logs/{}_{}_{}_{}_{}_{}_'.format(args['prefix'], args['model_name'], args['net_type'],
                                                   "multi_task", args['init_cls'], args['increment'])+ time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    
    wandb.init(project="AV_incremental", name=args["prefix"], config=args)


# -------------------------------------------------- NOT COMPLETED! ---------------------------------------------------------------------------
    task_list = []
    if args["is_task_incremental"] == 1:
        task_list = args["task_order"]
    else:
        task_list.append(args["dataset"])

    # model init
    model = factory.get_model(args['model_name'], args, logfilename)

    for task_idx, cur_task in enumerate(task_list):
        args["dataset"] = cur_task
        # print(args["dataset"])
        data_manager = DataManager(cur_task, args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
        # args['class_order'] = data_manager._class_order
        if cur_task == "LLP":
            is_weak = 1
        else:
            is_weak = 0
        model.update_net(args, cur_task, is_weak)
        model.get_dataset_name(cur_task, args)
        model._network.get_numtask(task_idx)

        # # 尝试加载之前保存的模型参数
        # if task_idx > 0:
        #     logfilename = os.path.abspath(logfilename)
        #     saved_model_path = os.path.join(logfilename, "task_{}.pth".format(task_idx-1))
        #     if os.path.exists(saved_model_path):
        #         load_partial_state_dict(model._network, saved_model_path)
        #         logging.info(f"Loaded model parameters from {saved_model_path}")
        #     else:
        #         logging.info(f"No saved model parameters found for {saved_model_path}")

        
        # task_best_acc_list = []
        # step_forgetting_list = []
        # step_result_list = []
        # result_old = []
        for task in range(data_manager.nb_tasks):
            model.incremental_train(data_manager)
            model.after_task()
            model.task_eval_step(args, data_manager)
            
            # if args["is_task_incremental"]:
            #     cur_step = task_idx
            #     result_old = task_eval(args, model, task_list[0:task_idx+1], task_best_acc_list, step_result_list, result_old, logfilename)
            # else:
            #     cur_step = task
            #     F_event, step_forgetting, task_best_acc_list, step_result_list = class_eval(args, model, cur_step, task_best_acc_list, step_result_list)

            #     if step_forgetting is not None:
            #         step_forgetting_list.append(step_forgetting)

            
           
            # logfilename = os.path.abspath(logfilename)
            # if not os.path.exists(logfilename):
            #     os.makedirs(logfilename)
            # torch.save(model._network.state_dict(), os.path.join(logfilename, "task_{}.pth".format(int(cur_step))))

# -------------------------------------------------- NOT COMPLETED! ---------------------------------------------------------------------------
    
    # if args["is_task_incremental"]:
    #     Mean_forgetting = np.mean(step_forgetting_list)
    #     logging.info("Mean_forgetting:{}".format(Mean_forgetting))
    #     wandb.log({"Mean_forgetting:": Mean_forgetting})

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))