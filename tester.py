
from utils import factory
from data_manager import DataManager
import wandb


logger_name = 'logs/tri_ave_avqa_llp_wo_shallow_mid_deep_AVLearner_slip_multi_task_7_7_2024-10-15-11:32:05'



print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("start test")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



def test(args):
    wandb.init(project="AV_incremental", name=args["prefix"], config=args)

    if args["is_task_incremental"] == 1:
        task_list = args["task_order"]
    else:
        task_list.append(args["dataset"])

    model = factory.get_model(args['model_name'], args, logger_name)

    # for task_idx, cur_task in enumerate(task_list):
    cur_task = task_list[-1]
    task_idx = len(args["task_order"]) - 1
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

    for task in range(data_manager.nb_tasks):
        # model.incremental_train(data_manager)
        # model.after_task()
        model.task_eval_step(args, data_manager)