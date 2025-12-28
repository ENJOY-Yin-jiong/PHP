import torch
import numpy as np


def get_mix_lambda(mixup_alpha, batch_size):
    mixup_lambdas = [np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in range(batch_size)]
    return np.array(mixup_lambdas).astype(np.float32)


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
	# labels = labels[:, :, :-1]  # 28 denote background
	_, targets = labels.max(-1)
	# pos pred
	is_event_scores = is_event_scores.sigmoid()
	scores_pos_ind = is_event_scores > 0.5
	scores_mask = scores_pos_ind == 0
	_, event_class = event_scores.max(-1)  # foreground classification
	pred = scores_pos_ind.long()
	pred *= event_class[:, None]
	# add mask
	pred[scores_mask] = 28  # 141 denotes bg
	correct = pred.eq(targets)
	correct_num = correct.sum().double()
	acc = correct_num * (100. / correct.numel())

	return acc


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union+eps)) / N

    return iou



class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
