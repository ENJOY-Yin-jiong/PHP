import numpy as np
import torch


class BaseLearner(object):
    def __init__(self, args):
        self._cur_domain = 0
        self._known_classes = []
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        # self.topk = 5

        self._memory_size = args['memory_size']
        self._memory_per_class = args['memory_per_class']
        self._fixed_memory = args['fixed_memory']
        self._device = int(args['device'][0])
        self._multiple_gpus = args['device']

        self.dataset_name = None



    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), 'Exemplar size error.'
        return len(self._targets_memory)

    
    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def after_task(self):
        pass

    def _eval_cnn(self):
        pass

    def _evaluate(self):
        pass

    def eval_task(self, loader):
        # y_pred, y_true = self._eval_cnn(self.test_loader)
        # cnn_accy = self._evaluate(y_pred, y_true)

        # if hasattr(self, '_class_means'):
        #     y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
        #     nme_accy = self._evaluate(y_pred, y_true)
        # else:
        #     nme_accy = None
        pass
    
    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            'tasks': self._cur_domain,
            'model_state_dict': self._network.state_dict(),
        }
        torch.save(save_dict, '{}_{}.pkl'.format(filename, self._cur_domain))