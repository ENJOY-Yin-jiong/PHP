# The codes are from https://github.com/iamwangyabin/S-Prompts

import logging
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, Normalize
from PIL import Image
import glob
import os
import torch
import torchvision
import random
import ast
import json
import pandas as pd
import torchaudio
import pickle


from data import AVE, LLP, AVQA, AVS, generate_category_list


def generate_incremental_domain_order(dataset, domain_order):
    
    if dataset == 'AVE':
        path = '/opt/data/private/dataset/AVE/labels/domain_class.json'
        tasks_name, id_to_idx = generate_category_list('/opt/data/private/dataset/AVE/', 'AVE')
    elif dataset == 'LLP':
        path = '/opt/data/private/dataset/AVVP/LLP_dataset/labels/domain_class.json'
        tasks_name, id_to_idx = generate_category_list('/opt/data/private/dataset/AVVP/LLP_dataset/', 'LLP')
    with open(path, 'r') as f:
        domain_class = json.load(f) 

    incremental_order = []
    for domain in domain_order:
        domain_list = []
        for cat in domain_class[domain]:
            domain_list.append(id_to_idx[cat])
        incremental_order.append(domain_list)
    
    return incremental_order


def generate_incremental_task_order(dataset):
    if dataset == 'AVE':
        tasks_name, id_to_idx = generate_category_list('/opt/data/private/dataset/AVE/', 'AVE')
    elif dataset == 'LLP':
        tasks_name, id_to_idx = generate_category_list('/opt/data/private/dataset/AVVP/LLP_dataset/', 'LLP')
    # 缺AVQA    
    elif dataset == 'AVQA':
        type_categories = ['Counting', 'Existential', 'Location', 'Comparative', 'Temporal']
        id_to_idx = {id: index for index, id in enumerate(type_categories)}
    elif dataset == 'AVS':
        # Read categories from CSV for AVS
        df_all = pd.read_csv('/opt/data/private/dataset/AVS/Single-source/s4_data/s4_meta_data.csv', sep=',')
        categories = df_all['category'].unique().tolist()
        id_to_idx = {id: index for index, id in enumerate(categories)}

    incremental_order = []
    # incremental_order.append(np.arange(0, len(id_to_idx)))
    incremental_order.append(list(range(0, len(id_to_idx))))
    return incremental_order



class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args
        self.dataset_name = dataset_name

        self._setup_data(dataset_name, shuffle, seed)

        if args["is_task_incremental"] == 0 and args["domain_order"]:
            self._increments = generate_incremental_domain_order(self.dataset_name, args["domain_order"])
            # print(self._increments)
        elif args["is_task_incremental"]:
            # print(self.dataset_name)
            self._increments = generate_incremental_task_order(self.dataset_name)
        else:
            assert init_cls <= len(self._class_order), 'No enough classes.'
            self._increments = []
            self._increments.append(list(range(0, init_cls)))
            class_order_num = sum(len(sublist) for sublist in self._increments)
            while class_order_num + increment < len(self._class_order):
                self._increments.append(list(range(class_order_num, class_order_num + increment)))
                class_order_num = sum(len(sublist) for sublist in self._increments)


    @property
    def nb_tasks(self):
        return len(self._increments)


    def get_task_order(self, task):
        return self._increments[task]
    

    def get_dataset(self, indices, source, appendent=None, ret_data=False):
        
        if source == 'train':
            x = self._train_data
            self.raw_gt = self.train_gt
        elif source == 'test':
            x = self._test_data
            self.raw_gt = self.test_gt
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        data = []
        for idx in indices:
            class_data = x[idx]
            for smp_data in class_data:
                data.append(smp_data)
        if self.dataset_name != 'AVQA':
            return DummyDataset(self.dataset_name, data, self.label, self.trsf, self.data_path, self.raw_gt, split=source)
        else:
            return DummyDataset(self.dataset_name, data, self.label, self.trsf, self.data_path, self.raw_gt, self.ans_vocab, self.word_to_ix, split=source)

        # TODO:附加的
        # if appendent is not None and len(appendent) != 0:
        #     appendent_data, appendent_targets = appendent
        #     data.append(appendent_data)
        #     targets.append(appendent_targets)

        # data, targets = np.concatenate(data), np.concatenate(targets)

        # if ret_data:
        #     return data, targets, DummyDataset(data, targets, self.use_path)
        # else:
        #     return DummyDataset(data, targets, self.use_path)
        

    # def get_anchor_dataset(self, mode, appendent=None, ret_data=False):
    #     if mode == 'train':
    #         trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
    #     # elif mode == 'flip':
    #     #     trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
    #     elif mode == 'test':
    #         trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
    #     else:
    #         raise ValueError('Unknown mode {}.'.format(mode))

    #     data, targets = [], []
    #     if appendent is not None and len(appendent) != 0:
    #         appendent_data, appendent_targets = appendent
    #         data.append(appendent_data)
    #         targets.append(appendent_targets)

    #     data, targets = np.concatenate(data), np.concatenate(targets)

    #     if ret_data:
    #         return data, targets, DummyDataset(data, targets, trsf, self.use_path)
    #     else:
    #         return DummyDataset(data, targets, trsf, self.use_path)
        

    # def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
    #     if source == 'train':
    #         x, y = self._train_data, self._train_targets
    #     elif source == 'test':
    #         x, y = self._test_data, self._test_targets
    #     else:
    #         raise ValueError('Unknown data source {}.'.format(source))

    #     if mode == 'train':
    #         trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
    #     elif mode == 'test':
    #         trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
    #     else:
    #         raise ValueError('Unknown mode {}.'.format(mode))

    #     train_data, train_targets = [], []
    #     val_data, val_targets = [], []
    #     for idx in indices:
    #         class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
    #         val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
    #         train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
    #         val_data.append(class_data[val_indx])
    #         val_targets.append(class_targets[val_indx])
    #         train_data.append(class_data[train_indx])
    #         train_targets.append(class_targets[train_indx])

    #     if appendent is not None:
    #         appendent_data, appendent_targets = appendent
    #         for idx in range(0, int(np.max(appendent_targets))+1):
    #             append_data, append_targets = self._select(appendent_data, appendent_targets,
    #                                                        low_range=idx, high_range=idx+1)
    #             val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
    #             train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
    #             val_data.append(append_data[val_indx])
    #             val_targets.append(append_targets[val_indx])
    #             train_data.append(append_data[train_indx])
    #             train_targets.append(append_targets[train_indx])

    #     train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
    #     val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

    #     return DummyDataset(train_data, train_targets, trsf, self.use_path), \
    #         DummyDataset(val_data, val_targets, trsf, self.use_path)



    def _setup_data(self, dataset_name, shuffle, seed):
        # 得到不同数据集的配置情况， 类别 标注

        idata = _get_idata(dataset_name, self.args)
        idata.download_data()


        self._train_data = idata.train_data
        self._test_data = idata.test_data

        # self._target = idata.raw_gt
        self.label = idata.labels
        
        if dataset_name == "AVE":
            self.train_gt = idata.raw_gt
            self.test_gt = idata.raw_gt
        elif dataset_name == "LLP":
            self.train_gt = idata.train_gt
            self.test_gt = idata.test_gt
        elif dataset_name == "AVQA":
            self.train_gt = idata.train_gt
            self.test_gt = idata.test_gt
            self.word_to_ix = idata.word_to_ix
            self.ans_vocab = idata.train_ans_vocab
        elif dataset_name == "AVS":
            self.train_gt = idata.train_gt
            self.test_gt = idata.test_gt


        self.trsf = idata.trsf
        self.data_path = idata.root_path


        order = [i for i in range(len(self._train_data.keys()))]
        # print()

        # TODO: shuffle
        # if shuffle:
        #     np.random.seed(seed)
        #     order = np.random.permutation(len(order)).tolist()
        # else:
        #     order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)
        

        # Data
        # self._train_data, self._train_targets = idata.train_data, idata.train_targets
        # self._test_data, self._test_targets = idata.test_data, idata.test_targets
        # self.use_path = idata.use_path

        # Transforms
        # self._train_trsf = idata.train_trsf
        # self._test_trsf = idata.test_trsf
        # self._common_trsf = idata.common_trsf

        # Order
        # order = [i for i in range(len(np.unique(self._train_targets)))]
        # if shuffle:
        #     np.random.seed(seed)
        #     order = np.random.permutation(len(order)).tolist()
        # else:
        #     order = idata.class_order
        # self._class_order = order
        # logging.info(self._class_order)

        # Map indices
        # self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        # self._test_targets = _map_new_class_index(self._test_targets, self._class_order)



    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]




class DummyDataset(Dataset):
    def __init__(self, dataset_name, data, labels, trsf, root_path, raw_gt, ans_vocab=None, word_to_ix=None, split='train'):
        self.dataset_name = dataset_name
        self.data_idx = data
        self.labels = labels

        self.normalize = trsf['normalize']
        self.norm_mean = trsf['norm_mean']
        self.norm_std = trsf['norm_std']

        self.root_path = root_path
        self.raw_gt = raw_gt

        self.ans_vocab = ans_vocab
        self.word_to_ix = word_to_ix
        self.split = split
        

    def __len__(self):
        return len(self.data_idx)
        # return 3


    def __getitem__(self, idx):
        real_idx = self.data_idx[idx]
        
        # Determine file name and other parameters based on dataset
        if self.dataset_name == "AVE":
            return self._get_ave_item(real_idx)
        elif self.dataset_name == "LLP":
            return self._get_llp_item(real_idx)
        elif self.dataset_name == "AVQA":
            return self._get_avqa_item(real_idx)
        elif self.dataset_name == "AVS":
            return self._get_avs_item(real_idx)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")


    def _get_ave_item(self, real_idx):
        file_name = self.raw_gt.iloc[real_idx].iloc[1]
        
        # Load image and audio
        total_img = self._load_standard_frames(file_name)
        wave = self._load_standard_audio(file_name)
        
        return {
            'GT': self.labels[real_idx],
            'image': total_img,
            'wave': wave
        }

    def _get_llp_item(self, real_idx):
        row = self.raw_gt.loc[real_idx, :]
        file_name = row.iloc[0][:11]
        ids = row.iloc[-1].split(',')     
        label = self.ids_to_multinomial(ids)
        
        # Load image and audio
        total_img = self._load_standard_frames(file_name)
        wave = self._load_standard_audio(file_name)
        
        return {
            'GT': label,
            'wave': wave,
            'image': total_img
        }
    
    def _get_avqa_item(self, real_idx):
        sample = self.raw_gt[real_idx]
        file_name = sample['video_id']
        
        # Load image and audio
        total_img = self._load_standard_frames(file_name)
        wave = self._load_standard_audio(file_name)
        
        # Question processing
        question_id = sample['question_id']
        question = sample['question_content'].rstrip().split(' ')
        question[-1] = question[-1][:-1]

        self.max_len = 14  # question length

        p = 0
        for pos in range(len(question)):
            if '<' in question[pos]:
                question[pos] = ast.literal_eval(sample['templ_values'])[p]
                p += 1
        if len(question) < self.max_len:
            n = self.max_len - len(question)
            for i in range(n):
                question.append('<pad>')
        idxs = [self.word_to_ix[w] for w in question]
        ques = torch.tensor(idxs, dtype=torch.long)

        # Answer processing
        answer = sample['anser']
        label = self.ids_to_multinomial(answer, self.ans_vocab)
        label = torch.from_numpy(np.array(label)).long()
        
        return {
            'image': total_img,
            'question': ques,
            'GT': label,
            'wave': wave
        }

    def _get_avs_item(self, real_idx):
        if not isinstance(self.raw_gt, pd.DataFrame):
            raise ValueError("Expected raw_gt to be a pandas DataFrame for AVS dataset")
            
        row = self.raw_gt.iloc[real_idx]
        video_name = row[0]  # Assuming video name is in first column
        category = row['category']
        
        mask_num = 1 if self.split == 'train' else 5

        # Load images - AVS specific pathway
        total_img, gt_mask = self._load_avs_frames(video_name, category, mask_num)
        
        # Load audio - AVS specific pathway
        audio_log_mel, wave = self._load_avs_audio(video_name, category)
        
        return {
            'image': total_img,
            'wave': wave,
            'audio_lm': audio_log_mel,
            'GT': gt_mask,
            'category': category,
            'video_name': video_name
        }

    def _load_standard_frames(self, file_name):
        # Standard frame loading for most datasets
        total_num_frames = len(glob.glob(os.path.join(self.root_path, 'frames') + '/' + file_name + '/*.jpg'))
        
        total_img = []
        for vis_idx in range(10):
            frame_path = os.path.join(self.root_path, 'frames', file_name, f"{vis_idx+1:08d}.jpg")
            tmp_img = torchvision.io.read_image(frame_path) / 255
            tmp_img = self.normalize(tmp_img)
            total_img.append(tmp_img)
            
        return torch.stack(total_img)

    def _load_standard_audio(self, file_name):
        # Standard audio loading for most datasets
        wave_path = os.path.join(self.root_path, 'wave')
        file_path = os.path.join(wave_path, f"{file_name}.npy")
        wave = np.load(file_path, allow_pickle=True)
        wave = torch.from_numpy(wave)
        wave = wave.view(10, 32000)
        
        # Ensure enough audio data
        while wave.size(-1) < 32000 * 10:
            wave = torch.cat((wave, wave), dim=-1)
        wave = wave[:, :32000 * 10]
        
        return wave

    def _load_avs_frames(self, video_name, category, mask_num):
        """加载AVS帧并包含掩码加载作为ground truth"""
        # 基于S4Dataset的__getitem__方法
        img_base_path = os.path.join(self.root_path, 's4_data/visual_frames', 
                                self.split, category, video_name)
        
        # 掩码基路径 - 从S4Dataset加载掩码的方式
        mask_base_path = os.path.join(self.root_path, 's4_data/gt_masks', 
                                    self.split, category, video_name)
        
        total_img = []
        masks = []
        
        for img_id in range(1, 6):  # S4Dataset loads 5 frames
            # 加载图像
            img_path = os.path.join(img_base_path, f"{video_name}_{img_id}.png")
            
            # try:
            # 加载图像使用PIL
            img_PIL = Image.open(img_path).convert('RGB')
            img = self.normalize(transforms.ToTensor()(img_PIL))
            total_img.append(img)
            
            # 加载对应的掩码
        for mask_id in range(1, mask_num+1):
            mask_path = os.path.join(mask_base_path, f"{video_name}_{mask_id}.png")
            mask = Image.open(mask_path).convert('1')  # 二值图像模式
            mask_tensor = transforms.ToTensor()(mask)
            masks.append(mask_tensor)
            
            # except FileNotFoundError:
            #     # 如果PNG不存在，尝试JPG (原始代码中没有这个后备方案，但增加健壮性)
            #     try:
            #         img_path = os.path.join(img_base_path, f"{video_name}_{img_id}.jpg")
            #         img_PIL = Image.open(img_path).convert('RGB')
            #         img = self.normalize(transforms.ToTensor()(img_PIL))
            #         total_img.append(img)
                    
            #         # 同样尝试JPG格式的掩码
            #         mask_path = os.path.join(mask_base_path, f"{video_name}_{img_id}.jpg")
            #         if not os.path.exists(mask_path):
            #             # 有时掩码可能仍为PNG即使图像是JPG
            #             mask_path = os.path.join(mask_base_path, f"{video_name}_{img_id}.png")
            #         mask = Image.open(mask_path).convert('1')
            #         mask_tensor = transforms.ToTensor()(mask)
            #         masks.append(mask_tensor)
            #     except Exception as e:
            #         print(f"无法加载图像或掩码: {e}")
            #         # 创建空图像和掩码作为后备
            #         img = torch.zeros(3, 224, 224)
            #         mask_tensor = torch.zeros(1, 224, 224)
            #         total_img.append(img)
            #         masks.append(mask_tensor)
        
        # 堆叠图像和掩码沿第一个维度
        total_img = torch.stack(total_img, dim=0)  # [5, 3, 224, 224]
        masks = torch.stack(masks, dim=0)  # [5, 1, 224, 224]
        
        return total_img, masks

    def _load_audio_lm(self, audio_lm_path):
        with open(audio_lm_path, 'rb') as fr:
            audio_log_mel = pickle.load(fr)
        audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
        return audio_log_mel

    def _load_avs_audio(self, video_name, category):
        
        audio_lm_path = os.path.join(self.root_path, 's4_data', 'audio_log_mel', self.split, category, video_name + '.pkl')
        audio_log_mel = self._load_audio_lm(audio_lm_path)

        wave_path = os.path.join(self.root_path, 's4_data/wave', 
                                self.split, category, 'AVS.npy')
        
        wave_dict = np.load(wave_path, allow_pickle=True).item()
        wave = wave_dict[f"{video_name}.wav"]
        wave = torch.from_numpy(wave)
        wave = wave.view(5, 32000)  # Reshape to expected dimensions
            
        # except (FileNotFoundError, KeyError):
        #     # Fallback to direct audio processing using torchaudio
        #     try:
        #         audio_path = os.path.join(self.root_path, 's4_data/audio_wav',
        #                                 self.split, category, f"{video_name}.wav")
                
        #         # This follows _wav2fbank method in S4Dataset
        #         waveform, sr = torchaudio.load(audio_path)
        #         waveform = waveform - waveform.mean()
                
        #         # Process audio similarly to S4Dataset
        #         sample_indices = np.linspace(0, waveform.shape[1] - sr*10.1, num=5, dtype=int)
                
        #         wave_segments = []
        #         for idx in range(5):
        #             segment = waveform[:, sample_indices[idx]:sample_indices[idx]+int(sr*10.0)]
        #             wave_segments.append(segment)
                
        #         wave = torch.stack(wave_segments)
        #         wave = wave.view(5, -1)  # Reshape to get 5 segments
                
        #         # Ensure consistent length by padding/truncating
        #         if wave.shape[1] < 32000:
        #             padding = torch.zeros(5, 32000 - wave.shape[1])
        #             wave = torch.cat([wave, padding], dim=1)
        #         else:
        #             wave = wave[:, :32000]
                
        #     except Exception as e:
        #         print(f"Error loading audio for {video_name}: {e}")
        #         # Create dummy audio as fallback
        #         wave = torch.zeros(5, 32000)
        
        # Ensure we have the right shape
        wave = wave.view(5, 32000)
        while wave.size(-1) < 32000 * 5:
            wave = torch.cat((wave, wave), dim=-1)     
        wave = wave[:, :32000*5]
        
        return audio_log_mel, wave

    # def __getitem__(self, idx):
        
    #     # video
    #     file_name = None

    #     real_idx = self.data_idx[idx]
    #     if self.dataset_name == "AVE":
    #         file_name = self.raw_gt.iloc[real_idx].iloc[1]
    #     elif self.dataset_name == "LLP":
    #         row = self.raw_gt.loc[real_idx, :]
    #         file_name = row.iloc[0][:11]
    #         ids = row.iloc[-1].split(',')     
    #         label = self.ids_to_multinomial(ids)
    #     elif self.dataset_name == "AVQA":
    #         # print(self.data_idx)
    #         sample = self.raw_gt[real_idx] 
    #         file_name = sample['video_id']
    #     elif self.dataset_name == "AVS":
    #         row = self.raw_gt.iloc[real_idx]
    #         file_name = row[0]  # First column is the video name in s4_meta_data.csv
    #         category = row['category']


    #     total_num_frames = len(glob.glob(os.path.join(self.root_path, 'frames') + '/' + file_name + '/*.jpg'))

    #     # sample_indx = np.linspace(1, total_num_frames, num=10, dtype=int)
    #     total_img = []
    #     for vis_idx in range(10):
    #         # tmp_idx = sample_indx[vis_idx]
    #         tmp_img = torchvision.io.read_image(
    #             os.path.join(self.root_path,'frames') + '/' + file_name + '/' + str("{:08d}".format(vis_idx+1)) + '.jpg') / 255
    #         tmp_img = self.normalize(tmp_img)
    #         total_img.append(tmp_img)
    #     total_img = torch.stack(total_img)


    #     if self.dataset_name == "AVQA":
    #         # question
    #         question_id = sample['question_id']
    #         question = sample['question_content'].rstrip().split(' ')
    #         question[-1] = question[-1][:-1]

    #         self.max_len = 14  # question length

    #         p = 0
    #         for pos in range(len(question)):
    #             if '<' in question[pos]:
    #                 question[pos] = ast.literal_eval(sample['templ_values'])[p]
    #                 p += 1
    #         if len(question) < self.max_len:
    #             n = self.max_len - len(question)
    #             for i in range(n):
    #                 question.append('<pad>')
    #         idxs = [self.word_to_ix[w] for w in question]
    #         ques = torch.tensor(idxs, dtype=torch.long)

    #         # answer
    #         answer = sample['anser']
    #         label = self.ids_to_multinomial(answer, self.ans_vocab)
    #         label = torch.from_numpy(np.array(label)).long()


    #     # audio wave
    #     wave_path = os.path.join(self.root_path, 'wave')
    #     file_name += '.npy'
    #     filepath = os.path.join(wave_path, file_name)
    #     wave = np.load(filepath, allow_pickle=True)
    #     wave = torch.from_numpy(wave)
    #     wave = wave.view(10, 32000)
    #     while wave.size(-1) < 32000 * 10:
    #         wave = torch.cat((wave, wave), dim=-1)
    #     wave = wave[:, :32000 * 10]
    #     # 需要查看一下wave的shape
    #     if self.dataset_name == "AVE":
    #         return {
    #                 'GT': self.labels[real_idx],
    #                 'image': total_img,
    #                 'wave': wave
    #                 }
    #     elif self.dataset_name == "LLP":
    #         return {
	# 				'GT': label, 
	# 				'wave': wave,
	# 				'image':total_img
	# 		}
    #     elif self.dataset_name == "AVQA":
    #         return{
    #             'image': total_img,
	# 			 # 'visual_nega': sample['visual_nega'],
	# 			'question': ques,
	# 			'GT': label,
	# 			'wave': wave
    #         }
    #     elif self.dataset_name == "AVS":
    #         return {
    #             'GT': self.labels[real_idx] if self.labels is not None else None,
    #             'image': total_img,
    #             'wave': wave
    #         }



    def ids_to_multinomial(self, ids, categories=None):
        """ label encoding

        Returns:
        1d array, multimonial representation, e.g. [1,0,1,0,0,...]
        """
        if self.dataset_name == "LLP":
            categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                        'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                        'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                        'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                        'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                        'Clapping']
            id_to_idx = {id: index for index, id in enumerate(categories)}
            y = np.zeros(len(categories))
            for id in ids:
                index = id_to_idx[id]
                y[index] = 1
            return y

        elif self.dataset_name == "AVQA":
            id_to_idx = {id: index for index, id in enumerate(categories)}

            with open('AVQA_cat.json', 'w', encoding='utf-8') as file:
                json.dump(id_to_idx, file, ensure_ascii=False, indent=4)

            return id_to_idx[ids]

        


# def _map_new_class_index(y, order):
#     return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "ave":
        return AVE(args)
    elif name == "llp":
        return LLP(args)
    elif name == "avqa":
        return AVQA(args)
    elif name == "avs":
        return AVS(args)
    # elif name == 'core50':
    #     return iCore50(args)
    # elif name == 'domainnet':
    #     return iDomainNet(args)
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))
