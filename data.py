import numpy as np
import os
import h5py
import pandas as pd
from torchvision.transforms import Compose, Resize, Normalize
from PIL import Image
import json
import ast


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def generate_category_list(root_path, dataset_name):
    if dataset_name == "vggsound":
        file_path = os.path.join(root_path, 'data/vggsound/VggsoundAVEL40kCategories.txt')
    elif dataset_name == "AVE":
        file_path = '/opt/data/private/dataset/AVE/labels/categories.txt'
    elif dataset_name == "LLP":
        categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
        id_to_idx = {id: index for index, id in enumerate(categories)}
        
        return categories, id_to_idx
        
        # with open(dataset_name + '_cat.json', 'w', encoding='utf-8') as file:
        #     json.dump(id_to_idx, file, ensure_ascii=False, indent=4)
    elif dataset_name == 'AVQA':
        type_categories = ['Counting', 'Existential', 'Location', 'Comparative', 'Temporal']
        id_to_idx = {id: index for index, id in enumerate(type_categories)}
        return category_list, id_to_idx
    
    elif dataset_name == 'AVS':
        # Read categories from CSV for AVS
        df_all = pd.read_csv('/opt/data/private/dataset/AVS/Single-source/s4_data/s4_meta_data.csv', sep=',')
        categories = df_all['category'].unique().tolist()
        id_to_idx = {id: index for index, id in enumerate(categories)}

        return categories, id_to_idx
    else:
        raise NotImplementedError
        
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    id_to_idx = {id: index for index, id in enumerate(category_list)}

    # with open(dataset_name + '_cat.json', 'w', encoding='utf-8') as file:
    #     json.dump(id_to_idx, file, ensure_ascii=False, indent=4)

    return category_list, id_to_idx



class iData(object):
    class_order = None


class AVE(iData):
    def __init__(self, args):
        self.args = args
        class_order = np.arange(28).tolist()
        self.class_order = class_order

        self.root_path = '/opt/data/private/dataset/AVE/'

        self.tasks_name, self.id_to_idx = generate_category_list(self.root_path, 'AVE')


    def download_data(self):
        # self.image_list_root = self.args["data_path"]
        # self.image_list_root = 'data/'

        train_dataset = []
        test_dataset = []

        # with h5py.File(os.path.join(self.root_path, 'data/AVE/mil_labels.h5'), 'r') as hf:
        #     self.mil_labels = hf['avadataset'][:]

        with h5py.File(os.path.join(self.root_path, 'labels/labels.h5'), 'r') as hf:
            self.labels = hf['avadataset'][:]
            
        with h5py.File(os.path.join(self.root_path, 'labels/train_order.h5'), 'r') as hf:
            train_dataset = hf['order'][:]
        with h5py.File(os.path.join(self.root_path, 'labels/test_order.h5'), 'r') as hf:
            test_dataset = hf['order'][:]
        
        self.raw_gt = pd.read_csv(os.path.join(self.root_path, "labels/Annotations_with_domain.txt"), sep="&")

        self.train_data = dict()
        self.test_data = dict()

        for i in range(len(self.id_to_idx)):
            self.train_data[i] = []
            self.test_data[i] = []

        # print(self.id_to_idx)
        
        for idx in train_dataset: 
            # print(self.raw_gt.iloc[idx])
            cat = self.raw_gt.iloc[idx][0]
            self.train_data[self.id_to_idx[cat]].append(idx)

        for idx in test_dataset:
            name = self.raw_gt.iloc[idx][1]

            # 78xOH_VHip8   -DOcQ2NVnHE  -yGno5_ywmA
            # if name == '78xOH_VHip8':
            cat = self.raw_gt.iloc[idx][0]
            self.test_data[self.id_to_idx[cat]].append(idx)

        self.trsf = {}
        
        self.trsf['normalize'] = Compose([
                                    Resize([224,224], interpolation=Image.BICUBIC, antialias=True),
                                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                                    ])
        
        self.trsf['norm_mean'] = -4.1426
        self.trsf['norm_std'] = 3.2001

        

class LLP(iData):
    def __init__(self, args):
        self.args = args
        class_order = np.arange(24).tolist()
        self.class_order = class_order
        
        self.root_path = '/opt/data/private/dataset/AVVP/LLP_dataset'
        self.tasks_name, self.id_to_idx = generate_category_list(self.root_path, 'LLP')
        # self.categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
		# 				'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
		# 				'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
		# 				'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
		# 				'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
		# 				'Clapping']


    def download_data(self):
        
        self.audio_folder = os.path.join(self.root_path, 'audio')
        self.video_folder = os.path.join(self.root_path, 'frame')
        train_dataset = pd.read_csv(os.path.join(self.root_path, 'labels/AVVP_train.csv'), header=0, sep='\t')
        test_dataset = pd.read_csv(os.path.join(self.root_path, 'labels/AVVP_test_pd.csv'), header=0, sep='\t')

        self.train_gt = train_dataset
        self.test_gt = test_dataset

        # self.raw_gt = pd.read_csv(os.path.join(self.root_path, "labels/Annotations.txt"), sep="&")
        # print(test_dataset)

        self.train_data = dict()
        self.test_data = dict()
        for i in range(len(self.id_to_idx)):
            self.train_data[i] = []
            self.test_data[i] = []



        for idx in range(len(train_dataset)):
            row = train_dataset.loc[idx, :]
            file_name = row.iloc[0][:11]
            ids = row.iloc[-1].split(',')
            for cat in ids:
                self.train_data[self.id_to_idx[cat]].append(idx)

        for idx in range(len(test_dataset)):
            row = test_dataset.loc[idx, :]
            file_name = row.iloc[0][:11]
            # row = test_dataset.loc[idx, :]
            # ids = row.iloc[-1].split(',')
            # for cat in ids:
            #     self.test_data[self.id_to_idx[cat]].append(idx)
            
            # sample:  McK4y6_znE4   ORMBTAJOTQQ
            # if file_name == 'McK4y6_znE4':
            ids = row.iloc[-1].split(',')
            for cat in ids:
                self.test_data[self.id_to_idx[cat]].append(idx)




        self.labels = None

        self.trsf = {}
        
        # self.trsf['normalize'] = Compose([
        #                             	Resize([192, 192], interpolation=Image.BICUBIC),
		# 		                        Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        #                             ])
        
        # self.trsf['norm_mean'] = -4.984795570373535
        # self.trsf['norm_std'] = 3.7079780101776123
        
        self.trsf['normalize'] = Compose([
                                    Resize([224,224], interpolation=Image.BICUBIC, antialias=True),
                                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                                    ])
        
        self.trsf['norm_mean'] = -4.1426
        self.trsf['norm_std'] = 3.2001



class AVQA(iData):
    def __init__(self, args):
        self.args = args
        # class_order = np.arange(24).tolist()
        # self.class_order = class_order
        
        self.root_path = '/opt/data/private/dataset/AVQA'
        
        type_categories = ['Counting', 'Existential', 'Location', 'Comparative', 'Temporal']
        
        self.id_to_idx = {id: index for index, id in enumerate(type_categories)}

        # with open('AVQA_cat.json', 'w', encoding='utf-8') as file:
        #     json.dump(self.id_to_idx, file, ensure_ascii=False, indent=4)

        # 类别
        # self.tasks_name, self.id_to_idx = generate_category_list(self.root_path, 'LLP')
       

    def download_data(self):
        
        self.audio_folder = os.path.join(self.root_path, 'audio_wave')
        self.video_folder = os.path.join(self.root_path, 'frames')
        # train_dataset = pd.read_csv(os.path.join(self.root_path, 'labels/AVVP_train.csv'), header=0, sep='\t')
        # test_dataset = pd.read_csv(os.path.join(self.root_path, 'labels/AVVP_test_pd.csv'), header=0, sep='\t')

        train_dataset = json.load(open(os.path.join(self.root_path, 'labels/json/avqa-train.json'), 'r'))
        test_dataset  = json.load(open(os.path.join(self.root_path, 'labels/json/avqa-test.json'), 'r') )

        # nax =  nne
        train_ques_vocab = ['<pad>']
        train_ans_vocab = []
        i = 0
        for sample in train_dataset:
            i += 1
            question = sample['question_content'].rstrip().split(' ')
            question[-1] = question[-1][:-1]

            p = 0
            for pos in range(len(question)):
                if '<' in question[pos]:
                    question[pos] = ast.literal_eval(sample['templ_values'])[p]
                    p += 1

            for wd in question:
                if wd not in train_ques_vocab:
                    train_ques_vocab.append(wd)
            if sample['anser'] not in train_ans_vocab:
                train_ans_vocab.append(sample['anser'])

        self.train_ques_vocab = train_ques_vocab
        self.train_ans_vocab = train_ans_vocab
        self.word_to_ix = {word: i for i, word in enumerate(self.train_ques_vocab)}

        self.train_gt = train_dataset
        self.test_gt = test_dataset


        self.train_data = dict()
        self.test_data = dict()

        for i in range(len(self.id_to_idx)):
            self.train_data[i] = []
            self.test_data[i] = []


        for idx, sample in enumerate(train_dataset):
            type = sample["type"].strip("[]").replace("\"", '').replace(" ", '').split(",")[1]
            self.train_data[self.id_to_idx[type]].append(idx)

        for idx, sample in enumerate(test_dataset):
            video_id = sample['video_id']
            
            # 00000450    00002274   00002294   00003423   00006548
            # if video_id == '00000450':
            type = sample["type"].strip("[]").replace("\"", '').replace(" ", '').split(",")[1]
            self.test_data[self.id_to_idx[type]].append(idx)


        # for idx, sample in range(len(train_dataset)):
            
        #     row = train_dataset.loc[idx, :]
        #     file_name = row[0][:11]
        #     ids = row[-1].split(',')
        #     for cat in ids:
        #         self.train_data[self.id_to_idx[cat]].append(idx)

        # for idx in range(len(test_dataset)):
        #     row = test_dataset.loc[idx, :]
        #     file_name = row[0][:11]
        #     ids = row[-1].split(',')
        #     for cat in ids:
        #         self.test_data[self.id_to_idx[cat]].append(idx)


        self.labels = None

        self.trsf = {}
        
        self.trsf['normalize'] = Compose([
                                    Resize([224,224], interpolation=Image.BICUBIC, antialias=True),
                                    Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                                    ])
        
        self.trsf['norm_mean'] = -4.1426
        self.trsf['norm_std'] = 3.2001


class AVS(iData):
    def __init__(self, args):
        self.args = args
        self.root_path = '/opt/data/private/dataset/AVS/Single-source/'
        
        # Read categories from CSV
        df_all = pd.read_csv(os.path.join(self.root_path, 's4_data/s4_meta_data.csv'), sep=',')
        categories = df_all['category'].unique().tolist()
        self.id_to_idx = {id: index for index, id in enumerate(categories)}
        self.class_order = np.arange(len(categories)).tolist()
        self.tasks_name = categories

        self.labels = None
    def download_data(self):
        df_all = pd.read_csv(os.path.join(self.root_path, 's4_data/s4_meta_data.csv'), sep=',')
        
        self.train_data = dict()
        self.test_data = dict()
        
        # Initialize empty lists for each category
        for i in range(len(self.id_to_idx)):
            self.train_data[i] = []
            self.test_data[i] = []
        
        # Split data into train and test
        df_train = df_all[df_all['split'] == 'train']
        df_test = df_all[df_all['split'] == 'test']
        
        # Populate train data
        for idx, row in df_train.iterrows():
            category = row['category']
            self.train_data[self.id_to_idx[category]].append(idx)
            
        # Populate test data
        for idx, row in df_test.iterrows():
            category = row['category']
            self.test_data[self.id_to_idx[category]].append(idx)
            
        self.train_gt = df_all
        self.test_gt = df_all
        
        # Set up image transforms
        self.trsf = {}
        self.trsf['normalize'] = Compose([
            Resize([224, 224], interpolation=Image.BICUBIC, antialias=True),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        
        # Audio normalization parameters (from S4Dataset)
        self.trsf['norm_mean'] = -5.210531711578369
        self.trsf['norm_std'] = 3.5918314456939697