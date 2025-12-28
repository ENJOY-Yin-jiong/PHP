import os

dir_path = '/opt/data/private/dataset/AVS/Single-source/s4_data/gt_masks/train'

labels_list = os.listdir(dir_path)

with open('categories.txt', 'w') as f:
    for label in labels_list:
        f.write(label + '\n')
