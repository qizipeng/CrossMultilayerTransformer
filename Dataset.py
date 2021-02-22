from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import math
import random
import pandas as pd
import cv2
from skimage import io
from Augmentations import Augmentations, Resize
import skimage


class Datasets(Dataset):
    def __init__(self, data_file, transform=None):
        self.transform = transform
        self.data_info = pd.read_csv(data_file, index_col=0)
        self.img_name=[]
        self.label_name=[]
        self.ori_img=[]
        self.ori_label=[]
        for index in self.data_info.index:
            data = self.data_info.iloc[index]
            self.img_name.append(data['img'])
            self.label_name.append(data['label'])
            self.ori_img.append(io.imread(data['img'], as_gray=False))
            self.ori_label.append(io.imread(data['label'], as_gray=True))
            #assert (self.ori_img[index] is not None and self.ori_label[index] is not None), f'{self.img_name[index]} or {self.label_name[index]} is not valid'
        print('Load dataset finished!!!!!!!!!!!!!!!!')

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        data = self.pull_item(index)
        return data

    def pull_item(self, index):
        """
        :param index: image index
        :return: torch.from_numpy(img).permute(2, 0, 1), bboxs, width, height
        """
        # data = self.data_info.iloc[index]
        # img_name = data['img']
        # label_name = data['label']
        # ori_img = io.imread(img_name, as_gray=False)
        # ori_label = io.imread(label_name, as_gray=True)
        # assert (ori_img is not None and ori_label is not None), f'{img_name} or {label_name} is not valid'
        ori_img = self.ori_img[index]
        ori_label = self.ori_label[index]
        img_name = self.img_name[index]
        if self.transform is not None:
            img, label = self.transform((ori_img, ori_label))
        one_hot_label = np.zeros([2] + list(label.shape), dtype=np.float)
        one_hot_label[0] = label == 0
        one_hot_label[1] = label > 0
        return {'img': torch.from_numpy(img).permute(2, 0, 1),
                'label': torch.from_numpy(one_hot_label),
                'ori_label': torch.from_numpy(label > 0).long(),
                'name': img_name
                }


def get_data_loader(args, config):
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': args.is_shuffle,
        'drop_last': False,
        'collate_fn': collate_fn,
        'num_workers': args.num_workers
    }
    #  data_file, config, transform=None
    train_set = Datasets(args.dataset, Augmentations(eval(config['IMG_SIZE']), config['PRIOR_MEAN'], config['PRIOR_STD'], 'train'))
    patterns = ['train']
    if args.is_val:
        val_params = {
            'batch_size': args.val_batch_size,
            'shuffle': False,
            'drop_last': False,
            'collate_fn': collate_fn,
            'num_workers': args.num_workers
        }
        val_set = Datasets(args.val_dataset,
                           Augmentations(eval(config['IMG_SIZE']), config['PRIOR_MEAN'], config['PRIOR_STD'], 'val'))
        patterns += ['val']

    data_loaders = {}
    for x in patterns:
        data_loaders[x] = DataLoader(eval(x+'_set'), **eval(x+'_params'))
    return data_loaders


def collate_fn(batch):
    def to_tensor(item):
        if torch.is_tensor(item):
            return item
        elif isinstance(item, type(np.array(0))):
            return torch.from_numpy(item).float()
        elif isinstance(item, type('0')):
            return item
        elif isinstance(item, list):
            return item

    data = {'img': [], 'label': [], 'ori_label': [], 'name': []}

    for sample in batch:
        for key, value in sample.items():
            data[key].append(to_tensor(value))
    keys = ['img', 'label', 'ori_label']
    for key in keys:
        data[key] = torch.stack(data[key], dim=0)

    return data