# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

import pickle

import torch.utils.data as data
import torchvision.datasets as datasets


class WholeDataLoader(Dataset):
    def __init__(self,option):
        self.data_split = option.data_split
        if option.data_split == 'train':
            data_list = ['data_batch_%d'%(i+1) for i in range(5)]
            self.T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(option.input_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822,0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif option.data_split == 'test':
            data_list = ['test_batch']
            self.T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822,0.4465), (0.2023, 0.1994, 0.2010)),
            ])


        self.data = []
        self.label = []
        for filename in data_list:
            filepath = os.path.join(os.path.join(option.data_dir,filename))
            with open(filepath, 'rb') as f:
                entry = pickle.load(f,encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.label.extend(entry['labels'])
                else:
                    self.label_list.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1,3,option.input_size,option.input_size)
        self.data = self.data.transpose((0,2,3,1)) #NHWC

        self.label = np.array(self.label)
        



    def __getitem__(self,index):
        label = self.label[index]
        image = self.data[index]



        
        return self.T(image), label.astype(np.long)




    def __len__(self):
        return self.data.shape[0]






if __name__ == '__main__':
    class Option(object):
        pass
    option = Option()
    option.data_dir = '../../dataset/cifar10'
    option.input_size = 32
    option.batch_size = 8
    option.num_workers = 2
    option.data_split='test'
    
    loader = WholeDataLoader(option)
    print(loader.data.shape)
    img,lab = loader.__getitem__(3)

    print(type(loader.data[0,0,0,0]))
    print(type(loader.label)) # list
