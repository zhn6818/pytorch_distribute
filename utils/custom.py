import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *
import cv2

#from .base_dataset import BaseDataset
#cropsize=(2048, 1024)
cropsize = (1280,640) #默认
cropsize2 = (40, 20)
# cropsize=(512, 512)
#cropsize=(512, 256)
#cropsize = (640,640)
class Custom(Dataset):
    def __init__(self, datapath, cropsize=cropsize, 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(Custom, self).__init__(*args, **kwargs)
 
        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            # ColorJitter(
            #     brightness = 0.5,
            #     contrast = 0.5,
            #     saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            # RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            # RandomCrop(cropsize)
            ])
        self.trans_train3 = Compose([
            HorizontalFlip(),
            ])
        self.trans_train2 = Compose([
            # ColorJitter(
            #     brightness = 0.5,
            #     contrast = 0.5,
            #     saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            # RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            # RandomCrop(cropsize2)
            ])
        
        self.class_weights = torch.FloatTensor([0.8,1.1]).cuda()
        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
        #                                 1.0166, 0.9969, 0.9754, 1.0489,
        #                                 0.8786, 1.0023, 0.9539, 0.9843, 
        #                                 1.1116, 0.9037, 1.0865, 1.0955, 
        #                                 1.0865, 1.1529, 1.0507]).cuda()
        self.img_paths = []
        self.gt_paths = []

        with open(datapath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\r\n")
                if not os.path.exists(line):
                    continue
                imgname = line.split("/")[-1]
                imgfolder = line.split("/")[-2]
                no_train_folder = [3,10]
                #print((imgfolder.split('image')[-1]).split('_')[-1])
                if int((imgfolder.split('image')[-1]).split('_')[-1]) in no_train_folder:
                     continue
                maskfolder = 'mask'+imgfolder.split('image')[-1]
                #road 
                #gtline = '/'.join(line.split("/")[0:-3])+'/'+'/mask_new/'+maskfolder+'/'+imgname.split('.jpg')[0]+'.png'
                #fog 
                gtline = '/'.join(line.split("/")[0:-3])+'/'+'/mask/'+maskfolder+'/'+imgname
                self.img_paths.append(line)
                self.gt_paths.append(gtline)
    # def __getitem__(self, index):
    #     img_path = self.img_paths[index]
    #     gt_path = self.gt_paths[index]
    #     # print('gt_path---------------:',gt_path)
    #     img = Image.open(img_path).convert('RGB')
    #     label = cv2.imread(gt_path, flags=cv2.IMREAD_GRAYSCALE)
    #     img = img.resize(cropsize)
    #     label = cv2.resize(label,cropsize2)
    #     label[label>128] = 1
    #     # label[label<128] = 0
    #     label = Image.fromarray(label)
    #     # im_lb = dict(im = img, lb = label)
    #     # im_lb = self.trans_train3(im_lb)
    #     # img, label = im_lb['im'], im_lb['lb']
        
    #     label2 = np.array(label).astype("bool")#.astype()
    #     label3 = (~label2).astype(np.float32)
        
    #     label = np.array(label).astype(np.int64)
        
    #     img = self.to_tensor(img)
        
    #     return img, label
        

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        img = Image.open(img_path).convert('RGB')

        # print('gt_path---------------:',gt_path)
        label = cv2.imread(gt_path,0)
        label = cv2.resize(label,cropsize2)
        label = label / 255.0 
        label[label>0.5] = 1
        label[label<0.5] = 0
        ##road train开启
        img = img.resize(cropsize)
        
        ###
        label = Image.fromarray(label)
  
        im_lb = dict(im = img, lb = label)
        im_lb = self.trans_train(im_lb)
        
        
        # img_lb2 = dict(im = img, lb = label)
        # img_lb2 = self.trans_train2(img_lb2)
        
        
        img, label = im_lb['im'], im_lb['lb']
        
        
        img = self.to_tensor(img)
        
        label = np.array(label).astype(np.int64)
        
        
        
        # label2 = img_lb2['lb']
        # label2 = np.array(label2).astype(np.int64)

        return img, label


    def __len__(self):
        return len(self.img_paths)

