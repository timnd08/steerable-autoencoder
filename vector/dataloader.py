#!/usr/bin/env python

import random

from PIL import Image, ImageFilter

import torch
import torch.utils.data as data
import os
from tqdm import tqdm

from torchvision.transforms import transforms

class ImageDataset(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = transform
        self.init()

    def init(self):
        self.list_img1_paths = []

        with open(self.ann_file, 'r') as f:
            im_folders = [line.strip().split(' ')[0] for line in f.readlines()]
            #im_folders = im_folders[:100] # for debugging, remove this line for full training
        
        for im_folder in tqdm(im_folders):
            list_img1_names = os.listdir(im_folder)
            if not list_img1_names:
                continue
            
            list_img1_paths_folder = [os.path.join(im_folder, name) for name in list_img1_names]

            self.list_img1_paths.extend(list_img1_paths_folder)
            
        print(f"Total number of samples: {len(self.list_img1_paths)}")


    def __getitem__(self, index):
        img1_path = self.list_img1_paths[index]
        img_base_name = "_".join(img1_path.split('/')[-1].split('.')[0].split('_')[:-2])

        img2_path = os.path.join('/'.join(img1_path.split('/')[:-1]), f"{img_base_name}_112_112.png")

        mask_path = os.path.join('/'.join(img1_path.split('/')[:-1]), f"{img_base_name}_112_112.avif")
        
        vector = img1_path.split('/')[-1].split('.')[0]
        vecX = (int(vector.split('_')[-2]) - 112) / 112.0
        vecY = (int(vector.split('_')[-1]) - 112) / 112.0
        vector = torch.tensor([vecX, vecY], dtype=torch.float32)

        with Image.open(img1_path) as img1:
            img1 = img1.convert('RGB')
            img1 = self.transform(img1)
        

        with Image.open(img2_path) as img2:
            img2 = img2.convert('RGB')
            img2 = self.transform(img2)

        #convert mask to binary mask, then convert to tensor
        if os.path.exists(mask_path):
            with Image.open(mask_path) as mask:
                mask = mask.convert('L')
                mask = transforms.ToTensor()(mask)
                mask = (mask > 0).float()
        else:
            mask = torch.ones((1, 224, 224), dtype=torch.float32)

        return img1, img2, mask, vector

    def __len__(self):
        return len(self.list_img1_paths)

def train_loader(args):
    augmentation = [
        transforms.ToTensor(),
    ]

    train_trans = transforms.Compose(augmentation)

    train_dataset = ImageDataset(args.train_list, transform=train_trans)   
    
    if args.parallel == 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_dataset,
                            rank=args.rank,
                            num_replicas=args.world_size,
                            shuffle=True)         
    else:  
        train_sampler = None    

    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=(train_sampler is None),
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True,
                    sampler=train_sampler,
                    drop_last=(train_sampler is None))

    return train_loader

def val_loader(args):

    val_trans = transforms.Compose([
                    transforms.ToTensor()
                  ])

    val_dataset = ImageDataset(args.val_list, transform=val_trans)      

    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    pin_memory=True)

    return val_loader 

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x