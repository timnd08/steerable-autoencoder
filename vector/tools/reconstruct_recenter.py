#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt

import torch

from torchvision.transforms import transforms

import sys
sys.path.append("./")

import utils
import models.builer as builder
import dataloader

import os

import random
import PIL.Image as Image

import PIL.ImageDraw as ImageDraw
import math


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(42)

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--val_list', type=str)              
    
    args = parser.parse_args()

    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def imgPrinter(img_name,random_base_dir, start_loc):
    curr_vector = img_name.split('.')[0]
    start_loc.append((curr_vector.split('_')[-2], curr_vector.split('_')[-1]))
    vect_cent = 112.0
    vecX = int(curr_vector.split('_')[-2]) - vect_cent
    vecY = int(curr_vector.split('_')[-1]) - vect_cent
    src_img_path = os.path.join(random_base_dir, img_name)
    src_img = Image.open(src_img_path).convert('RGB')
    src_img = transforms.ToTensor()(src_img).unsqueeze(0).cuda()
    vector = (vecX, vecY)
    x = vector[0] / 112.0
    y = vector[1] / 112.0
    return src_img, torch.tensor([[x, y]], dtype=torch.float32)
    

def addArrow(src_img_pil, center_x, center_y, object_x, object_y):
    draw = ImageDraw.Draw(src_img_pil)
    draw.line((center_x, center_y, object_x, object_y), fill=(255, 0, 0), width=3)
    angle = math.atan2(object_y - center_y, object_x - center_x)
    tip = (object_x, object_y)
    left_flank = (object_x - 20 * math.cos(angle - math.pi / 6), object_y - 20 * math.sin(angle - math.pi / 6))
    right_flank = (object_x - 20 * math.cos(angle + math.pi / 6), object_y - 20 * math.sin(angle + math.pi / 6))
    draw.polygon([tip, left_flank, right_flank], fill=(255, 0, 0))
    return src_img_pil


def main(args,u):
    print('=> torch version : {}'.format(torch.__version__))

    #utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)
    
    # print('=> building the dataloader ...')
    # train_loader = dataloader.val_loader(args)

    plt.figure(figsize=(16, 9))

    model.eval()
    print('=> reconstructing ...')

    # randomly select an image from the val set
    with open(args.val_list, "r") as f:
        image_dirs = [line.strip().split(' ')[0] for line in f.readlines()]
    print(f"=> total number of images in val set: {len(image_dirs)}")

    random_base_dir = image_dirs[random.randint(0, len(image_dirs)-1)]
    #random_img_name = os.listdir(random_base_dir)[random.randint(0, len(os.listdir(random_base_dir))-1)] 
    # while random_img_name.split('.')[0].split('_')[-2:] == ['112', '112']:
    #     random_img_name = os.listdir(random_base_dir)[random.randint(0, len(os.listdir(random_base_dir))-1)]
    
    start_loc = []
    src_imgs = []
    vectors_tensor = []
    print(len(os.listdir(random_base_dir)))
    
    base_dir_lists = os.listdir(random_base_dir)
    img_name = "_".join(base_dir_lists[0].split('.')[0].split('_')[:-2]) + '_112_112.png'
    src_img, vector_tensor = imgPrinter(img_name, random_base_dir, start_loc)
    src_imgs.append(src_img)
    vectors_tensor.append(vector_tensor)
    base_dir_lists = [img for img in base_dir_lists if img != img_name]

    for img_name in base_dir_lists:
        if img_name.split('.')[-1] != 'png':
            continue
        src_img, vector_tensor = imgPrinter(img_name, random_base_dir, start_loc)
        src_imgs.append(src_img)
        vectors_tensor.append(vector_tensor)

    # reconstruction loss mask
    # img_name = "_".join(base_dir_lists[0].split('.')[0].split('_')[:-2]) + '_112_112.avif'
    # src_img, vector_tensor = imgPrinter(img_name, random_base_dir, start_loc)
    # src_imgs.append(src_img)
    # vectors_tensor.append(vector_tensor)

    
    # set up subplots
    _, axes =  plt.subplots(2, len(vectors_tensor), figsize=(32, 20))
    for i in range(len(vectors_tensor)):
        axes[0,i].axis('off')
        axes[1,i].axis('off')
    
    # center_ax = axes[0,0]
    # center_ax.imshow(transforms.ToPILImage()(src_img.squeeze().cpu()))
    # center_ax.set_title("Source Image")

    print("=> generating transformations ...")

    src_img = src_img.cuda(non_blocking=True)
    
    axes[0,0].set_ylabel(f"Pre-saccade", fontsize=40)
    axes[1,0].set_ylabel(f"Remapped", fontsize=40)

    
    for i, vec in enumerate(vectors_tensor):
        src_img = src_imgs[i]
        loc = start_loc[i]
        #curr_vector = os.listdir(random_base_dir)[i].split('.')[0]
        vecX, vecY = vec.squeeze().cpu().numpy() * 112.0
        src_img_pil = transforms.ToPILImage()(src_img.squeeze().cpu())
        

        #axes[0,i].set_title(f"Object at location ({loc[0]}, {loc[1]})", fontsize=25)
        if vecX == 0 and vecY == 0:
            axes[0,i].set_title(f"Centered", fontsize=60, pad=20)

        else:
            axes[0,i].set_title(f"{'Right' if vecX > 0 else 'Left'}, {'Down' if vecY > 0 else 'Up'}", fontsize=60, pad=20)
            src_img_pil = addArrow(src_img_pil, 112, 112, vecX + 112, vecY + 112)

        axes[0,i].imshow(src_img_pil)
        axes[0,i].axis('on')
        axes[0,i].set_xticks([])
        axes[0,i].set_yticks([])
        #axes[0,i].xaxis.set_ticks_position('top')
        vec = vec.cuda(non_blocking=True)
        out = model(src_img, vec)
        out = transforms.ToPILImage()(out.squeeze().cpu())       

        #vecX, vecY = vec.squeeze().cpu().numpy() * 112.0
        axes[1,i].imshow(out)
        axes[1,i].set_xlabel(f"Centered Output", fontsize=60)
        #axes[1,i].set_xlabel(f"Eye shift vector: ({vecX:.2f}, {vecY:.2f})  \n {'Right' if vecX > 0 else 'Left'} {abs(vecX):.2f}, {'Down' if vecY > 0 else 'Up'} {abs(vecY):.2f} ", fontsize=25)
        axes[1,i].axis('on')
        axes[1,i].set_xticks([])
        axes[1,i].set_yticks([])
        #axes[1,i].xaxis.set_ticks_position('top')
    #171 #104
    #plt.suptitle(f"Autoencoder Reconstruction (epoch: {int(str(args.resume).split('/')[-1].split('.')[0])-299})", fontsize=30)
    

    # #remove top right plot
    # axes[1,2].imshow(axes[0,2].get_images()[0].get_array())
    # axes[1,2].set_xlabel(f"Mask", fontsize=60)
    # plt.delaxes(axes[0,2])

    axes[0,0].set_ylabel(f"Pre-saccade", fontsize=60, labelpad=20)
    axes[1,0].set_ylabel(f"Remapped", fontsize=60, labelpad=20)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.savefig(f"figs/recent/reconstruction_vector_recent_{u}.png")
  
    #print(model)
    




if __name__ == '__main__':

    args = get_args()
    for i in range(30):
        main(args,i)

