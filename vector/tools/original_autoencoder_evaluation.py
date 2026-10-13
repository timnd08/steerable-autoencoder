import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
import utils
import models.builder as builder
import os
import PIL.Image as Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import PIL.Image as Image
import torch.nn as nn
from torchvision.transforms import transforms
from pytorch_msssim import ssim, ms_ssim
from tqdm import tqdm

# get arguments
def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Autoencoder evaluator')
    parser.add_argument('--arch', default='vgg_original', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', default='results/imagenet-vgg16/imagenet-vgg16.pth', type=str)
    parser.add_argument('--val_list', default='list/imageNet_val_list.txt', type=str)           
    args = parser.parse_args()
    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

# function to load image directories from a list file
def loadImageDirs(image_list_file):
    with open(image_list_file, "r") as f:
        image_dirs = [line.strip().split(' ')[0] for line in f.readlines()]
    print(f"=> total number of images in val set: {len(image_dirs)}")
    image_dirs = sorted(image_dirs, key=lambda x: x.split('/')[-1])  # sort by image name
    return image_dirs

# define loss function
class hybridMSSIML1Loss(nn.Module):
    def __init__(self, alpha=0.84):
        super(hybridMSSIML1Loss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        ms_ssim_loss = 1 - ms_ssim(pred, target, data_range=1.0)
        l1_loss = self.l1_loss(pred, target)
        hybrid_loss = self.alpha * ms_ssim_loss + (1 - self.alpha) * l1_loss
        return hybrid_loss

def loss_calculator(out_images, target_images):
    criterion = hybridMSSIML1Loss()
    losses = []
    for out_img, target_img in zip(out_images, target_images):
        out_img, target_img = out_img.unsqueeze(0), target_img.unsqueeze(0)
        loss = criterion(out_img, target_img)
        losses.append(loss.item())
    
    return losses

# function to load images
def imageLoader(image_path, image_files):
    images = []
    for image_file in image_files:
        img = Image.open(os.path.join(image_path, image_file)).convert('RGB')
        images.append(img)
    return images, vectors

# main function
def main(args):
    # set up the environment
    print('=> torch version : {}'.format(torch.__version__))
    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)
    model.eval()

    val_image_dirs = loadImageDirs(args.val_list)

    losses = []

    for image_dir in tqdm(val_image_dirs):
        input_images = imageLoader(image_dir, os.listdir(image_dir))
        input_images = torch.stack([transforms.ToTensor()(img) for img in input_images]).cuda()
        out_images = model(input_images)
        
        losses.extend(loss_calculator(out_images, input_images))

    print('=> average loss: {}'.format(sum(losses) / len(losses)))

if __name__ == '__main__':

    args = get_args()
    main(args)