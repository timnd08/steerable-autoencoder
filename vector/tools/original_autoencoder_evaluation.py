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
import torch.nn.functional as F 

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
    def __init__(self, alpha=0.80):
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

def correlation_calculator(out_images, target_images):
    cosine_correlation = []
    pearson_correlation = []
    for out_img, target_img in zip(out_images, target_images):
        out_img, target_img = out_img.unsqueeze(0), target_img.unsqueeze(0)
        out_img_flat = out_img.view(-1)
        target_img_flat = target_img.view(-1)
        correlation = F.cosine_similarity(out_img_flat, target_img_flat, dim=0)
        cosine_correlation.append(correlation.item())

        if not (torch.isfinite(out_img_flat).all() and torch.isfinite(target_img_flat).all()):
            pearson_correlation.append(float("nan"))
            continue

        out_centered = out_img_flat - out_img_flat.mean()
        target_centered = target_img_flat - target_img_flat.mean()

        denominator = torch.linalg.vector_norm(out_centered) * torch.linalg.vector_norm(target_centered)

        if denominator == 0:
            pearson_correlation.append(float("nan"))
            continue

        correlation = torch.dot(out_centered, target_centered) / denominator
        pearson_correlation.append(correlation.item())
    return cosine_correlation, pearson_correlation

def finite_average(values):
    values = torch.tensor(values, dtype=torch.float64)
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return float("nan"), values.numel()
    return finite_values.mean().item(), values.numel() - finite_values.numel()

# function to load images
def imageLoader(image_path, image_files):
    images = []
    for image_file in image_files:
        img = Image.open(os.path.join(image_path, image_file)).convert('RGB')
        images.append(img)
    return images

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
    cosine_correlations = []
    pearson_correlations = []
    for image_dir in tqdm(val_image_dirs):
        input_images = imageLoader(image_dir, os.listdir(image_dir))
        input_images = torch.stack([transforms.ToTensor()(img) for img in input_images]).cuda()
        out_images = model(input_images)
        
        losses.extend(loss_calculator(out_images, input_images))
        cosine_correlation, pearson_correlation = correlation_calculator(out_images, input_images)
        cosine_correlations.extend(cosine_correlation)
        pearson_correlations.extend(pearson_correlation)

        # # visualize the input and output images
        # for i in range(len(input_images)):
        #     input_img = transforms.ToPILImage()(input_images[i].cpu())
        #     output_img = transforms.ToPILImage()(out_images[i].cpu())
            
        #     # create a new image to concatenate the two images
        #     concatenated_img = Image.new('RGB', (input_img.width * 2, input_img.height))
        #     concatenated_img.paste(input_img, (0, 0))
        #     concatenated_img.paste(output_img, (input_img.width, 0))

        #     # save the concatenated image
        #     os.makedirs("figs/test", exist_ok=True)
        #     concatenated_img.save(os.path.join("figs/test", f"comparison_{val_image_dirs.index(image_dir)}_{i}.png"))

    print('=> average loss: {}'.format(sum(losses) / len(losses)))
    average_cosine, invalid_cosine = finite_average(cosine_correlations)
    average_pearson, invalid_pearson = finite_average(pearson_correlations)
    print('=> average cosine similarity: {}'.format(average_cosine))
    print('=> average Pearson correlation: {}'.format(average_pearson))
    print('=> invalid cosine samples: {}'.format(invalid_cosine))
    print('=> undefined/non-finite Pearson samples: {}'.format(invalid_pearson))
    
if __name__ == '__main__':

    args = get_args()
    main(args)