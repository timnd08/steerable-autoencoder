import argparse
import csv
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
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--resume', default='results/imageMini-vgg16_fc/000.pth', type=str)
    parser.add_argument('--val_list', default='list/imageNet_val_list.txt', type=str)           
    parser.add_argument('--print_shapes', action='store_true',
                        help='print layer output dimensions before evaluation')
    parser.add_argument('--shape_csv', default=None, type=str,
                        help='optional CSV path for layer output dimensions')
    parser.add_argument('--input_size', default=224, type=int,
                        help='dummy square input size used for the shape report')
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


def show_model_dimensions(model, input_size=224, csv_path=None):
    """Print and optionally save the output shape of every leaf layer."""
    inspected_model = model.module if isinstance(model, nn.DataParallel) else model
    layer_shapes = []
    hooks = []

    def record_shape(name, layer):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                shape = tuple(output.shape)
            elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                shape = tuple(output[0].shape)
            else:
                shape = str(type(output).__name__)
            parameters = sum(parameter.numel() for parameter in layer.parameters(recurse=False))
            layer_shapes.append((name, layer.__class__.__name__, shape, parameters))

        return hook

    for name, layer in inspected_model.named_modules():
        if name and len(list(layer.children())) == 0:
            hooks.append(layer.register_forward_hook(record_shape(name, layer)))

    device = next(inspected_model.parameters()).device
    dummy_image = torch.zeros(1, 3, input_size, input_size, device=device)
    dummy_vector = torch.zeros(1, 2, device=device)
    with torch.no_grad():
        model(dummy_image, dummy_vector)

    for hook in hooks:
        hook.remove()

    print('\n=> layer output dimensions:')
    print(f"{'layer':55} {'type':20} {'output shape':25} parameters")
    print('-' * 110)
    for name, layer_type, shape, parameters in layer_shapes:
        print(f'{name:55} {layer_type:20} {str(shape):25} {parameters}')

    if csv_path:
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['layer', 'type', 'output_shape', 'parameters'])
            writer.writerows(layer_shapes)
        print(f'=> saved layer dimensions to {csv_path}')

# function to load images, vectors, and ground truth images
def imageAndVectorLoader(image_path, image_files):
    input_images = []
    groundtruth_images = []
    vectors = []

    for image_file in image_files:
        name_parts = image_file.split('.')[0].split('_')

        input_img = Image.open(os.path.join(image_path, image_file)).convert('RGB')

        gt_img_name = "_".join(name_parts[:-2]) + '_112_112.png'
        gt_img = Image.open(os.path.join(image_path, gt_img_name)).convert('RGB')

        vectorX = (float(name_parts[-2]) - 112.0) / 112.0
        vectorY = (float(name_parts[-1]) - 112.0) / 112.0

        vector = (vectorX, vectorY)

        input_images.append(input_img)
        groundtruth_images.append(gt_img)
        vectors.append(vector)

    return input_images, groundtruth_images, vectors

def cross_correlation_calculator(out_images, target_images):
    cross_correlation = []
    for out_img, target_img in zip(out_images, target_images):
        out_img, target_img = out_img.unsqueeze(0), target_img.unsqueeze(0)
        out_img_flat = out_img.view(-1)
        target_img_flat = target_img.view(-1)
        correlation = F.cosine_similarity(out_img_flat, target_img_flat, dim=0)
        cross_correlation.append(correlation.item())
    return cross_correlation

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

    if args.print_shapes or args.shape_csv:
        show_model_dimensions(model, args.input_size, args.shape_csv)

    val_image_dirs = loadImageDirs(args.val_list) 

    losses = []
    cross_correlations = []

    for image_dir in tqdm(val_image_dirs):
        input_images, groundtruth_images, input_vectors = imageAndVectorLoader(image_dir, os.listdir(image_dir))
        input_images = torch.stack([transforms.ToTensor()(img) for img in input_images]).cuda()
        groundtruth_images = torch.stack([transforms.ToTensor()(img) for img in groundtruth_images]).cuda()
        input_vectors = torch.tensor(input_vectors, dtype=torch.float32).cuda()
        out_images = model(input_images, input_vectors)
        
        losses.extend(loss_calculator(out_images, groundtruth_images))
        cross_correlations.extend(cross_correlation_calculator(out_images, input_images))

        # visualize the input, output, and ground truth images
        # for i in range(len(input_images)):
        #     input_img = transforms.ToPILImage()(input_images[i].cpu())
        #     output_img = transforms.ToPILImage()(out_images[i].cpu())
        #     gt_img = transforms.ToPILImage()(groundtruth_images[i].cpu())

        #     # create a new image to concatenate the three images
        #     concatenated_img = Image.new('RGB', (input_img.width * 3, input_img.height))
        #     concatenated_img.paste(input_img, (0, 0))
        #     concatenated_img.paste(output_img, (input_img.width, 0))
        #     concatenated_img.paste(gt_img, (input_img.width * 2, 0))

        #     # save the concatenated image
        #     os.makedirs("figs/test", exist_ok=True)
        #     concatenated_img.save(os.path.join("figs/test", f"comparison_{val_image_dirs.index(image_dir)}_{i}.png"))

    print('=> average loss: {}'.format(sum(losses) / len(losses)))
    print('=> average cross correlation: {}'.format(sum(cross_correlations) / len(cross_correlations)))

    # model dimensions for each layers
    # print('=> model dimensions for each layer:')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.size()}")

if __name__ == '__main__':

    args = get_args()
    main(args)