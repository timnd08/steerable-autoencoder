#!/usr/bin/env python

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import utils
import models.builer as builder
import dataloader
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, 
                        help='backbone architechture')
    parser.add_argument('--train_list', type=str)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', 
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--pth-save-fold', default='results/tmp', type=str,
                        help='The folder to save pths')
    parser.add_argument('--pth-save-epoch', default=1, type=int,
                        help='The epoch to save pth')
    parser.add_argument('--parallel', type=int, default=1, 
                        help='1 for parallel, 0 for non-parallel')
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--pretrained', default = None, type=str, help='path to pretrained pth')
    parser.add_argument('--resume', default = None, type=str, help='path to resume pth')                                            

    args = parser.parse_args()

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    ngpus_per_node = torch.cuda.device_count()
    print('=> ngpus : {}'.format(ngpus_per_node))

    if args.parallel == 1: 
        # single machine multi card       
        args.gpus = ngpus_per_node
        args.nodes = 1
        args.nr = 0
        args.world_size = args.gpus * args.nodes

        args.workers = int(args.workers / args.world_size)
        args.batch_size = int(args.batch_size / args.world_size)
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        args.world_size = 1
        main_worker(ngpus_per_node, args)

class hybridMSSIML1Loss(nn.Module):
    def __init__(self, alpha=0.80):
        super(hybridMSSIML1Loss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target, mask):
        # put 5x heavier weight on the masked region
        weights = 1.0 + 4.0 * mask

        l1_loss = (torch.abs(pred - target) * weights).sum() / (weights.sum() * pred.shape[1])
        ms_ssim_loss = 1.0 - ms_ssim(pred, target, data_range=1.0)

        #l1_loss = self.l1_loss(pred, target)
        hybrid_loss = self.alpha * ms_ssim_loss + (1.0 - self.alpha) * l1_loss
        return hybrid_loss

    
def main_worker(gpu, args):
    utils.init_seeds(1 + gpu, cuda_deterministic=False)
    if args.parallel == 1:
        args.gpu = gpu
        args.rank = args.nr * args.gpus + args.gpu

        torch.cuda.set_device(gpu)
        torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)  
           
    else:
        # two dummy variable, not real
        args.rank = 0
        args.gpus = 1 
    if args.rank == 0:
        print('=> modeling the network {} ...'.format(args.arch))
    
    model = builder.BuildAutoEncoder(args)
    if args.resume:
        #find latest checkpoint
        lastest_checkpoint = sorted([f for f in os.listdir(args.resume) if f.endswith('.pth')])[-1]
        if lastest_checkpoint is None:
            print('=> no checkpoint found in {} ...'.format(args.resume))
            return
        print('=> loading pth from {} ...'.format(os.path.join(args.resume, lastest_checkpoint)))
        utils.load_dict(os.path.join(args.resume, lastest_checkpoint), model)
        epoch = int(lastest_checkpoint.split('.')[0])
        args.start_epoch = epoch +1
        print(' : successfully loaded pth from epoch {}'.format(epoch))
        if args.pretrained:
            for name, param in model.named_parameters():
                if name.startswith('module.encoder.'):
                    param.requires_grad = False
 

    if args.pretrained and not args.resume:
        if os.path.isfile(args.pretrained):
            if args.rank == 0:
                print("=> loading pretrained model '{}' ...".format(args.pretrained))
            
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            pretrained_state_dict = checkpoint['state_dict']
            model_state_dict = model.state_dict()

            encoder_state_dict = {}
            for k, v in pretrained_state_dict.items():
                if k.startswith('module.encoder.'):
                    new_key=k
                    if new_key in model_state_dict and model_state_dict[new_key].shape == v.shape:
                        encoder_state_dict[new_key] = v
            model.load_state_dict(encoder_state_dict, strict=False)

            if args.rank == 0:
                print("=> successfully loaded pretrained model '{}'".format(args.pretrained))
                print("=> freezing the encoder weights ...")
            for name, param in model.named_parameters():
                if name.startswith('module.encoder.'):
                    param.requires_grad = False
        else:
            if args.rank == 0:
                print("=> no pretrained model found at '{}'".format(args.pretrained))

    if args.rank == 0:       
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
        print('=> num of trainable params: {} ({}M)'.format(trainable_params, int(trainable_params * 4 / (1024*1024))))
    
    if args.rank == 0:
        print('=> building the optimizer ...')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,)
    # optimizer = torch.optim.SGD(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         args.lr,
    #         momentum=args.momentum,
    #         weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), checkpoint['optimizer']['param_groups'][0]['lr'],)

    if args.rank == 0:
        print('=> building the dataloader ...')
    train_loader = dataloader.train_loader(args)

    if args.rank == 0:
        print('=> building the criterion ...')
    #criterion = nn.MSELoss()
    criterion = hybridMSSIML1Loss(alpha=0.80)

    global iters
    iters = 0

    model.train()
    if args.rank == 0:
        print('=> starting training engine ...')
    for epoch in range(args.start_epoch, args.epochs):
        
        global current_lr
        current_lr = utils.adjust_learning_rate_cosine(optimizer, epoch, args)

        train_loader.sampler.set_epoch(epoch)
        
        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args)

        # save pth
        if epoch % args.pth_save_epoch == 0 and args.rank == 0:
            state_dict = model.state_dict()

            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'optimizer' : optimizer.state_dict(),
                },
                os.path.join(args.pth_save_fold, '{}.pth'.format(str(epoch).zfill(3)))
            )
            
            print(' : save pth for epoch {}'.format(epoch))


def do_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.2f')
    data_time = utils.AverageMeter('Data', ':2.2f')
    losses = utils.AverageMeter('Loss', ':.4f')
    learning_rate = utils.AverageMeter('LR', ':.8f')
    
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    # update lr
    learning_rate.update(current_lr)

    for i, (input_batch, target_batch, mask_batch, vectors_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1

        batch_size = input_batch.size(0)
        num_targets = target_batch.size(1)
        
        input_batch = input_batch.view(-1, 3, 224, 224)
        target_batch = target_batch.view(-1, 3, 224, 224)
        mask_batch = mask_batch.view(-1, 1, 224, 224)
        vectors_batch = vectors_batch.view(-1, 2)
         
        input_batch = input_batch.cuda(non_blocking=True)
        target_batch = target_batch.cuda(non_blocking=True)
        mask_batch = mask_batch.cuda(non_blocking=True)
        vectors_batch = vectors_batch.cuda(non_blocking=True)

        output_batch = model(input_batch, vectors_batch)

        loss = criterion(output_batch, target_batch, mask_batch)

        # compute gradient and do solver step
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weights
        optimizer.step()

        # syn for logging
        torch.cuda.synchronize()

        # record loss
        losses.update(loss.item(), batch_size*num_targets)          

        # measure elapsed time
        # if args.rank == 0:
        #     batch_time.update(time.time() - end)        
        #     end = time.time()   

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)

if __name__ == '__main__':

    args = get_args()

    main(args)

