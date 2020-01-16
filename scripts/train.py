#!/usr/bin/env python3

import os
import torch
import argparse
import random
import sys
from diarization_dataset import DiarDataset
import numpy as np
import shutil
import socket
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg, cfg_from_file
from utils import train
import pickle

np.set_printoptions(suppress=True)
print(socket.gethostname())

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(
    description='Region Proposal Network based Speaker Diarization Training')
parser.add_argument('exp_dir', type=str,
                    help='path of experiment')
parser.add_argument('train_dir', type=str,
                    help='directory for training')
parser.add_argument('dev_dir', default=None, type=str,
                    help='directory for validation')
parser.add_argument('--cfg_file', default="", type=str,
                    help='configure file')

# data process parameters
parser.add_argument('--rate', default=8000, type=int,
                    help='sample rate')
parser.add_argument('--frame_size', default=512, type=int,
                    help='frame size')
parser.add_argument('--frame_shift', default=80, type=int,
                    help='frame shift')
parser.add_argument('--merge_dis', default=0.0, type=float,
                    help='merge two segments if their distance is smaller than merge_dis')
parser.add_argument('--min_dis', default=0.2, type=float,
                    help='minimum length of each segment, discard segments that are too short')
parser.add_argument('--padded_len', default=15, type=int,
                    help='label length after padding')

# training parameters
parser.add_argument('--resume', default=None, type=str,
                    help='path to latest checkpoint')
parser.add_argument('--initialize', default=0, type=int,
                    help='whether to use checkpoint to initialize model parameters')
parser.add_argument('--freeze', default=0, type=int,
                    help='whether to freeze the model parameters')
parser.add_argument('--set_bn_fix', default=0, type=int,
                    help='whether to set batchnorm fixed')
parser.add_argument('--pretrain_model', default=None, type=str,
                    help='the directory of pretrained model')
parser.add_argument('--pretrain_resnet_model', default=None, type=str,
                    help='the directory of pretrained resnet model')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=8, type=int,
                    help='mini-batch size')
parser.add_argument('--num_workers', default=0, type=int,
                    help='number of workers for data loading')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='optimizer')
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--scheduler', default='reduce', type=str,
                    help='learning rate scheduler')
parser.add_argument('--min_lr', default=1e-5, type=float, 
                    help='minimum learning rate')
parser.add_argument('--patience', default=10, type=int, 
                    help='patience to reduce learning rate')
parser.add_argument('--clip', default=5.0, type=float, 
                    help='gradient clip')
parser.add_argument('--seed', default=7, type=int,
                    help='random seed')
parser.add_argument('--ignore_cls_loss', default=0, type=int,
                    help='ignore classification loss for validation, because the dev labels are \
                            not in the training set')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='it seems that the RCNN_loss_cls_spk is dominating \
                         the loss function. So I want to give it a smaller weight')

# network parameters
parser.add_argument('--arch', default='res101', type=str, 
                    help='model architecture')
parser.add_argument('--large_scale', default=0, type=int, 
                    help='whether use large image scale')
parser.add_argument('--nclass', default=5963, type=int, 
                    help='number of classes (5962 speakers and background)')

# validate parameters
parser.add_argument('--eval_interval', default=20, type=int,
                    help='number of epochs to save the model')
parser.add_argument('--num_dev', default=-1, type=int,
                    help='the dev set is too large, just use some of it')

# visualize
parser.add_argument('--use_tfb', dest='use_tfboard',
                    help='whether use tensorboard',
                    action='store_true')

def main():
    global args
    args = parser.parse_args()
    print(args)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare training set and dev set
    train_dataset = DiarDataset(args.train_dir, args.rate, args.frame_size, args.frame_shift, None, args.padded_len, args.merge_dis, args.min_dis)
    dev_dataset = DiarDataset(args.dev_dir, args.rate, args.frame_size, args.frame_shift, None, args.padded_len, args.merge_dis, args.min_dis, args.num_dev)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)
    print("{} TRAIN segments, {} DEV segments".format(len(train_dataset), len(dev_dataset)))

    if args.cfg_file == "":
        args.cfg_file = "cfgs/{}_ls.yml".format(args.arch) if args.large_scale else "cfgs/{}.yml".format(args.arch)
    print("Using configure file {}".format(args.cfg_file))

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    # save cfg file
    with open('{}/cfg.pkl'.format(args.exp_dir), 'wb') as handle:
        pickle.dump(cfg, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # initilize the network here.
    start_epoch = 1
    if args.arch == 'res101':
        model = resnet(args.nclass, 101, pretrained=args.pretrain_resnet_model, freeze=args.freeze, set_bn_fix=args.set_bn_fix)
    else:
        print("Network is not supported")
    model.create_architecture()
    model = model.to(device)

    if args.pretrain_model is not None:
        print("Loading pretrained weights from {}".format(args.pretrain_model))
        checkpoint = torch.load(args.pretrain_model)
        pretrained_dict = checkpoint['model'] 
        model_dict = model.state_dict()
        pretrained_dict_new = {}
        para_list = []
        for k, v in pretrained_dict.items():
            assert k in model_dict
            if model_dict[k].size() == pretrained_dict[k].size():
                pretrained_dict_new[k] = v
            else:
                para_list.append(k)
        print("Total {} parameters, Loaded {} parameters".format(len(pretrained_dict), len(pretrained_dict_new)))
        print("Not loading {} because of different sizes".format(", ".join(para_list)))
        model_dict.update(pretrained_dict_new) 
        model.load_state_dict(model_dict)
        print("Loaded checkpoint '{}' (epoch {} iter {})".format(args.pretrain_model, checkpoint['epoch'], checkpoint['iter']))
        print("Best score {}".format(checkpoint['best_score']))

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':args.lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':args.lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    # define optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=0.9)
    else:
        raise ValueError("Optimizer type not defined.")

    start_epoch, start_iter, best_score = 1, 1, float('inf')
    # load parameters
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            if not args.initialize:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                start_iter = checkpoint['iter'] + 1
                if start_iter > len(train_loader):
                    start_epoch += 1
                    start_iter = 1 

            print("loaded checkpoint '{}' (epoch {} iter {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['iter']))
            print("best score {:.4f}".format(checkpoint['best_score']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    # use tensorboard to monitor the loss
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("{}/log".format(args.exp_dir))
    else:
        logger = None

    args.start_epoch, args.start_iter, args.best_score = start_epoch, start_iter, best_score
    # train
    train(train_loader, dev_loader, model, device, optimizer, logger, args)

    if args.use_tfboard:
        logger.close()
    return 0
            
if __name__ == "__main__":
    main()
