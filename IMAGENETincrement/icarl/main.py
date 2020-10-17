#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import torch

from data import get_dataset
from model import load_protonet_conv
from train import train

#network paprameters: 32*32*3*1024 + 1024*1024 + 1024*1024 + 1024*64
#ensemble: 3*32*32*10
#protos: (32*32*3*1024 + 1024*1024 + 1024*1024 + 1024*64)/(3*32*32*10)

parser = ArgumentParser('Proto Continual Learning Implementation')
parser.add_argument('--n-repeats', type=int, default=1) #repeat the program n-repeats time and store it in the name of n-repeats

parser.add_argument('--task-number', type=int, default=10)
parser.add_argument('--epochs-per-task', type=int, default=2)
parser.add_argument('--epochs-per-nexttask', type=int, default=15)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--weight-decay', type=float, default=1e-04)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--test-size', type=int, default=100)
parser.add_argument('--proto-size', type=int, default=100) #only enabled in fcn_storeProto
parser.add_argument('--proto-eachsize', type=int, default=100) #only enabled in fcn_storeProto
parser.add_argument('--oriproto-size', type=int, default=100) #only enabled in fcn_storeOriProto #172 is in full; 1633220/(10*32*32*3)
parser.add_argument('--oriproto-eachsize', type=int, default=100) #ori proto size per task (changes across task)
parser.add_argument('--replay-freq', type=int, default=2) #prevent overfitting

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

# model args
parser.add_argument('--model-mode', type=int, default=1) #1:ICARL-MSE; 2:ICARL-BCE; 3:ICARL-KLD;
parser.add_argument('--model-name', type=str, default='ICARL-MSE') #1:ICARL-MSE; 2:ICARL-BCE; 3:ICARL-KLD;
parser.add_argument('--model-dir', type=str, default='models_icml/')
parser.add_argument('--result-dir', type=str, default='results_icml/')
parser.add_argument('--x-dim', type=int, default=3*32*32) #input channel
parser.add_argument('--hid-dim', type=int, default=1024) #layer output channel
parser.add_argument('--z-dim', type=int, default=500) #encoder output channel
parser.add_argument('--temperature', type=int, default=2) #encoder output channel

# dataset
parser.add_argument('--dataset-classes', type=int, default=0) #input classes #dynamic
parser.add_argument('--dataset-current_classes', type=int, default=0) #input classes for current task #dynamic
parser.add_argument('--dataset-channels', type=int, default=3) #input classes #fixed
parser.add_argument('--dataset-samples', type=int, default=1) #input samples per class #fixed 
parser.add_argument('--dataset-episodes', type=int, default=4800) #5000 images per class; thus 5000/dataset_samples
parser.add_argument('--dataset-nextepisodes', type=int, default=10) #5000 images per class; thus 5000/dataset_samples
parser.add_argument('--dataset-width', type=int, default=32) #input width/height size #fixed
parser.add_argument('--dataset-nsupport', type=int, default=0) #input support size #fixed
parser.add_argument('--dataset-nquery', type=int, default=1) #input query size #fixed #n_support + n_query must be equal to n_samples

# processing dataset
parser.add_argument('--dataroot', type=str, default='../protoCL/data', help="The root folder of dataset or downloaded data")
parser.add_argument('--dataset', type=str, default='MINIIMAGENET', help="MNIST(default)|CIFAR10|CIFAR100|MINIIMAGENET")
parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0; split dataset when =0 ")
parser.add_argument('--first_split_size', type=int, default=10)
parser.add_argument('--other_split_size', type=int, default=10) 
parser.add_argument('--no_class_remap', dest='no_class_remap', default=True, action='store_true',
                    help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                    help="Allow data augmentation during training")
parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                    help="Fixed; cannot be chagned; Randomize the classes in splits")
parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                    help="Fixed; cannot be chagned; Randomize the order of splits")

## main function
if __name__ == '__main__':
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda

    # prepare datasets.
    train_datasets, test_datasets, task_output_space = get_dataset(args)

    model = load_protonet_conv(args)
    
    # prepare the cuda if needed.
    if cuda:
        model.cuda()
        
    train(model, train_datasets, test_datasets, task_output_space, args, cuda)
        
    
