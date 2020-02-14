#!/usr/bin/env python3
from argparse import ArgumentParser
import torch

from data import get_dataset
from model import Protonet
from train import train

#network paprameters: 32*32*3*1024 + 1024*1024 + 1024*1024 + 1024*64
#ensemble: 3*32*32*10
#protos: (32*32*3*1024 + 1024*1024 + 1024*1024 + 1024*64)/(3*32*32*10)

#from few-shot versus full: epochs-per-tasknext: 1/4; oriproto-eachsize: 10/53; oriproto-size: 10/53; model-name; dataset-nextepisodes: 5/500; 

parser = ArgumentParser('Proto Continual Learning Implementation')
parser.add_argument('--n-repeats', type=int, default=1) #repeat the program n-repeats time and store it in the name of n-repeats

parser.add_argument('--task-number', type=int, default=0)
parser.add_argument('--epochs-per-task', type=int, default=20)
parser.add_argument('--epochs-per-tasknext', type=int, default=50)#change to 1 for few shot; 4 in full shots
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--weight-decay', type=float, default=1e-04)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--test-size', type=int, default=100)
parser.add_argument('--val-size', type=int, default=100)
parser.add_argument('--proto-size', type=int, default=10) #only enabled in fcn_storeProto
parser.add_argument('--oriproto-size', type=int, default=100) #total number of original images stored for all classes; this number at least equal to number of classes in total
parser.add_argument('--oriproto-eachsize', type=int, default=100) #ori proto size per task (changes across task); full size = 53; fewshot = 10
parser.add_argument('--recall-eachsize', type=int, default=2)
parser.add_argument('--replay-freq', type=int, default=2) #prevent overfitting

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

# model args
parser.add_argument('--model-name', type=str, default='VP')#T2 equals temperature =0.5; and vice versa
parser.add_argument('--model-dir', type=str, default='models/')
parser.add_argument('--result-dir', type=str, default='results/')
parser.add_argument('--hid-dim', type=int, default=1024) #layer output channel
parser.add_argument('--z-dim', type=int, default=500) #encoder output channel
parser.add_argument('--temperature', type=int, default=2) #encoder output channel

# dataset
parser.add_argument('--dataset-classes', type=int, default=0) #leave to default
parser.add_argument('--dataset-current_classes', type=int, default=0) #leave to default
parser.add_argument('--dataset-channels', type=int, default=3) #input classes #fixed
parser.add_argument('--dataset-samples', type=int, default=2) #input samples per class #fixed 
parser.add_argument('--dataset-episodes', type=int, default=8000) #leave to default
parser.add_argument('--dataset-nextepisodes', type=int, default=5) #leave to default
parser.add_argument('--dataset-total', type=int, default=8000) #total num of images per class
parser.add_argument('--dataset-total2', type=int, default=10) #total num of images per class
parser.add_argument('--dataset-width', type=int, default=32) #input width/height size #fixed
parser.add_argument('--dataset-nsupport', type=int, default=5) #input support size #fixed
parser.add_argument('--dataset-nquery', type=int, default=5) #input query size #fixed #n_support + n_query must be equal to n_samples

# processing dataset
parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
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

    model = Protonet(args)
    
    # prepare the cuda if needed.
    if cuda:
        model.cuda()
        
    train(model, train_datasets, test_datasets, task_output_space, args, cuda)
        
    
