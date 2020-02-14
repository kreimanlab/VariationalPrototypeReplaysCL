#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import torch

from data import get_dataset
from train import train
from model import Protonet

parser = ArgumentParser('Proto Continual Learning Implementation')

#there are 10 images per class per task
parser.add_argument('--n-repeats', type=int, default=1) #repeat the program n-repeats time and store it in the name of n-repeats

parser.add_argument('--task-number', type=int, default=50)
parser.add_argument('--epochs-per-task', type=int, default=2)
parser.add_argument('--epochs-per-tasknext', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-02)
parser.add_argument('--weight-decay', type=float, default=1e-03)
parser.add_argument('--batch-size', type=int, default=20) #nuber of classes * number of samples per class
parser.add_argument('--test-size', type=int, default=100)
parser.add_argument('--proto-size', type=int, default=5) #number of images to compute prototypes
parser.add_argument('--oriproto-size', type=int, default=1) #number of sample images per class per task; minimum = number of tasks
parser.add_argument('--oriproto-eachsize', type=int, default=1) #ori proto size per task (changes across task); 50 is in full; and 20 in fewshot
parser.add_argument('--replay-freq', type=int, default=2) #prevent overfitting

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

# model args
parser.add_argument('--model-name', type=str, default='VP')
parser.add_argument('--model-dir', type=str, default='models/')
parser.add_argument('--result-dir', type=str, default='results/')
parser.add_argument('--x-dim', type=int, default=28*28) #input channel
parser.add_argument('--hid-dim', type=int, default=1000) #layer output channel
parser.add_argument('--z-dim', type=int, default=64) #encoder output channel
parser.add_argument('--temperature', type=int, default=2) #encoder output channel

# dataset
parser.add_argument('--dataset-classes', type=int, default=10) #input classes #fixed
parser.add_argument('--dataset-channels', type=int, default=1) #input classes #fixed
parser.add_argument('--dataset-samples', type=int, default=2) #input samples per class #fixed 
parser.add_argument('--dataset-episodes', type=int, default=3000) #input episodes = len(dataset)/(n_class*n_samples) = 60000/(10*10)
parser.add_argument('--dataset-nextepisodes', type=int, default=5) #input episodes = len(dataset)/(n_class*n_samples) = 60000/(10*10)
parser.add_argument('--dataset-width', type=int, default=28) #input width/height size #fixed
parser.add_argument('--dataset-nsupport', type=int, default=5) #input support size #fixed
parser.add_argument('--dataset-nquery', type=int, default=5) #input query size #fixed

if __name__ == '__main__':
    args = parser.parse_args()
    print('===========================================================')
    print(args.n_repeats)
    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda

    # generate permutations for the tasks.
    np.random.seed(args.random_seed)
    permutations = [
        np.random.permutation(args.dataset_width**2) for
        _ in range(args.task_number)
    ]

    # prepare mnist datasets.
    train_datasets = [
        get_dataset('mnist', permutation=p) for p in permutations
    ]
    test_datasets = [
        get_dataset('mnist', train=False, permutation=p) for p in permutations
    ]

    model = Protonet(args)
    
    # prepare the cuda if needed.
    if cuda:
        model.cuda()
        
    train(model, train_datasets, test_datasets, args, cuda)
        
    
    
    
