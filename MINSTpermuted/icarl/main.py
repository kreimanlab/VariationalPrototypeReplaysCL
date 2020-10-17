#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import torch

from data import get_dataset
from model import load_protonet_conv
from train import train

parser = ArgumentParser('Proto Continual Learning Implementation')

parser.add_argument('--n-repeats', type=int, default=1) #repeat the program n-repeats time and store it in the name of n-repeats
 
parser.add_argument('--task-number', type=int, default=50)
parser.add_argument('--epochs-per-task', type=int, default=2)
parser.add_argument('--epochs-per-tasknext', type=int, default=20)
parser.add_argument('--lr', type=float, default=2e-03)
parser.add_argument('--weight-decay', type=float, default=1e-03)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--test-size', type=int, default=100)
parser.add_argument('--proto-size', type=int, default=10) #only enabled in fcn_storeProto
parser.add_argument('--proto-eachsize', type=int, default=10) #only enabled in fcn_storeProto
parser.add_argument('--oriproto-size', type=int, default=10) #only enabled in fcn_storeOriProto #500 is in full; #oriproto:50 in full and 10 in fewshot
parser.add_argument('--oriproto-eachsize', type=int, default=10) #ori proto size per task (changes across task)
parser.add_argument('--replay-freq', type=int, default=2) #prevent overfitting

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

# model args
parser.add_argument('--model-mode', type=int, default=2) #1:ICARL-MSE; 2:ICARL-BCE; 3:ICARL-KLD;
parser.add_argument('--model-name', type=str, default='ICARL-BCE') #1:ICARL-MSE; 2:ICARL-BCE; 3:ICARL-KLD;
parser.add_argument('--model-dir', type=str, default='models_icml/')
parser.add_argument('--result-dir', type=str, default='results_icml/')
parser.add_argument('--x-dim', type=int, default=28*28) #input channel
parser.add_argument('--hid-dim', type=int, default=1000) #layer output channel
parser.add_argument('--z-dim', type=int, default=10) #encoder output channel
parser.add_argument('--temperature', type=int, default=1) #encoder output channel

# dataset
parser.add_argument('--dataset-classes', type=int, default=10) #input classes #fixed
parser.add_argument('--dataset-channels', type=int, default=1) #input classes #fixed
parser.add_argument('--dataset-samples', type=int, default=1) #input samples per class #fixed 
parser.add_argument('--dataset-episodes', type=int, default=6000) #input episodes = len(dataset)/(n_class*n_samples) = 60000/(10*10)
parser.add_argument('--dataset-nextepisodes', type=int, default=10) #input episodes = len(dataset)/(n_class*n_samples) = 60000/(10*10)
parser.add_argument('--dataset-width', type=int, default=28) #input width/height size #fixed
parser.add_argument('--dataset-nsupport', type=int, default=1) #input support size #fixed
parser.add_argument('--dataset-nquery', type=int, default=0) #input query size #fixed

if __name__ == '__main__':
    args = parser.parse_args()

    print('===========================================================')
    print(args.n_repeats)
    print(args.model_name)
    print(args.model_mode)
    
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

    model = load_protonet_conv(args)
    
    # prepare the cuda if needed.
    if cuda:
        model.cuda()
        
    train(model, train_datasets, test_datasets, args, cuda)
        
    
    
    
