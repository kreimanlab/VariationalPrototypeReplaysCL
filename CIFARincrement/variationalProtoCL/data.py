import torch.utils.data
import torch
import random
from collections import defaultdict
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen

def get_dataset(args):
    # Prepare dataloaders
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.train_aug)
    if args.n_permutation>0:
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                             args.n_permutation,
                                                                             remap_class=not args.no_class_remap)
    else:
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                          first_split_sz=args.first_split_size,
                                                                          other_split_sz=args.other_split_size,
                                                                          rand_split=args.rand_split,
                                                                          remap_class=not args.no_class_remap)

    return train_dataset_splits, val_dataset_splits, task_output_space
    


class BalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples equal number of objects from each class from a given list of indices
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, num_samples=10, num_episodes = 600):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) 
        
        # label list and sort based on class number
        label_list = defaultdict(list)
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_list[label].append(idx)
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = num_samples
        self.num_classes = len(label_list.keys())
        self.num_episodes = num_episodes     
    
        self.iteratorlist = []
        for epi in range(self.num_episodes):
            for cla in label_list.keys():
                for sam in range(self.num_samples):                                
                    randind = random.choice(label_list[cla])
                    self.iteratorlist.append(randind)

    def _get_label(self, dataset, idx):
        img, label, name = dataset.__getitem__(idx)
        return label    
                
    def __iter__(self):
        return iter(self.iteratorlist)

    def __len__(self):
        return self.num_episodes

class BalancedDatasetSamplerTest(torch.utils.data.sampler.Sampler):
    """Samples equal number of objects from each class from a given list of indices
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, num_samples=1, num_classes=10, num_episodes = 6000):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) 
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_episodes = num_episodes
            
        # label list and sort based on class number
        label_list = defaultdict(list)
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_list[label].append(idx)
    
        self.iteratorlist = []
        for epi in range(self.num_episodes):
            for cla in range(self.num_classes):
                for sam in range(self.num_samples):                                
                    randind = random.choice(label_list[cla])
                    self.iteratorlist.append(randind)

    def _get_label(self, dataset, idx):
        return dataset.test_labels[idx].item()       
                
    def __iter__(self):
        return iter(self.iteratorlist)

    def __len__(self):
        return self.num_episodes