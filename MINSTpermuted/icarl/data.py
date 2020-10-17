from torchvision import datasets, transforms
import torch.utils.data
import torch
import random
from collections import defaultdict

def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    
    image = image.view(-1, c)
    image = image[permutation, :]    
    image = image.view(c, h, w)
    
    return image


def get_dataset(name, train=True, download=True, permutation=None):
    dataset_class = AVAILABLE_DATASETS[name]
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),        
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])


    return dataset_class(
        '../protoCL/data/{name}'.format(name=name), train=train,
        download=download, transform=dataset_transform,
    )
    


class BalancedDatasetSamplerTrain(torch.utils.data.sampler.Sampler):
    """Samples equal number of objects from each class from a given list of indices
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, num_samples=10, num_classes=10, num_episodes = 600):
                
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
        return dataset.train_labels[idx].item()       
                
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

    def __init__(self, dataset, num_samples=10, num_classes=10, num_episodes = 600):
                
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




AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST
}
