from torch.utils import data
from torchvision.datasets.folder import default_loader
import os
import csv
from tqdm import tqdm
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

pathminiImageNet = '/home/mengmi/Projects/Proj_CL/mini_imagenet/mini_imagenet/'
pathImages = os.path.join(pathminiImageNet,'images/')

filesCSVSachinRavi = [os.path.join(pathminiImageNet,'train.csv'),
                      os.path.join(pathminiImageNet,'val.csv'),
                      os.path.join(pathminiImageNet,'test.csv')]
par_dict = {'train':[0], 'val':[1], 'test':[2], 'trainval':[0,1], 'trainvaltest':[0,1,2]}

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

#change filesCSVSachinRavi to be train, val, test set
def find_classes(partition):
    classes = []
    filesCSVSachinRavi_par = [filesCSVSachinRavi[i] for i in par_dict[partition]]
    for filename in filesCSVSachinRavi_par:
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader, None)
            print('Reading IDs....')
            for row in tqdm(csv_reader):
                if row[1] not in classes:
                    classes.append(row[1])
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def make_dataset(class_to_idx, partition, pathImages=pathImages):
    images = []
    full_set_per_cls = {}

    filesCSVSachinRavi_par = [filesCSVSachinRavi[i] for i in par_dict[partition]]
    for filename in filesCSVSachinRavi_par:
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            next(csv_reader, None)

            print('Reading IDs....')
            for row in tqdm(csv_reader):
                img_file_path  = os.path.join(pathImages, row[0])
                item = (img_file_path, class_to_idx[row[1]])
                assert is_image_file(img_file_path)
                try:
                    full_set_per_cls[row[1]].append(item)
                except:
                    #print('first example for the class')
                    full_set_per_cls[row[1]] = []
                    full_set_per_cls[row[1]].append(item)

                images.append(item)

    return images, full_set_per_cls




def load_data_into_memory(images):
    im_data = []
    im_label = []
    print('loading data into memory...')
    for idx, item in enumerate(tqdm(images)):
        im_data.append(cv2.imread(item[0]))
        im_label.append(item[1])
    return np.stack(im_data), np.stack(im_label)

class Mini_imagenet(data.Dataset):
    """ data loader for mini_imagenet:


    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    @property
    def train_labels(self):
        return self.targets

    @property
    def test_labels(self):
        return self.targets

    def __init__(self, root, split, partition='trainvaltest', transform=None, target_transform=None,
                 loader=default_loader, loading_sampled_set=True):
        classes, class_to_idx = find_classes(partition)
        self.classes = classes
        self.imgs, self.full_set_per_cls = make_dataset(class_to_idx, partition)

        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        if loading_sampled_set:
            self.imgs = self.load_sampled_set(split)
        self.data, self.targets = load_data_into_memory(self.imgs)


        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # should be called only once, thus train and val set are fixed for experiments
    def sample_train_val(self, sample_ratio=0.2):
        sampled_miniimagenet_train = {}
        sampled_miniimagenet_val = {}

        for idx, item in enumerate(self.full_set_per_cls.keys()):
            # per_cls_sampled = random.sample(full_set_per_cls[item], len(full_set_per_cls[item]))
            sampled_miniimagenet_val[item] = self.full_set_per_cls[item][:int(len(self.full_set_per_cls[item]) * sample_ratio)]
            sampled_miniimagenet_train[item] = self.full_set_per_cls[item][int(len(self.full_set_per_cls[item]) * sample_ratio):]

        f = open('/home/mengmi/Projects/Proj_CL/mini_imagenet/mini_imagenet/sampled_miniimagenet_val.pt', "wb")
        pickle.dump(sampled_miniimagenet_val, f, protocol=2)
        f.close()

        f = open('/home/mengmi/Projects/Proj_CL/mini_imagenet/mini_imagenet/sampled_miniimagenet_train.pt', "wb")
        pickle.dump(sampled_miniimagenet_train, f, protocol=2)
        f.close()

    def load_sampled_set(self, split):

        if split == 'train':
            f = open('/home/mengmi/Projects/Proj_CL/mini_imagenet/mini_imagenet/sampled_miniimagenet_train.pt', "rb")
            sampled_train_per_cls = pickle.load(f)
            sampled_train_per_cls_concat = []
            for cls in self.classes:
                sampled_train_per_cls_concat += sampled_train_per_cls[cls]
            return sampled_train_per_cls_concat

        elif split == 'val':
            f = open('/home/mengmi/Projects/Proj_CL/mini_imagenet/mini_imagenet/sampled_miniimagenet_val.pt', "rb")
            sampled_val_per_cls = pickle.load(f)
            sampled_val_per_cls_concat = []
            for cls in self.classes:
                sampled_val_per_cls_concat += sampled_val_per_cls[cls]
            return sampled_val_per_cls_concat

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

#this must be run once in local computers
#if __name__ == '__main__':
#    #make_dataset()
    
#    ## RUN THESE for the very FIRST time
#    mini_imagenet = Mini_imagenet(pathminiImageNet,'trainval',loading_sampled_set=False)
#    mini_imagenet.sample_train_val()
#    #df =mini_imagenet.__getitem__(1000)
#    df =mini_imagenet.__len__()
#    print(df)
    
#    ## RUN THESE for plotting images and have a sense
#    mini_imagenet = Mini_imagenet(pathminiImageNet,'val',loading_sampled_set=True)    
#    img, target =mini_imagenet.__getitem__(1)
#    plt.show(img)
#    #print(img)
#    print(target)
    
    
