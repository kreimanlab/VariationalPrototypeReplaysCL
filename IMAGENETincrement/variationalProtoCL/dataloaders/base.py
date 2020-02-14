import torchvision
from torchvision import transforms
from .wrapper import CacheClassLabel
from load_mini_imagenet_pytorch import Mini_imagenet
import cv2

def MINIIMAGENET(data_root=None, train_aug=False, img_size=84):
    # normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.Resize(32),#initial image size is 84
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = Mini_imagenet('/home/mengmi/Projects/Proj_CL/mini_imagenet/mini_imagenet/', 'train',loading_sampled_set=True,
                                  transform=train_transform)
    train_dataset = CacheClassLabel(train_dataset, 100, train_dataset.train_labels)

    val_dataset = Mini_imagenet('/home/mengmi/Projects/Proj_CL/mini_imagenet/mini_imagenet/', 'val',loading_sampled_set=True,
                                  transform=train_transform)
    val_dataset = CacheClassLabel(val_dataset, 100, val_dataset.test_labels)

    return train_dataset, val_dataset


def MNIST(dataroot, train_aug=False):
    # Add padding to make 32x32
    #normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset, 10, train_dataset.train_labels)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset, 10, val_dataset.test_labels)

    return train_dataset, val_dataset

def CIFAR10(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
        )
    for i in range(100):
        cv2.imshow('image', cv2.cvtColor(train_dataset.data[i], cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    train_dataset = CacheClassLabel(train_dataset, 10, train_dataset.train_labels)


    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset, 10, val_dataset.test_labels)

    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset, 100, train_dataset.train_labels)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset, 100, val_dataset.test_labels)

    return train_dataset, val_dataset

