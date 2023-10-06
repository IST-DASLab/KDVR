"""
Dataset loading utilities
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode

from torch.utils.data import TensorDataset, Subset


DATASETS_NAMES = ['imagenet', 'cifar10', 'cifar100', 'mnist']

__all__ = ["get_datasets", "extract_resnet50_features", "classification_num_classes",
           "interpolation_flag"]


def classification_dataset_str_from_arch(arch):
    if 'cifar100' in arch:
        dataset = 'cifar100'
    elif 'cifar' in arch:
        dataset = 'cifar10'
    elif 'mnist' in arch:
        dataset = 'mnist'
    else:
        dataset = 'imagenet'
    return dataset


def extract_resnet50_features(model, data_loader, device, update_bn_stats=False, bn_updates=10):
    # this is where the extracted features will be added:
    data_features_tensor = None
    data_labels_tensor = None

    if update_bn_stats:
        # if true, update the BatchNorm stats by doing a few dummy forward passes through the data
        # if false, use the ImageNet Batch Norm stats
        model.train()
        for i in range(bn_updates):
            with torch.no_grad():
                for sample, target in data_loader:
                    sample = sample.to(device)
                    model(sample)
    model.eval()

    # register a forward hook:
    h = None
    activation = {}

    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output.detach()

        return hook

    h = model.avgpool.register_forward_hook(get_activation('avgpool'))

    # get the features before the FC layer
    with torch.no_grad():
        for i, (sample, target) in enumerate(data_loader):
            sample = sample.to(device)
            sample_output = model(sample)
            sample_feature = activation['avgpool'].cpu()

            sample_feature = sample_feature.view(sample_feature.size(0), -1)
            if data_features_tensor is None:
                data_features_tensor = sample_feature
                data_labels_tensor = target
            else:
                data_features_tensor = torch.cat((data_features_tensor, sample_feature))
                data_labels_tensor = torch.cat((data_labels_tensor, target))
            if i % 100 == 0:
                print("extracted for {} batches".format(i))
    h.remove()
    # tensor_features_data = torch.utils.data.TensorDataset(data_features_tensor, data_labels_tensor)
    return data_features_tensor, data_labels_tensor


def classification_num_classes(dataset):
    return {'cifar10': 10,
            'cifar100': 100,
            'mnist': 10,
            'imagenet': 1000,
            }.get(dataset, None)


def classification_get_input_shape(dataset):
    if dataset == 'imagenet':
        return 1, 3, 224, 224
    elif dataset in ('cifar10', 'cifar100'):
        return 1, 3, 32, 32
    elif dataset == 'mnist':
        return 1, 1, 28, 28
    else:
        raise ValueError("dataset %s is not supported" % dataset)


def __dataset_factory(dataset):
    return globals()[f'{dataset}_get_datasets']


def interpolation_flag(interpolation):
    if interpolation == 'bilinear':
        return InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        return InterpolationMode.BICUBIC
    raise ValueError("interpolation must be one of 'bilinear', 'bicubic'")


def get_datasets(dataset, dataset_dir, **kwargs):
    datasets_fn = __dataset_factory(dataset)
    return datasets_fn(dataset_dir, **kwargs)


def mnist_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear'):
    # interpolation not used, here for consistent call.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset


def cifar10_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear',
                         use_imagenet_stats=False, standard_transforms=False):
    interpolation = interpolation_flag(interpolation)
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)

    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
    if standard_transforms:
        if use_data_aug:
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            train_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    else:
        if use_data_aug:
            train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                                  # transforms.RandomCrop(32, padding=4),transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(means, stds)])

        else:
            train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(means, stds)])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    if standard_transforms:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    else:
        test_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, stds)
                                         ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def cifar100_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear', use_imagenet_stats=False):
    interpolation = interpolation_flag(interpolation)

    means = (0.5071, 0.4867, 0.4408)
    stds = (0.2675, 0.2565, 0.2761)
    if use_imagenet_stats:
        print("Use ImageNet stats for train and test")
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)

    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                      download=True, transform=train_transform)

    test_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(means, stds)])

    test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                     download=True, transform=test_transform)

    return train_dataset, test_dataset


def imagenet_get_datasets(data_dir, use_data_aug=True, interpolation='bilinear'):
    interpolation = interpolation_flag(interpolation)
    print("getting imagenet datasets")
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if use_data_aug:
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])
    else:
        train_transform = transforms.Compose([transforms.Resize(256, interpolation=interpolation),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset





