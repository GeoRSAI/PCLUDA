from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
import csv
import torchvision
import os
from PIL import Image
import numpy as np

dataset_dict = {
    'UCMD': {
        "train_path": 'dataset/UCMD/train.csv',
        "val_path": 'dataset/UCMD/val.csv',
        "test_path": 'dataset/UCMD/test.csv',
        'database_path': 'dataset/UCMD/database.csv',
        'class_names': ['agricultural', 'baseball_diamond', 'beach', 'dense_residential', 'forest',
                        'medium_residential', 'parking_lot', 'river', 'sparse_residential', 'storage_tanks']
    },
    'NWPU': {
        "train_path": 'dataset/NWPU/train.csv',
        "val_path": 'dataset/NWPU/val.csv',
        "test_path": 'dataset/NWPU/test.csv',
        'database_path': 'dataset/NWPU/database.csv',
        'class_names': ['agricultural', 'baseball_diamond', 'beach', 'dense_residential', 'forest',
                        'medium_residential', 'parking_lot', 'river', 'sparse_residential', 'storage_tanks']
    },
    'AID': {
        "train_path": 'dataset/AID/train.csv',
        "val_path": 'dataset/AID/val.csv',
        "test_path": 'dataset/AID/test.csv',
        'database_path': 'dataset/AID/database.csv',
        'class_names': ['agricultural', 'baseball_diamond', 'beach', 'dense_residential', 'forest',
                        'medium_residential', 'parking_lot', 'river', 'sparse_residential', 'storage_tanks']
    }
}

rs_dataset_name = ['UCMD', 'NWPU', 'AID']

class RS_dataset(Dataset):

    def __init__(self, root, index_file, transform, strong_transform=None):

        with open(index_file, 'r') as fh:
            reader = csv.reader(fh)
            img_paths = []
            labels = []

            for line in reader:
                img_path = os.path.join(root, line[0])
                img_paths.append(img_path)
                labels.append(int(line[1]))

        self.n_class = max(labels) + 1
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        img = Image.open(img_path).convert('RGB')
        img_weakly = self.transform(img)

        if self.strong_transform is not None:
            img_strong = self.strong_transform(img)
            return img_weakly, img_strong, label

        return img_weakly, label

    def __len__(self):
        return len(self.img_paths)

    def get_class(self):
        return self.n_class


class RS_dataset_imgpath(Dataset):

    def __init__(self, root, index_file, transform):

        with open(index_file, 'r') as fh:
            reader = csv.reader(fh)
            img_paths = []
            labels = []
            # for line in fh:
            for line in reader:
                img_path = os.path.join(root, line[0])
                img_paths.append(img_path)
                labels.append(int(line[1]))

        self.n_class = max(labels) + 1
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path

    def __len__(self):
        return len(self.img_paths)

    def get_class(self):
        return self.n_class

def get_rs_dataset(root, dataset_name, transform, appli, strong_transform=None):

    if appli == 'train':
        data_csv = dataset_dict[dataset_name]['train_path']
    elif appli == 'val':
        data_csv = dataset_dict[dataset_name]['val_path']
    elif appli == 'test':
        data_csv = dataset_dict[dataset_name]['test_path']
    else:
        data_csv = dataset_dict[dataset_name]['database_path']

    dataset = RS_dataset(root, index_file=data_csv, transform=transform, strong_transform=strong_transform)

    n_class = dataset.get_class()

    return dataset, n_class


def get_rs_dataset_imgpath(root, dataset_name, transform, appli):

    if appli == 'train':
        data_csv = dataset_dict[dataset_name]['train_path']
    elif appli == 'val':
        data_csv = dataset_dict[dataset_name]['val_path']
    elif appli == 'test':
        data_csv = dataset_dict[dataset_name]['test_path']
    else:
        data_csv = dataset_dict[dataset_name]['database_path']

    dataset = RS_dataset_imgpath(root, index_file=data_csv, transform=transform)

    n_class = dataset.get_class()

    return dataset, n_class


def get_rs_class_name(dataset_name):
    class_names = dataset_dict[dataset_name]['class_names']
    return class_names

if __name__ == '__main__':
    dataset_name = 'UCMD'
    transform = transforms.Compose(
                [transforms.Resize([256, 256]),
                 transforms.RandomCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])

    dataset, n_class = get_rs_dataset(dataset_name, transform=transform, appli='val')

    train_dl = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    imgs, labels = next(iter(train_dl))
