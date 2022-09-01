"""
Data preprocessing tools
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""

import os
import sys
import pickle
import random

import cv2
import scipy.io as sio
from scipy import sparse
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import utils


DEFAULT_DATA_ROOT = './data'
PROCESSED_DATA_ROOT = os.path.join(DEFAULT_DATA_ROOT, 'processed')
RAW_DATA_ROOT = os.path.join(DEFAULT_DATA_ROOT, 'raw')


def export_dataset(name, views, labels):
    """
    Save dataset as .npz files
    :param name:
    :param views:
    :param labels:
    :return:
    """
    os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_ROOT, f"{name}.npz")
    npz_dict = {"labels": labels, "n_views": len(views)}
    for i, v in enumerate(views):
        npz_dict[f"view_{i}"] = v
    np.savez(file_path, **npz_dict)


def image_edge(img):
    """
    :param img:
    :return:
    """
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return np.stack((img, edge), axis=-1)


def _mnist(dataset_class):
    img_transforms = transforms.Compose([image_edge,
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
    dataset = dataset_class(root=RAW_DATA_ROOT, train=True,
                            download=True, transform=img_transforms)

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, labels = list(loader)[0]
    return data, labels


def emnist():
    data, labels = _mnist(torchvision.datasets.MNIST)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("emnist", views=views, labels=labels)


def fmnist():
    data, labels = _mnist(torchvision.datasets.FashionMNIST)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("fmnist", views=views, labels=labels)


def coil(n_objs=20):
    from skimage.io import imread
    assert n_objs in [20, 100]
    data_dir = os.path.join(RAW_DATA_ROOT, f"coil-{n_objs}")
    img_size = (1, 128, 128) if n_objs == 20 else (3, 128, 128)
    n_imgs = 72
    n_views = 3

    n = (n_objs * n_imgs) // n_views

    views = []
    labels = []

    img_idx = np.arange(n_imgs)

    for obj in range(n_objs):
        obj_list = []
        obj_img_idx = np.random.permutation(img_idx).reshape(n_views, n_imgs // n_views)
        labels += (n_imgs // n_views) * [obj]

        for view, indices in enumerate(obj_img_idx):
            sub_view = []
            for i, idx in enumerate(indices):
                if n_objs == 20:
                    fname = os.path.join(data_dir, f"obj{obj + 1}__{idx}.png")
                    img = imread(fname)[None, ...]
                else:
                    fname = os.path.join(data_dir, f"obj{obj + 1}__{idx * 5}.png")
                    img = imread(fname)
                if n_objs == 100:
                    img = np.transpose(img, (2, 0, 1))
                sub_view.append(img)
            obj_list.append(np.array(sub_view))
        views.append(np.array(obj_list))
    views = np.array(views)
    views = np.transpose(views, (1, 0, 2, 3, 4, 5)).reshape(n_views, n, *img_size)
    labels = np.array(labels)
    export_dataset(f"coil-{n_objs}", views=views, labels=labels)


def _load_npz(name):
    return np.load(os.path.join(PROCESSED_DATA_ROOT, f"{name}.npz"))

def _load_mat(name):
    return sio.loadmat(os.path.join(PROCESSED_DATA_ROOT, f"{name}.mat"))


class MultiviewDataset(Dataset):

    def __init__(self, views, labels, transform=None):
        self.data = views
        self.targets = torch.LongTensor(labels)
        if self.targets.min() == 1:
            self.targets -= 1
        self.transform = transform
        self.num_view = len(self.data)

    def __getitem__(self, idx):
        views = [self.data[v][idx].float() for v in range(self.num_view)]
        if self.transform is not None:
            views = [self.transform(view) for view in views]
        return views, self.targets[idx]

    def __len__(self):
        return len(self.targets)


def load_dataset(name, classification=False):
    mat = _load_mat(name)
    if name == 'Scene-15':
        X = mat['X'][0]
        XV1 = X[0].astype('float32')
        XV2 = X[1].astype('float32')
        Y = np.squeeze(mat['Y'])
    elif name == 'LandUse-21':
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)
        XV1 = train_x[1][index]
        XV2 = train_x[2][index]
        Y = np.squeeze(mat['Y']).astype('int')[index]
    elif name == 'NoisyMNIST':
        train = DataSet_NoisyMNIST(mat['X1'], mat['X2'], mat['trainLabel'])
        tune = DataSet_NoisyMNIST(mat['XV1'], mat['XV2'], mat['tuneLabel'])
        test = DataSet_NoisyMNIST(mat['XTe1'], mat['XTe2'], mat['testLabel'])
        XV1 = np.concatenate([tune.images1, test.images1], axis=0)
        XV2 = np.concatenate([tune.images2, test.images2], axis=0)
        Y = np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])])
    elif name == 'Caltech101-20':
        X = mat['X'][0]
        x = X[3]
        XV1 = utils.normalize(x).astype('float32')
        x = X[4]
        XV2 = utils.normalize(x).astype('float32')
        Y = np.squeeze(mat['Y']).astype('int')
    else:
        raise ValueError("Dataset name error.")
    view1, view2 = torch.from_numpy(XV1), torch.from_numpy(XV2)
    if classification:
        size = view1.shape[0] // 10
        train_num = size * 8
        valid_num = size
        train_set, train_labels = [view1[:train_num, :], view2[:train_num, :]], Y[:train_num]
        valid_set, valid_labels = [view1[train_num:train_num+valid_num, :], view2[train_num:train_num+valid_num, :]], Y[train_num: train_num+valid_num]
        test_set, test_labels = [view1[train_num+valid_num:, :], view2[train_num+valid_num:, :]], Y[train_num+valid_num:]
        return MultiviewDataset(train_set, train_labels), MultiviewDataset(valid_set, valid_labels), MultiviewDataset(test_set, test_labels)
    else:
        dataset = MultiviewDataset([view1, view2], Y)
        return dataset


class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                # print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                # print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]





if __name__ == '__main__':
    # for name in ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST']:
    #     print(name)
    #     dataset = load_dataset(name)
    #     print("Data size: ", len(dataset))
    #     [x1, x2], y = dataset[0]
    #     print(f"label num: {len(dataset.targets.unique())}, start: {dataset.targets.unique()}")
    #     print(f"x1.shape: {x1.shape}, x2.shape: {x2.shape}, label: {y}")
        
    #     print("For classification:")
    #     train_dataset, valid_dataset, test_dataset = load_dataset(name, classification=True)
    #     print(f"train set: {len(train_dataset)}, valid set: {len(valid_dataset)}, test set: {len(test_dataset)}")
    dataset = load_dataset('NoisyMNIST')    
    y = dataset.targets.unique()
    x1s = []
    x2s = []
    ys = []

    for _ in y:
        idx = dataset.targets == _
        x1, x2, yy = dataset.data[0][idx, :], dataset.data[1][idx, :], dataset.targets[idx]
        x1, x2, yy = x1[:100], x2[:100], yy[:100]
        x1s.append(x1)
        x2s.append(x2)
        ys.append(yy)
    
    x1s = torch.vstack(x1s)
    x2s = torch.vstack(x2s)
    ys = torch.concat(ys)
    
    test_batch = {
        "x1": x1s,
        "x2": x2s,
        "y": ys
    }
    torch.save(test_batch, './experiments/results/test_batch')
    
            
        
        
