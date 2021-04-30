# License: MIT
# Author: Karl Stelzner

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from numpy.random import random_integers
from PIL import Image
from torch.utils.data._utils.collate import default_collate


def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def make_sprites(n=50000, height=64, width=64):
    images = np.zeros((n, height, width, 3))
    counts = np.zeros((n,))
    print('Generating sprite dataset...')
    for i in range(n):
        num_sprites = random_integers(0, 2)
        counts[i] = num_sprites
        for j in range(num_sprites):
            pos_y = random_integers(0, height - 12)
            pos_x = random_integers(0, width - 12)

            scale = random_integers(12, min(16, height-pos_y, width-pos_x))

            cat = random_integers(0, 2)
            sprite = np.zeros((height, width, 3))

            if cat == 0:  # draw circle
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        dist_center_sq = (x - center_x)**2 + (y - center_y)**2
                        if  dist_center_sq < (scale // 2.0)**2:
                            sprite[x][y][cat] = 1.0
            elif cat == 1:  # draw square
                sprite[pos_x:pos_x + scale, pos_y:pos_y + scale, cat] = 1.0
            else:  # draw square turned by 45 degrees
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        if abs(x - center_x) + abs(y - center_y) < (scale // 2.0):
                            sprite[x][y][cat] = 1.0
            images[i] += sprite
        if i % 100 == 0:
            progress_bar(i, n)
    images = np.clip(images, 0.0, 1.0)

    return {'x_train': images[:4 * n // 5],
            'count_train': counts[:4 * n // 5],
            'x_test': images[4 * n // 5:],
            'count_test': counts[4 * n // 5:]}


class Sprites(Dataset):
    def __init__(self, directory, n=50000, canvas_size=64,
                 train=True, transform=None):
        np_file = 'sprites_{}_{}.npz'.format(n, canvas_size)
        full_path = os.path.join(directory, np_file)
        if not os.path.isfile(full_path):
            try:
                os.mkdir('./data')
            except:
                print("data folder found !")
            gen_data = make_sprites(n, canvas_size, canvas_size)
            np.savez(full_path, **gen_data)

        data = np.load(full_path)

        self.transform = transform
        self.images = data['x_train'] if train else data['x_test']
        self.counts = data['count_train'] if train else data['count_test']

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img).float()
        return img, self.counts[idx]


class Clevr(Dataset):
    def __init__(self, directory, train=True, transform=None):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.n = len(self.filenames)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        imgpath = os.path.join(self.directory, self.filenames[idx])
        img = Image.open(imgpath)
        if self.transform is not None:
            img = self.transform(img).float()
        return img, 1
        
        
########################################################################
#
#  ADDED
#
#########################################################################

class MultiObjectDataset(Dataset):

    def __init__(self, data_path, train, split=0.9, transform = None):
        super().__init__()

        # Load data
        data = np.load(data_path, allow_pickle=True)

        # Rescale images and permute dimensions
        x = np.asarray(data['x'], dtype=np.float32) / 255
        x = np.transpose(x, [0, 3, 1, 2])  # batch, channels, h, w

        # Get labels
        try:
            labels = data['labels'].item()
        except:
            labels = data['labels']
            print(type(labels))

        # Split train and test
        split = int(split * len(x))
        if train:
            indices = range(split)
        else:
            indices = range(split, len(x))

        # From numpy/ndarray to torch tensors (labels are lists of tensors as
        # they might have different sizes)
        self.x = torch.from_numpy(x[indices])
        
        try:
            labels.pop('text', None)
            labels.pop('brut', None)
        except:
            print("No text to pop !")
        
        self.labels = self._labels_to_tensorlist(labels, indices)


    @staticmethod
    def _labels_to_tensorlist(labels, indices):
        out = {k: [] for k in labels.keys()}
        for i in indices:
            for k in labels.keys():
                t = labels[k][i]
                t = torch.as_tensor(t)
                out[k].append(t)
        return out

    def __getitem__(self, index):
        x = self.x[index]
        try:
            labels = {k: self.labels[k][index] for k in self.labels.keys()}
        except:
            labels = self.labels
        return x, labels

    def __len__(self):
        return self.x.size(0)
        
class MultiObjectDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        assert 'collate_fn' not in kwargs
        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(batch):
        

        # The input is a batch of (image, label_dict)
        _, item_labels = batch[0]
        keys = item_labels.keys()

        # Max label length in this batch
        # max_len[k] is the maximum length (in batch) of the label with name k
        # If at the end max_len[k] is -1, labels k are (probably all) scalars
        max_len = {k: -1 for k in keys}

        # If a label has more than 1 dimension, the padded tensor cannot simply
        # have size (batch, max_len). Whenever the length is >0 (i.e. the sequence
        # is not empty, store trailing dimensions. At the end if 1) all sequences
        # (in the batch, and for this label) are empty, or 2) this label is not
        # a sequence (scalar), then the trailing dims are None.
        trailing_dims = {k: None for k in keys}

        # Make first pass to get shape info for padding
        for _, labels in batch:
            for k in keys:
                try:
                    max_len[k] = max(max_len[k], len(labels[k]))
                    if len(labels[k]) > 0:
                        trailing_dims[k] = labels[k].size()[1:]
                except TypeError:   # scalar
                    pass

        # For each item in the batch, take each key and pad the corresponding
        # value (label) so we can call the default collate function
        pad = MultiObjectDataLoader._pad_tensor
        for i in range(len(batch)):
            for k in keys:
                if trailing_dims[k] is None:
                    continue

                size = [max_len[k]] + list(trailing_dims[k])
                batch[i][1][k] = pad(batch[i][1][k], size)

        return default_collate(batch)
    
    @staticmethod
    def _pad_tensor(x, size, value=None):
        assert isinstance(x, torch.Tensor)
        input_size = len(x)
        if value is None:
            value = float('nan')

        # Copy input tensor into a tensor filled with specified value
        # Convert everything to float, not ideal but it's robust
        out = torch.zeros(*size, dtype=torch.float)
        out.fill_(value)
        if input_size > 0:  # only if at least one element in the sequence
            out[:input_size] = x.float()
        return out

