import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from fastai.data.external import untar_data, URLs

coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"

paths = glob.glob(coco_path+"/*.jpg") # All paths to COCO dataset images
np.random.seed(123) # Seeding for reproducible results
paths_subset = np.random.choice(paths, 10_000, replace=False) # Randomly choosing 10,000 images
rand_idxs = np.random.permutation(10_000) # Shuffling the indexes
train_idxs = rand_idxs[:8000] # Using first 8000 images for training
val_idxs = rand_idxs[8000:] # Using last 2000 images for validating
train_paths = paths_subset[train_idxs] 
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                # transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):  # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
