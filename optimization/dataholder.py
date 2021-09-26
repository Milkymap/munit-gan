import numpy as np 
import itertools as it, functools as ft 

import torch as th 
from libraries.strategies import *  
from torch.utils.data import Dataset 

from glob import glob 
from os import path 

class DataHolder(Dataset):
    def __init__(self, root, extension='*.jpg', mapper=None):
        self.files = glob(path.join(root, extension))
        self.mapper = mapper 

    def normalize(self, image):
        normalized_image = image / th.max(image)  # value between 0 ~ 1 
        if self.mapper is not None: 
            mapped_image = self.mapper(normalized_image)
            return mapped_image
        return normalized_image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        current_file = self.files[idx]
        current_image = read_image(current_file, by='th')
        left_image, right_image = th.chunk(current_image, 2, dim=2)
        return self.normalize(left_image), self.normalize(right_image)


if __name__ == '__main__':
    source = DataHolder('../BICYCLE-GAN/data/edges2shoes/val/')
    L, R = source[1]    
    print(L.shape, R.shape)
