import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch

class Shapes3DDataset(Dataset):
    def __init__(self, hdf5_path, allowed_compositions, object_hue_values, scale_values, shape_values, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.object_hue_map = {round(v, 1): i for i, v in enumerate(object_hue_values)}
        self.scale_map = {round(v, 1): i for i, v in enumerate(scale_values)}
        self.shape_map = {int(v): i for i, v in enumerate(shape_values)}

        self.allowed_compositions = set(allowed_compositions)
        print(f"Allowed compositions: {self.allowed_compositions}")
        self.indices = []
        with h5py.File(self.hdf5_path, 'r') as f:
            labels = f['labels']

            for i in range(len(labels)):
                obj_hue = round(labels[i, 2], 1)
                scale = round(labels[i, 3], 1)
                shape = int(labels[i, 4])

                if obj_hue in self.object_hue_map and scale in self.scale_map and shape in self.shape_map:
                    comp = (
                        self.shape_map[shape],
                        self.object_hue_map[obj_hue],
                        self.scale_map[scale]
                        
                    )
                    if comp in self.allowed_compositions:
                        self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        with h5py.File(self.hdf5_path, 'r') as f:
            img = f['images'][index]
            label = f['labels'][index]

        image = Image.fromarray(img)
        if self.transform:
            image = self.transform(image)

        obj_hue = round(label[2], 1)
        scale = round(label[3], 1)
        shape = int(label[4])
        print(f"obj_hue: {obj_hue}, scale: {scale}, shape: {shape}")
        label_tuple = (
            self.shape_map[shape],
            self.object_hue_map[obj_hue],
            self.scale_map[scale]
            
        )
        #label = {i: int(name_labels[i]) for i in range(3)}

        return image, torch.tensor(label_tuple)
