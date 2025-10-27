import os, torch
from torch.utils.data.dataset import Dataset
import numpy as np
import PIL
from util.augmentation import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation

class MY_dataset(Dataset):

    def __init__(self, data_dir, split, input_h=288, input_w=512):
        super(MY_dataset, self).__init__()

        with open(os.path.join(data_dir, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w

        self.n_data = len(self.names)

        scale_range = (0.5, 2.0)
        self.augmentation_methods = Compose([
            RandomScale(scale_range),
        ]) if split == 'train' else None


    def read_image(self, name, folder,head):
        file_path = os.path.join(self.data_dir, '%s/%s%s.png' % (folder, head,name))
        image     = np.asarray(PIL.Image.open(file_path))
        return image

    def __getitem__(self, index):
        name = self.names[index]

        image = PIL.Image.open(os.path.join(self.data_dir, 'left/left%s.png' % name))
        label = PIL.Image.open(os.path.join(self.data_dir, 'labels/label%s.png' % name))
        depth = PIL.Image.open(os.path.join(self.data_dir, 'depth/depth%s.png' % name))

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
        }

        if self.split == 'train' and self.augmentation_methods:
            sample = self.augmentation_methods(sample)


        image = np.asarray(sample['image'].resize((self.input_w, self.input_h)))
        image = image.astype('float32')
        image = np.transpose(image, (2, 0, 1)) / 255.0

        depth = np.asarray(sample['depth'].resize((self.input_w, self.input_h)))
        depth = depth.astype('float32')
        M = max(depth.max(), 1e-8)
        depth = depth / M

        label = np.asarray(sample['label'].resize((self.input_w, self.input_h), resample=PIL.Image.NEAREST))
        label = label.astype('int64')

        return torch.cat((torch.tensor(image), torch.tensor(depth).unsqueeze(0)), dim=0), torch.tensor(label), name

    def __len__(self):
        return self.n_data

