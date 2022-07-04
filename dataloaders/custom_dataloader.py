import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
from PIL import Image
import logging
import os
import fnmatch


class CustomDataset(Dataset):
    def __init__(self, image_dir='edina_test', glob_patterns=None, skip_every_n_image=1, transform=None, size=(480, 640)):
        super(CustomDataset, self).__init__()

        # Default glob patterns
        if glob_patterns is None:
            glob_patterns = ['*.png', '*.jpg', '*.jpeg']

        self.to_tensor = transforms.ToTensor()
        self.__transform = transform
        self.im_h, self.im_w = size[0], size[1]

        self.root = image_dir

        self.data_info = []
        for glob_pattern in glob_patterns:
            self.data_info.extend(sorted(fnmatch.filter(os.listdir(self.root), glob_pattern)))

        self.data_info = self.data_info[::skip_every_n_image]

        # add the root back in for full path
        self.data_info = [os.path.join(self.root, path) for path in self.data_info]

        self.data_len = len(self.data_info)

        logging.info('Number of frames in directory {0} is {1}.'.format(image_dir, self.data_len))

    def __getitem__(self, index):
        color_info = self.data_info[index]
        # Open image
        color_img = Image.open(color_info).convert('RGB')

        # IDEO dataset
        if color_img.width == 960 and color_img.height == 540:
            # print('Cropping image')
            color_img = color_img.crop((120, 0, 840, 540))
            color_img = color_img.resize((640, 480), resample=Image.BILINEAR)

        color_img = color_img.resize((self.im_w, self.im_h), resample=Image.BILINEAR)
        assert color_img.width == self.im_w and color_img.height == self.im_h
        color_img_np = np.array(color_img) / 255.

        # transforms
        if self.__transform is not None:
            # sample
            sample = {}
            sample["image"] = color_img_np

            sample = self.__transform(sample)

            return {'image': torch.Tensor(sample["image"]),
                    'image-original': torch.Tensor(color_img_np),
                    'image-name': color_info}

        else:
            # This is for baseline depth
            return {'image': self.to_tensor(color_img),
                    'image-original': torch.Tensor(color_img_np),
                    'image-name': color_info}

    def __len__(self):
        return self.data_len