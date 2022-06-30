import re

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
from PIL import Image
import logging
import os


class ScanNetEdinaMultiRectsCropNoResizeDataset(Dataset):
    def __init__(self, usage='edina_test', skip_every_n_image=1, dataset_pickle_file='./data/scannet_edina.pkl',
                 transform=None, size=(480, 640)):
        super(ScanNetEdinaMultiRectsCropNoResizeDataset, self).__init__()
        self.to_tensor = transforms.ToTensor()
        self.usage = usage
        self.__transform = transform
        self.im_h, self.im_w = size[0], size[1]

        with open(dataset_pickle_file, 'rb') as file:
            self.data_info = pickle.load(file)[usage]

        self.idx = [i for i in range(0, len(self.data_info['color']), skip_every_n_image)]
        self.data_len = len(self.idx)
        logging.info('Number of frames for the usage {0} is {1}.'.format(usage, self.data_len))

    def __getitem__(self, index):
        # Get image name from the pandas df
        color_info = self.data_info['color'][self.idx[index]]
        depth_info = self.data_info['depth'][self.idx[index]]
        # print(index, color_info, depth_info)
        # gravity_info = self.data_info['gravity'][self.idx[index]]
        normal_info = self.data_info['normal'][self.idx[index]]
        # normal_mask_info = self.data_info['normal-mask'][self.idx[index]]

        # Open image
        color_img = Image.open(color_info).convert("RGB")

        try:
            depth_img = Image.open(depth_info).convert('F')  # Convert to float32
        except:
            logging.warning('Failed to load depth file %s, replacing with random item' % depth_info)
            return self.__getitem__(np.random.randint(0, self.data_len))

        normal_img = Image.open(normal_info)

        if 'test' in self.usage:
            if color_img.width == 960 and color_img.height == 540:
                color_img = color_img.crop((120, 0, 840, 540))
                color_img = color_img.resize((640, 480), resample=Image.BILINEAR)
                color_img = color_img.resize((self.im_w, self.im_h), resample=Image.BILINEAR)
            assert color_img.width == self.im_w and color_img.height == self.im_h

            if depth_img.width == 960 and depth_img.height == 540:
                depth_img = depth_img.crop((120, 0, 840, 540))
                depth_img = depth_img.resize((640, 480), resample=Image.NEAREST)
                depth_img = depth_img.resize((self.im_w, self.im_h), resample=Image.NEAREST)
            assert depth_img.width == self.im_w and depth_img.height == self.im_h

            if normal_img.width == 960 and normal_img.height == 540:
                normal_img = normal_img.crop((120, 0, 840, 540))
                normal_img = normal_img.resize((640, 480), resample=Image.NEAREST)
            assert normal_img.width == self.im_w and normal_img.height == self.im_h

            color_img = color_img.resize((self.im_w, self.im_h), resample=Image.BILINEAR)
            depth_img = depth_img.resize((self.im_w, self.im_h), resample=Image.NEAREST)
            normal_img = normal_img.resize((self.im_w, self.im_h), resample=Image.NEAREST)

        else:
            # while training, crop and resize for varying focal length
            assert color_img.width == depth_img.width
            assert color_img.height == color_img.height
            if color_img.width == 640 and color_img.height == 480:
                w = np.random.randint(low=450, high=640)
                h = 480 * w // 640
                y_lim, x_lim = 480 - h, 640 - w
                top = np.random.randint(low=0, high=x_lim)
                left = np.random.randint(low=0, high=y_lim)
                color_img = color_img.crop((top, left, top + w, left + h))
                color_img = color_img.resize((640, 480), resample=Image.BILINEAR)

            if depth_img.width == 640 and depth_img.height == 480:
                depth_img = depth_img.crop((top, left, top + w, left + h))
                depth_img = depth_img.resize((640, 480), resample=Image.NEAREST)

            assert color_img.width == self.im_w and color_img.height == self.im_h
            assert depth_img.width == self.im_w and depth_img.height == self.im_h

            # IDEO dataset
            if normal_img.width == 640 and normal_img.height == 480:
                # print('Cropping depth')
                normal_img = normal_img.crop((top, left, top + w, left + h))
                normal_img = normal_img.resize((640, 480), resample=Image.NEAREST)

            assert normal_img.width == self.im_w and normal_img.height == self.im_h

        normal_tensor = -self.to_tensor(normal_img) + 0.5
        normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=0)

        color_img_np = np.array(color_img) / 255.
        depth_img_np = np.array(depth_img) / 1000.0

        # mask; cf. project_depth_map.m in toolbox_nyu_depth_v2 (max depth = 10.0)
        mask = (depth_img_np > 0) & (depth_img_np < 10) # 10

        # transforms
        if self.__transform is not None:
            # sample
            sample = {}
            sample["image"] = color_img_np
            sample["depth"] = depth_img_np
            sample["mask"] = mask

            sample = self.__transform(sample)

            return {'image': torch.Tensor(sample["image"]),
                    'normal-mask': torch.Tensor(sample["mask"]),
                    'depth': torch.Tensor(sample["depth"]),
                    'image-original': torch.Tensor(color_img_np),
                    'normal': normal_tensor}

        else:
            # This is for baseline depth
            return {'image': self.to_tensor(color_img),
                    'normal-mask': torch.Tensor(mask)[None, ...],
                    'depth': (torch.Tensor(np.array(depth_img)) / 1000.0)[None, ...],
                    'image-original': torch.Tensor(color_img_np),
                    'normal': normal_tensor}

    def __len__(self):
        return self.data_len