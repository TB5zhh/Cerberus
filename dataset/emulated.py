import os
import re
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from .base import DatasetInterface

from IPython import embed
class EmulatedDataset(torch.utils.data.Dataset):
    """
    Don't forget to turn on shuffle in DataLoader
    """
    def __init__(self, data_dir: str, transforms, **kwargs) -> None:
        assert kwargs['cut'] >= 0, 'Negative start index?'
        assert kwargs['type'] in ['vehicle', 'road', 'legacy', 'double']
        if kwargs['type'] == 'double':
            self.img_list_v, self.img_list_x = [], []
            self.mask_list_v, self.mask_list_x = [], []
            self.depth_list_v, self.depth_list_x = [], []
            self.pose_v, self.pose_x = [], []
            series = sorted(os.listdir(data_dir), key=lambda s: int(s.split('.')[0])) if 'series' not in kwargs.keys() else kwargs['series']
            for s in series:
                s = str(s)
                if not os.path.isdir(f'{data_dir}/{s}'):
                    continue
                for i in sorted(os.listdir(os.path.join(data_dir, s, 'depth_v')), key=lambda s: int(s.split('.')[0])):
                    self.depth_list_v.append(os.path.join(data_dir, s, 'depth_v', i))
                for i in sorted(os.listdir(os.path.join(data_dir, s, 'depth_x')), key=lambda s: int(s.split('.')[0])):
                    self.depth_list_x.append(os.path.join(data_dir, s, 'depth_x', i))
                for i in sorted(os.listdir(os.path.join(data_dir, s, 'mask_v_idx')), key=lambda s: int(s.split('.')[0])):
                    self.mask_list_v.append(os.path.join(data_dir, s, 'mask_v_idx', i))
                for i in sorted(os.listdir(os.path.join(data_dir, s, 'mask_x_idx')), key=lambda s: int(s.split('.')[0])):
                    self.mask_list_x.append(os.path.join(data_dir, s, 'mask_x_idx', i))
                for i in sorted(os.listdir(os.path.join(data_dir, s, 'rgb_v')), key=lambda s: int(s.split('.')[0])):
                    self.img_list_v.append(os.path.join(data_dir, s, 'rgb_v', i))
                for i in sorted(os.listdir(os.path.join(data_dir, s, 'rgb_x')), key=lambda s: int(s.split('.')[0])):
                    self.img_list_x.append(os.path.join(data_dir, s, 'rgb_x', i))
                find = lambda a: [float(i) for i in re.findall('-?\d+\.\d+', a)]
                with open(os.path.join(data_dir, s, 'path.txt')) as f:
                    lines = [i.strip() for i in f.readlines()]
                    pose_x = find(lines[0])
                    for line in lines[1:]:
                        self.pose_x.append(pose_x)
                        self.pose_v.append(find(line))

        else:
            self.img_list = []
            self.mask_list = []
            for s in kwargs['series']:
                if not os.path.isdir(f'{data_dir}/{s}'):
                    continue
                rgb_dir = f"{data_dir}/{s}/" + ('rgb' if kwargs['type'] == 'legacy' else ('rgb_v' if kwargs['type'] == 'vehicle' else 'rgb_x'))
                mask_dir = f"{data_dir}/{s}/" + ('mask_idx' if kwargs['type'] == 'legacy' else ("mask_v_idx" if kwargs['type'] == 'vehicle' else 'mask_x_idx'))
                # mask_dir = f"{data_dir}/{s}/mask_idx" # FIXM
                last = min(len(os.listdir(rgb_dir)), len(os.listdir(mask_dir)))
                for idx in sorted(os.listdir(rgb_dir)[kwargs['cut']:last], key=lambda s: int(s.split('.')[0])):
                    self.img_list.append(f"{rgb_dir}/{idx}")
                    self.mask_list.append(f"{mask_dir}/{idx}")
            assert len(self.img_list) == len(self.mask_list)
        self.type = kwargs['type']
        self.transforms = transforms
        # print(f"In {data_dir} {len(self.img_list)} samples are found")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.type == 'double':
            data_v = [
                Image.open(self.img_list_v[index]),
                Image.open(self.mask_list_v[index]),
            ]
            data_x = [
                Image.open(self.img_list_x[index]),
                Image.open(self.mask_list_x[index]),
            ]
            depth_v = torch.as_tensor(np.asarray(Image.open(self.depth_list_v[index])), dtype=torch.float64) @ torch.as_tensor((1, 256, 65536), dtype=torch.float64) / (256 * 256 * 256 - 1) * 1000
            depth_x = torch.as_tensor(np.asarray(Image.open(self.depth_list_x[index])), dtype=torch.float64) @ torch.as_tensor((1, 256, 65536), dtype=torch.float64) / (256 * 256 * 256 - 1) * 1000

            return (
                *self.transforms(*data_v), 
                *self.transforms(*data_x), 
                depth_v, 
                depth_x, 
                torch.as_tensor(self.pose_v[index]).reshape((2, -1)), 
                torch.as_tensor(self.pose_x[index]).reshape((2, -1)),
            ) 
        else:
            data = [
                Image.open(self.img_list[index]),
                Image.open(self.mask_list[index]),
            ]
            out_data = list(self.transforms(*data))
            out_data.append(self.img_list[index])
            return (*self.transforms(*data), self.img_list[index])

    def __len__(self):
        if self.type == 'double':
            return len(self.img_list_v)
        else:
            return len(self.img_list)

if __name__ == '__main__':
    dataset = EmulatedDataset('datasets/swingcouch', lambda x, y: (x, y), )