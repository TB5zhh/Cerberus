import os
from typing import Tuple

import torch
from PIL import Image

from .base import DatasetInterface


class EmulatedDataset(torch.utils.data.Dataset):
    """
    Don't forget to turn on shuffle in DataLoader
    """
    def __init__(self, data_dir: str, transforms, **kwargs) -> None:
        assert kwargs['cut'] >= 0, 'Negative start index?'
        assert kwargs['type'] in ['vehicle', 'road', 'legacy']
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
        self.transforms = transforms
        print(f"In {data_dir} {len(self.img_list)} samples are found")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        data = [
            Image.open(self.img_list[index]),
            Image.open(self.mask_list[index]),
        ]
        out_data = list(self.transforms(*data))
        out_data.append(self.img_list[index])
        return (*self.transforms(*data), self.img_list[index])

    def __len__(self):
        return len(self.img_list)
