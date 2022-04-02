from abc import ABC, abstractmethod
from typing import Tuple

import torch


class DatasetInterface(ABC):

    @abstractmethod
    def __init__(self, data_dir: str, transforms, **kwargs) -> None:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass