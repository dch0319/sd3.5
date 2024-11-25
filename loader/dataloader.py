import random
from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms.functional as TVTF
from torch.utils.data import DataLoader

__DATASET__ = {}

from torchvision.datasets import VisionDataset


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls

    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(
        dataset: VisionDataset, batch_size: int, num_workers: int, train: bool
):
    dataloader = DataLoader(
        dataset, batch_size, shuffle=train, num_workers=num_workers, drop_last=train
    )
    return dataloader


def random_crop_patch(im, patch_size):
    H = im.shape[0]
    W = im.shape[1]
    if H < patch_size or W < patch_size:
        H = max(patch_size, H)
        W = max(patch_size, W)
        im = cv2.resize(im, (W, H))
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)
    patch = im[ind_H: ind_H + patch_size, ind_W: ind_W + patch_size]
    return patch


@register_dataset("Kernel")
class KernelDataset(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        data = np.load(root)
        self.kernel_list = data["arr_0"]

    def __len__(self):
        return len(self.kernel_list)

    def __getitem__(self, index):
        kernel = self.kernel_list[index]
        kernel = TVTF.to_tensor(kernel.copy())
        return kernel


@register_dataset("OpenImage")
class OpenImageDataset(VisionDataset):
    def __init__(self, root, patch_size=256):
        super().__init__(root)
        root = Path(root)
        a = list(root.glob("*.jpg"))
        self.image_list = sorted([str(x) for x in a])
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        image, _, _ = cv2.split(image)
        image = random_crop_patch(image, self.patch_size)
        image = TVTF.to_tensor(image.copy())
        return image


@register_dataset("CelebAHQ")
class CelebAHQDataset(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        root = Path(root)
        a = list(root.glob("*.jpg"))
        self.image_list = sorted([str(x) for x in a])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        image, _, _ = cv2.split(image)
        image = TVTF.to_tensor(image.copy())
        return image


@register_dataset("FFHQ")
class FFHQDataset(CelebAHQDataset):
    def __init__(self, root):
        super().__init__(root)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)


@register_dataset("AFHQ")
class AFHQDataset(CelebAHQDataset):
    def __init__(self, root):
        super().__init__(root)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return super().__getitem__(index)
