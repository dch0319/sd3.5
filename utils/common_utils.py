import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn

from networks.knet import Generator, ResNet18


def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_kernel_network(netG_path, netE_path, kernel_size):
    netG = Generator(kernel_size).cuda()
    netG = nn.DataParallel(netG)
    netG.load_state_dict(torch.load(netG_path))
    for p in netG.parameters():
        p.requires_grad = False
    netG.eval()

    netE = ResNet18().cuda()
    netE = nn.DataParallel(netE)
    netE.load_state_dict(torch.load(netE_path))
    for p in netE.parameters():
        p.requires_grad = False
    netE.eval()

    return netE, netG


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def get_color_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    y = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(y)

    # img = img[:512, :512, :]

    img = img
    img = img.transpose(2, 0, 1)

    return img.astype(np.float32) / 255., y.astype(np.float32) / 255.


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From H x W x C [0..1] to  H x W x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 1:
            ar = ar[2]
        else:
            ar = ar  # .transpose(1, 2, 0)

    return Image.fromarray(ar)


def save_img_np(save_dir, img_np):
    img_pil = np_to_pil(img_np)
    img_pil.save(save_dir)


def process(sample, i):
    image_processed = sample.detach().cpu().permute(0, 2, 3, 1)
    image_processed = image_processed.squeeze(0)
    image_processed = torch.clip((image_processed + 1.0) * 127.5, 0., 255.)
    image_processed = image_processed.numpy().astype(np.uint8)
    return image_processed


def compare_psnr_tensor(gt_tensor, blurry_tensor):
    if len(gt_tensor.shape) == len(blurry_tensor.shape) == 4:
        return peak_signal_noise_ratio(gt_tensor[0].detach().cpu().numpy(), blurry_tensor[0].detach().cpu().numpy())
    elif len(gt_tensor.shape) == len(blurry_tensor.shape) == 3:
        return peak_signal_noise_ratio(gt_tensor.detach().cpu().numpy(), blurry_tensor.detach().cpu().numpy())
    else:
        assert False


def compare_ssim_tensor(gt_tensor, blurry_tensor):
    if len(gt_tensor.shape) == len(blurry_tensor.shape) == 4:
        return structural_similarity(gt_tensor[0].detach().cpu().numpy(), blurry_tensor[0].detach().cpu().numpy(),
                                     channel_axis=0, data_range=1)
    elif len(gt_tensor.shape) == len(blurry_tensor.shape) == 3:
        return structural_similarity(gt_tensor.detach().cpu().numpy(), blurry_tensor.detach().cpu().numpy(),
                                     channel_axis=0, data_range=1)
    else:
        assert False


def blurring(cleanimg, kernel, kernelsize):
    '''
    cleanimg: B x 1 x H x W
    kernel: B x 1 x kernelsize x kernelsize
    '''
    num_pad = (kernelsize - 1) // 2

    clean_img_pad = F.pad(cleanimg, (num_pad,) * 4, mode='reflect')
    out = F.conv3d(clean_img_pad.unsqueeze(0), kernel.unsqueeze(1),
                   groups=cleanimg.shape[0])  # 1 x B x C x H x W
    return out.squeeze(0)


def apply_kernel(data, kernel):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    b_img = torch.zeros_like(data).to(device)
    num_pad = kernel.shape[-1] // 2
    data_pad = F.pad(data, (num_pad,) * 4, mode='reflect')
    for i in range(3):
        b_img[:, i, :, :] = F.conv2d(data_pad[:, i:i + 1, :, :], kernel)
    return b_img
