import argparse
import os

import numpy as np
import torch
import wandb
import yaml
from skimage.metrics import (
    peak_signal_noise_ratio as compare_psnr,
    structural_similarity as compare_ssim,
)
from tqdm import tqdm

from utils.SSIM import SSIM
from utils.common_utils import (
    dict2namespace,
    ensure_reproducibility,
    get_kernel_network,
    get_color_image,
    np_to_torch,
    torch_to_np,
    get_image,
    apply_kernel,
)
from utils.sd3_utils import SD3Inferencer
from utils.wandb_utils import log_wandb, init_wandb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default=r"./configs/afhq.yml",
    help="path of config file",
)
parser.add_argument("--use_wandb", type=bool, default=True, help="whether to use wandb")
parser.add_argument(
    "--load_pretrain", type=bool, default=False, help="whether to load pretrain model"
)
parser.add_argument(
    "--save_checkpoint", type=bool, default=False, help="whether to save checkpoint"
)
opt = parser.parse_args()
print(opt)


def main():
    with open(opt.config_path, "r") as f:
        config1 = yaml.safe_load(f)
    config = dict2namespace(config1)
    ### wandb initialization
    if opt.use_wandb:
        init_wandb(config, mode="deblur")
    ### Reproducibility
    ensure_reproducibility(config.sd_config.SEED)
    dtype = torch.cuda.FloatTensor
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    ### Define the sd3 network
    with torch.no_grad():
        if isinstance(config.sd_config.MODEL, str) and config.sd_config.MODEL.lower() == "none":
            config.sd_config.MODEL = None
        if isinstance(config.sd_config.VAEFile, str) and config.sd_config.VAEFile.lower() == "none":
            config.sd_config.VAEFile = None
        inferencer = SD3Inferencer()
        inferencer.load(config.sd_config.MODEL, config.sd_config.VAEFile, config.sd_config.SHIFT, verbose=False)

    ### Define the kernel network
    netE, netG = get_kernel_network(
        config.data.netG_path, config.data.netE_path, config.data.kernel_size
    )
    ### Define data
    new_path = os.path.join(config.data.save_path, config.data.img_name)
    os.makedirs(new_path, exist_ok=True)
    image_path = os.path.join(config.data.data_path, config.data.img_name + ".png")
    img, y = get_color_image(image_path)  # load image and convert to np.
    img_blur = np_to_torch(img).type(dtype) * 2 - 1  # 映射到[-1,1]
    img_gt_pillow, img_gt_np = get_image(
        os.path.join(config.data.gt_path, config.data.img_name + ".png")
    )
    y = np_to_torch(y).type(dtype)

    z = netE(y.unsqueeze(0))  # (1,100,1,1)
    w = netG.module.g1(z)  # (1,256,3,3)
    w.requires_grad = True

    # Define loss
    mse = torch.nn.MSELoss().type(dtype)
    ssim_loss = SSIM().type(dtype)
    # Define optimizer
    noise = torch.nn.parameter.Parameter(
        torch.randn(1, 16, config.sd_config.HEIGHT // 8, config.sd_config.WIDTH // 8, dtype=torch.float32).cuda(),
        requires_grad=True
    )

    if opt.load_pretrain:
        noise = torch.load(os.path.join(config.data.models_path, "noise.pth"))
        w = torch.load(os.path.join(config.data.models_path, "w.pth"))
        w.requires_grad = True
        print("load pretrain model")
    optimizer = torch.optim.Adam(
        [
            {"params": noise, "lr": config.task_config.lr_img},
            {"params": w, "lr": config.task_config.lr_kernel},
        ]
    )
    metric = {"PSNR": 0, "SSIM": 0, "step": 0}
    for step in tqdm(range(1, config.task_config.num_iter + 1)):
        optimizer.zero_grad()
        sampled_latent = inferencer.do_sampling(noise=noise,
                                                height=config.sd_config.HEIGHT,
                                                width=config.sd_config.WIDTH,
                                                conditioning=inferencer.get_cond(config.sd_config.PROMPT),
                                                neg_cond=inferencer.get_cond(''),
                                                steps=config.sd_config.STEPS,
                                                cfg_scale=config.sd_config.CFG_SCALE,
                                                denoise=config.sd_config.DENOISE,
                                                skip_layer_config=vars(
                                                    config.sd_config.skip_layer_config))  # (1,3,256,256)
        x_0_hat = inferencer.vae_decode(sampled_latent)
        out_k = netG.module.Gk(w)  # (1,1,31,31)
        # blurred_xt = nn.functional.conv2d(x_0_hat.view(-1, 1, config.data.image_size, config.data.image_size), out_k,
        #                                   padding="same", bias=None).view(1, 3, config.data.image_size,
        #                                                                   config.data.image_size)

        blurred_xt = apply_kernel(x_0_hat, out_k)

        # total_loss = mse(blurred_xt, img_blur)
        # total_loss = 1 - ssim_loss(blurred_xt, img_blur)
        if step < 50:
            total_loss = mse(blurred_xt, img_blur)
        else:
            total_loss = 1 - ssim_loss(blurred_xt, img_blur)
        total_loss.backward()
        optimizer.step()

        # if step % config.task_config.save_frequency == 0:
        out_x_np = inferencer.process(x_0_hat)  # [256,256,3] unit8
        psnr = compare_psnr(np.array(img_gt_pillow), out_x_np)
        ssim = compare_ssim(
            np.array(img_gt_pillow), out_x_np, multichannel=True, channel_axis=2
        )
        tqdm.write(f"PSNR={psnr:.2f}, SSIM={ssim:.3f}, loss={total_loss}")

        # out_x_np = out_x_np[padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2], :]
        # save_path = os.path.join(new_path, f'{step}_x.png')
        # Image.fromarray(out_x_np).save(save_path)
        # save_path = os.path.join(new_path, f'{step}_k.png')
        out_k_np = torch_to_np(out_k)
        out_k_np = out_k_np.squeeze()
        out_k_np /= np.max(out_k_np)
        # save_img_np(save_path, out_k_np)
        if opt.use_wandb:
            log_wandb(step, total_loss, psnr, ssim, out_x_np, out_k_np)

        if psnr > metric["PSNR"]:
            metric["PSNR"] = psnr
            metric["step"] = step

        if opt.save_checkpoint:
            torch.save(
                noise,
                os.path.join(config.data.models_path, "noise", f"noise-{step}.pth"),
            )
            torch.save(w, os.path.join(config.data.models_path, "w", f"w-{step}.pth"))

    if opt.use_wandb:
        wandb.finish()
    print(f"Best PSNR={metric['PSNR']}, step={metric['step']}")


if __name__ == "__main__":
    main()
