import datetime

import wandb


def init_wandb(config, mode):
    assert mode in ('generator', 'initializer', 'deblur')
    run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if mode == 'generator':
        wandb.init(project=mode, name=f'{run_name}',
                   config={
                       'dataset': config.data.dataset,
                       'kernel_size': config.data.kernel_size,
                       'num_epochs': config.task_config.num_epochs,
                       'lr': config.task_config.lr,
                       'batch_size': config.task_config.batch_size,
                   })
    elif mode == 'initializer':
        wandb.init(project=mode, name=f'{run_name}',
                   config={
                       'dataset': config.data.dataset,
                       'kernel_size': config.data.kernel_size,
                       'num_iters': config.task_config.num_iters,
                       'num_epochs': config.task_config.num_epochs,
                       'lr': config.task_config.lr,
                       'lr_z': config.task_config.lr_z,
                       'weight_lambda': config.task_config.weight_lambda,
                       'batch_size': config.task_config.batch_size,
                   })
    elif mode == 'deblur':
        wandb.init(project=mode, name=f'{run_name}',
                   config={
                       'dataset': config.data.dataset,
                       'img_name': config.data.img_name,
                       'num_iter': config.task_config.num_iter,
                       'lr_img': config.task_config.lr_img,
                       'lr_kernel': config.task_config.lr_kernel,
                       'delta_t': config.task_config.delta_t,
                   })


def log_wandb(step, total_loss, psnr, ssim, img_gt, img_kernel):
    # log
    wandb.log(step=step,
              data={'total_loss': total_loss.item(),
                    'psnr': psnr,
                    'ssim': ssim,
                    'image': wandb.Image(img_gt),
                    'kernel': wandb.Image(img_kernel)
                    })
