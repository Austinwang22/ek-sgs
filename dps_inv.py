import torch
import numpy as np
import dnnlib
import pickle
import os

import yaml
from helper import *
from argparse import ArgumentParser
from tqdm import tqdm
from omegaconf import OmegaConf

from utils.img_utils import *
from imaging import *
from mbd import *
from sgs import *
from dps_inv import *


def get_exp_decay_scale(min, max, decay_rate, N):
    lrs = torch.zeros(N)
    lrs = torch.pow(decay_rate, torch.arange(0, N)) * max
    lrs = torch.maximum(lrs, torch.ones_like(lrs) * min)
    return lrs


@torch.no_grad()
def dps_sampler(
    net,
    forward_op,
    observation,
    x_initial = None,
    scale=1000,
    num_steps=250,
    sigma_start=80.0,
    sigma_eps=0.002,
    rho=7,
):
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=observation.device)
    t_steps = (sigma_start ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_eps ** (1 / rho) - sigma_start ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    if x_initial == None:
        x_initial = torch.randn([1, 3, net.img_resolution, net.img_resolution], device=observation.device) * sigma_start
        
    x_next = x_initial.to(torch.float64)
    x_next.requires_grad = True
    pbar = tqdm(range(num_steps))
    for i in pbar:
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        x_cur = x_next

        with torch.enable_grad():
            denoised = net(x_cur, t_cur).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        
        with torch.enable_grad():
            difference = observation - forward_op.forward(denoised)
            norm = torch.linalg.norm(difference)
        ll_grad = torch.autograd.grad(norm, x_cur)[0]
        d_ll = ll_grad * t_cur
        
        pbar.set_description(f'Iteration {i + 1}/{num_steps}. Avg. Error: {difference.mean().item()}')
        
        coef = scale / norm
            
        with torch.enable_grad():
            x_next = x_cur + (t_next - t_cur) * d_cur
            x_next += (t_next - t_cur) * d_ll * coef

    return x_next


def subprocess(args):
    config = OmegaConf.load(args.config)
    
    device = torch.device(config.model.device)
    
    torch.manual_seed(args.seed)
    
    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    
    logger = create_logger(basedir)
    logger.info(f'Output directory created at {basedir}')
    ref_path_list = [os.path.join(config.ref_dir, f'{str(test_id).zfill(5)}.png') for test_id in args.id_list]
    num_samples = len(ref_path_list)
    logger.info(f'Found {num_samples} samples in {config.ref_dir}')
    
    model_config = OmegaConf.to_container(config.operator)
    if config.operator.name == 'inpainting':
        mask_opt = OmegaConf.to_container(config.mask_opt)
        masker = mask_generator(**mask_opt)
        mask = masker.gen_mask()
        model_config['mask'] = mask
    
    forward_operator = get_operator(**model_config)
    # save the forward operator
    with open(os.path.join(basedir, 'forward_operator.pkl'), 'wb') as f:
        pickle.dump({'forward': forward_operator}, f)
        
    with open_url(config.model.url, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
        net.eval()
        
    test = []
                
    
    for i, ref_path in enumerate(ref_path_list):
        # set up directory for each reference image
        filename = os.path.basename(ref_path).replace('.png', '')
        base_dir = os.path.join(basedir, f'{filename}')
        os.makedirs(base_dir, exist_ok=True)
        gt_dir = os.path.join(base_dir, 'ground_truth')
        os.makedirs(gt_dir, exist_ok=True)
        ob_dir = os.path.join(base_dir, 'observations')
        os.makedirs(ob_dir, exist_ok=True)
        gen_dir = os.path.join(base_dir, 'generated')
        os.makedirs(gen_dir, exist_ok=True)
        prog_dir = os.path.join(base_dir, config.log.progdir)
        os.makedirs(prog_dir, exist_ok=True)

        # load the reference image
        logger.info(f'Prcessing {ref_path}, ({i+1}/{num_samples})')
        ref_img = load_raw_image(ref_path)  # (C, H, W), np.uint8
        save_image(ref_img, os.path.join(gt_dir, 'ground_truth.png'))

        ref_img = normalize_img(ref_img)    # (C, H, W), np.float32
        ref_img = torch.from_numpy(ref_img).to(torch.float32).to(device)    # (C, H, W), torch.float32
        
        test.append(ref_img)

        # get the observation
        observation = forward_operator(ref_img)   # (C, H, W), torch.float32
        obs_np = unnormalize_img(observation.detach().cpu().numpy())    # (C, H, W), np.uint8
        save_image(obs_np, os.path.join(ob_dir, 'observation.png'))
        
        if args.x_initial != 'none':
            x_initial = torch.load(args.x_initial).to(device)
        else:
            img_size = config.mask_opt.image_size
            x_initial = torch.randn((config.sampling.num_samples, 3, img_size, img_size), device=device) * config.model.sigma_max
        
        samples = dps_sampler(net, forward_operator, observation, x_initial=x_initial, scale=config.sampling.scale, 
                              num_steps=config.sampling.num_steps, sigma_start=config.model.sigma_max, 
                              sigma_eps=config.model.sigma_min, rho=config.sampling.rho)
        
        figs_path = os.path.join(gen_dir, 'samples.png')
        plot_images((samples * 127.5 + 128).clip(0, 255).to(torch.uint8), figs_path)
        ckpt_path = os.path.join(gen_dir, 'samples.pt')
        torch.save(samples, ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dps64_inpaint.yml')
    parser.add_argument('--id_list', type=parse_int_list, default='0-9')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--x_initial', type=str, default='none')
    args = parser.parse_args()
    subprocess(args)