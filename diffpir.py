import torch
import numpy as np
import dnnlib
import pickle
import os

import yaml
from helper import *
from argparse import ArgumentParser
import tqdm
from omegaconf import OmegaConf

from utils.img_utils import *
from imaging import *

@torch.no_grad()
def fgsg_estimate(forward_op, observation, x_0, num_queries=10000, mu=0.05, batch_size=2000):
    num_batches = num_queries // batch_size
    
    def f(x):
        flat = torch.flatten((observation - forward_op.forward(x)), start_dim=1)
        return torch.linalg.norm(flat, dim=1)
    
    grad_est = torch.zeros_like(x_0)
    norm = f(x_0) # torch.Size([num_samples])
    
    shape = x_0.shape
    
    for i in range(len(x_0)):
        for j in range(num_batches):
            u = torch.randn((batch_size, shape[1], shape[2], shape[3]), device=x_0.device)
            
            x0_perturbed = x_0[i] + mu * u # batch_size x C x H x W
            perturbed_norm = f(x0_perturbed) # torch.Size([batch_size])
                        
            diff = (perturbed_norm - norm[i]).reshape(batch_size, 1, 1, 1)
            prod = u * (diff / (mu * num_queries))
            grad_est[i] += prod.sum(dim=0, keepdim=True).squeeze(0)
            
    return grad_est

@torch.no_grad()
def cgsg_estimate(forward_op, observation, x_0, num_queries=10000, mu=0.05, batch_size=2000):
    num_batches = num_queries // batch_size
    
    def f(x):
        flat = torch.flatten((observation - forward_op.forward(x)), start_dim=1)
        return torch.linalg.norm(flat, dim=1)
    
    grad_est = torch.zeros_like(x_0)
    # norm = f(x_0) # torch.Size([num_samples])
    shape = x_0.shape
    
    for i in range(len(x_0)):
        for j in range(num_batches):
            u = torch.randn((batch_size, shape[1], shape[2], shape[3]), device=x_0.device)
            
            # x0_perturbed = x_0[i] + mu * u # batch_size x C x H x W
            x0_perturbed_plus = x_0[i] + mu * u
            x0_perturbed_minus = x_0[i] - mu * u
            # perturbed_norm = f(x0_perturbed) # torch.Size([batch_size])
            perturbed_norm_plus = f(x0_perturbed_plus)
            perturbed_norm_minus = f(x0_perturbed_minus)
                        
            # diff = (perturbed_norm - norm[i]).reshape(batch_size, 1, 1, 1)
            diff = (perturbed_norm_plus - perturbed_norm_minus).reshape(batch_size, 1, 1, 1)
            prod = u * (diff / (mu * num_queries))
            grad_est[i] += prod.sum(dim=0, keepdim=True).squeeze(0)
            
    return grad_est

@torch.no_grad()
def diffpir_sample(net, forward_op, observation, num_steps, sigma_max, sigma_min, rho, sigma_n, lamb, xi,
                   progdir=None, num_samples=4, no_grad=False, num_queries=10000, mu=0.05, batch_size=2000):
    device = observation.device
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    pbar = tqdm.trange(num_steps)
    xt= torch.randn(num_samples, net.img_channels, net.img_resolution, net.img_resolution, device=device).to(torch.float64) * sigma_max
    # pbar = range(num_steps)
    for step in pbar:
        pbar.set_description(f'Iteration {step + 1}: {(observation - forward_op.forward(xt)).mean().item()}')
        
        if progdir != None:
            filepath = os.path.join(progdir, f'x{step}.png')
            plot_images((xt * 127.5 + 128).clip(0, 255).to(torch.uint8), filepath)
        
        sigma = t_steps[step]
        x0 = net(xt, sigma).to(torch.float64).clone().requires_grad_(True)
        
        if not no_grad:
            with torch.enable_grad():
                loss = ((forward_op.forward(x0) - observation) ** 2).norm()
                grad = torch.autograd.grad(loss, x0)[0]
        else:
            grad = cgsg_estimate(forward_op, observation, x0, num_queries, mu, batch_size)
        
        x0hat = x0 - sigma ** 2 / (2*lamb*sigma_n**2) * grad

        effect = (xt - x0hat)/sigma
        xt = x0hat + (np.sqrt(xi)* torch.randn_like(xt) + np.sqrt(1-xi)*effect) * t_steps[step+1]
    print(xt.mean())
    return xt


def subprocess(args):
    config = OmegaConf.load(args.config)
    
    device = torch.device(config.model.device)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
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
        observation = forward_operator(ref_img.to(torch.float64))   # (C, H, W), torch.float32
        print(observation.min(), observation.max(), ref_img.min(), ref_img.max())
        obs_np = unnormalize_img(observation.detach().cpu().numpy())    # (C, H, W), np.uint8
        save_image(obs_np, os.path.join(ob_dir, 'observation.png'))
        
        samples = diffpir_sample(net, forward_operator, observation, config.sampling.num_steps, config.model.sigma_max, 
                                 config.model.sigma_min, config.sampling.rho, forward_operator.sigma_noise, config.sampling.lamb,
                                 config.sampling.xi, num_samples=config.sampling.num_samples, no_grad=config.sampling.no_grad,
                                 num_queries=config.sampling.num_queries, mu=config.sampling.mu, batch_size=config.sampling.batch_size)
        
        # figs_path = os.path.join(gen_dir, 'samples.png')
        # plot_images((samples * 127.5 + 128).clip(0, 255).to(torch.uint8), figs_path)
        imgs = (samples * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu().numpy()
        for i, img in enumerate(imgs):
            save_image(img, os.path.join(gen_dir, f'sample-{i}.png'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ffhq256_diffpir_inpaint.yml')
    parser.add_argument('--id_list', type=parse_int_list, default='0-9')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--x_initial', type=str, default='none')
    args = parser.parse_args()
    subprocess(args)