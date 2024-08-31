import torch
import numpy as np
import dnnlib
import pickle
import os

import yaml
from helper import dict2namespace, plot_images
from argparse import ArgumentParser

import tqdm

@torch.no_grad()
def ode_sampler(
    net,
    x_initial,
    num_steps=18,
    sigma_start=80.0,
    sigma_eps=0.002,
    rho=7,
    printing=False
):
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=x_initial.device)
    t_steps = (sigma_start ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_eps ** (1 / rho) - sigma_start ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_next = x_initial.to(torch.float64)
    for i in range(num_steps):
        if printing:
            print(f'Iteration {i + 1}/{num_steps}')
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        x_cur = x_next

        denoised = net(x_cur, t_cur).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

    return x_next


def heun_solver(
    net, x_initial,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    class_labels = None
    if net.label_dim:
        print(net.label_dim)
        class_labels = torch.eye(net.label_dim, device=x_initial.device)[torch.randint(net.label_dim, size=[len(x_initial)], device=x_initial.device)]
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=x_initial.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = x_initial.to(torch.float64)
    for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
    return x_next

def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    
    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    ckpt_dir = os.path.join(basedir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    fig_dir = os.path.join(basedir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    progdir = os.path.join(basedir, 'prog')
    os.makedirs(progdir, exist_ok=True)
    
    device = torch.device(config.model.device)
    
    with dnnlib.util.open_url(config.model.url) as f:
        model = pickle.load(f)['ema'].to(device)
            
    num_steps = config.sampling.num_steps
    img_size = config.model.img_size
    latents = torch.randn((config.sampling.num_samples, 3, img_size, img_size), device=device).to(torch.float64) * config.model.sigma_max
    samples = ode_sampler(model, latents, num_steps=num_steps, sigma_start=config.model.sigma_max, 
                          sigma_eps=config.model.sigma_min, printing=True)
    # samples = heun_solver(model, latents, num_steps=num_steps, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, 
    #                       rho=config.sampling.rho)
    
    filepath = os.path.join(fig_dir, 'samples.png')
    plot_images((samples * 127.5 + 128).clip(0, 255).to(torch.uint8), filepath)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ffhq64_euler.yml')
    args = parser.parse_args()
    subprocess(args)
