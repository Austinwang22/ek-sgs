import torch
from sampling import ode_sampler
from helper import plot_images
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def get_exp_decay_rho(rho_min, rho_max, decay_rate, N):
    rhos = torch.zeros(N)
    rhos = torch.pow(decay_rate, torch.arange(0, N)) * rho_max
    rhos = torch.maximum(rhos, torch.ones_like(rhos) * rho_min)
    return rhos

def get_cov_diag(z):
    particles = z.reshape(len(z), -1)
    diff = particles - particles.mean(dim=0, keepdim=True)
    diff_sq = diff ** 2
    diag = diff_sq.sum(dim=0) / len(particles)
    return diag
    # return torch.diag(diag)
    
def visualize_cov(cov, dim, filename):
    img = cov[:dim, :dim].cpu()
    plt.imshow(img, cmap='Blues', interpolation='none')
    plt.colorbar()
    plt.title('Covariance')
    plt.savefig(filename)
    plt.close()

@torch.no_grad()
def ek_update(forward_operator, y, std_y, x, x_clean, scale):
    
    N, *spatial = x.shape
    
    preds = forward_operator.forward(x_clean)
    xs_diff = x - x.mean(dim=0, keepdim=True)
    pred_err = (preds - y)
    pred_diff = preds - preds.mean(dim=0, keepdim=True)
        
    coef = (
        torch.matmul(
            pred_err.reshape(pred_err.shape[0], -1) / (std_y ** 2),
            pred_diff.reshape(pred_diff.shape[0], -1).T,
        )
        / len(x)
    )
    
    dx = (coef @ xs_diff.reshape(N, -1)).reshape(N, *spatial)
    lr = scale / torch.linalg.matrix_norm(coef)
    
    return dx, lr

@torch.no_grad()
def ll_step(forward_operator, y, std_y, particles, rho, num_l_steps, scale, 
            resample=True, approx_cov=False, cov_filepath=None, particles_io=None):
    x = particles
    z_next = particles.clone()
    
    J, *spatial = particles.shape
    
    pbar = tqdm(range(num_l_steps))
    for _ in pbar:
        
        z_diff = (z_next - z_next.mean(dim=0, keepdim=True)).reshape(J, -1)
        
        if approx_cov:
            cov_diag = get_cov_diag(z_next)
            dz_reg = ((x - z_next).reshape(J, -1) * cov_diag).reshape(J, *spatial) / (rho ** 2)  
        else:
            cov = z_diff.T @ z_diff / len(z_diff)
            dz_reg = ((x - z_next).reshape(J, -1) @ cov).reshape(J, *spatial) / (rho ** 2)  
                
        dz_ll, lr_ll = ek_update(forward_operator, y, std_y, z_next, z_next, 1.0)        
        
        lr = scale / torch.linalg.matrix_norm((dz_ll + dz_reg).reshape(J, -1))
        
        z_next -= dz_ll * lr
        z_next += dz_reg * lr
        
        eps = torch.randn_like(z_next).reshape(J, -1)
        if approx_cov:
            cov_sqrt = torch.sqrt(cov_diag)
            noise = (eps * cov_sqrt).reshape(J, *spatial) * torch.sqrt(2 * lr)
        else:
            cov_sqrt = torch.linalg.cholesky((cov) + 0.01 * torch.eye(len(cov), device=z_next.device))
            noise = (eps @ cov_sqrt).reshape(J, *spatial) * torch.sqrt(2 * lr)
            
        z_next += noise
            
        avg_err = (forward_operator.forward(z_next) - y).mean()
        pbar.set_description(f'Avg. error: {avg_err.item()}')
    
    if cov_filepath != None:
        z_cov = (z_next - z_next.mean(dim=0, keepdim=True)).reshape(J, -1).T @ (z_next - z_next.mean(dim=0, keepdim=True)).reshape(J, -1) / len(z_next)
        visualize_cov(z_cov, 256, cov_filepath)
    z_cov_diag = get_cov_diag(z_next)
    print(f'Approximate variance: {z_cov_diag.mean()}')
    if resample:
        noise_diff = min(rho - torch.sqrt(z_cov_diag.mean()), rho)
        # noise_diff = rho
    else:
        noise_diff = 0
        
    if particles_io is None:
        return z_next + torch.randn_like(z_next) * noise_diff
    else:
        p_i, p_o = particles_io
        ratio = p_o // p_i
        repeated = z_next.tile((ratio, 1, 1, 1))
        return repeated + torch.randn_like(repeated) * noise_diff
    # return z_next + torch.randn_like(z_next) * noise_diff

@torch.no_grad()
def en_sgs_sampler(model, forward_operator, y, std_y, x_initial, N, rho_schedule, batch_size, 
                   num_prior_steps=50, num_l_steps=500, scale=1., resample=True, approx_cov=False,
                   progdir=None, vis_cov=False, ensemble_schedule=None):
    
    model.eval()
        
    x = x_initial.to(torch.float64)
    
    if ensemble_schedule is None:
        ensemble_schedule = [len(x_initial) for _ in range(N + 1)]
        
    for i in range(N):
        rho_cur = rho_schedule[i]
        particles_cur = ensemble_schedule[i]
        particles_next = ensemble_schedule[i + 1]
        
        print(f'Iteration {i}, rho = {rho_cur}, ensemble size = {len(x)}\n')
        
        # Likelihood step
        if vis_cov:
            cov_filepath = os.path.join(progdir, f'cov_{i}.png')
        else:
            cov_filepath = None
        z = ll_step(forward_operator, y, std_y, x, rho_cur, num_l_steps, scale=scale, 
                    resample=resample, approx_cov=approx_cov, cov_filepath=cov_filepath,
                    particles_io=(particles_cur, particles_next))
                                
        # Prior Step
        x = torch.zeros_like(z)
        num_batches = len(x) // batch_size
        pbar = tqdm(range(num_batches))
        for b in pbar:
            start = b * batch_size
            end = (b + 1) * batch_size
            ode_steps = min(num_prior_steps, 1 + (N - i))
            x[start : end] = ode_sampler(model, z[start : end], ode_steps, sigma_start=rho_cur)
        
        if progdir != None:
            plot_images((x[0 : 16] * 127.5 + 128).clip(0, 255).to(torch.uint8), os.path.join(progdir, f'x_{i}.png'))
            plot_images((z[0 : 16] * 127.5 + 128).clip(0, 255).to(torch.uint8), os.path.join(progdir, f'z_{i}.png'))
            # torch.save(z, os.path.join(progdir, f'z_{i}.pt'))
            # torch.save(x, os.path.join(progdir, f'x_{i}.pt'))
        
    return x