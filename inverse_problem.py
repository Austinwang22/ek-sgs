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
from diffpir import *
from helper import create_step_scheduler

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
        observation = forward_operator(ref_img)   # (C, H, W), torch.float32
        obs_np = unnormalize_img(observation.detach().cpu().numpy())    # (C, H, W), np.uint8
        save_image(obs_np, os.path.join(ob_dir, 'observation.png'))
        
        if args.x_initial != 'none':
            x_initial = torch.load(args.x_initial).to(device)
        else:
            img_size = config.mask_opt.image_size
            # x_initial = torch.randn((config.sampling.num_samples, 3, img_size, img_size), device=device) * config.sampling.rho_max
            x_initial = torch.randn((config.sampling.init_ensemble, 3, img_size, img_size), device=device) * config.sampling.rho_max
        
        rho_schedule = get_exp_decay_rho(config.sampling.rho_min, config.sampling.rho_max, 
                                         config.sampling.decay_rate, config.sampling.num_steps)
        
        # torch.cuda.memory._record_memory_history()
        
        ensemble_scheduler = create_step_scheduler(config.sampling.init_ensemble, config.sampling.growth_rate,
                                                   config.sampling.scheduler_steps, config.sampling.num_steps)
        
        samples = en_sgs_sampler(net, forward_operator, observation, forward_operator.sigma_noise, x_initial,
                                 config.sampling.num_steps, rho_schedule, config.sampling.batch_size, 
                                 config.sampling.prior_steps, config.sampling.l_steps, config.sampling.scale,
                                 config.sampling.resample, config.sampling.approx_cov, prog_dir, config.log.vis_cov,
                                 ensemble_scheduler)
        
        imgs = (samples * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu().numpy()
        for i, img in enumerate(imgs):
            save_image(img, os.path.join(gen_dir, f'sample-{i}.png'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ffhq64_inpaint.yml')
    parser.add_argument('--id_list', type=parse_int_list, default='0-9')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--x_initial', type=str, default='none')
    args = parser.parse_args()
    subprocess(args)