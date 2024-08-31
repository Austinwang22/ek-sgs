import os
import pickle
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser

import torch

from utils.img_utils import load_raw_image, save_image, normalize_img, unnormalize_img

from dps.precond import VPPrecond

from piq import LPIPS


def main(args):
    # set random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load config
    config = OmegaConf.load(args.config)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    ref_path_list = [os.path.join(config.ref_dir, f) for f in os.listdir(config.ref_dir) if f.endswith('.png')]
    num_samples = len(ref_path_list)

    # psnr_fn = PSNRMetric(max_val=1)
    lpips_fn = LPIPS(replace_pooling=True, reduction="none")
    # ssim_fn = SSIMMetric(spatial_dims=2)
    

    eval_batch = args.eval_batch
    psnr_all, lpips_all, ssim_all = [],[],[]
    for i, ref_path in enumerate(ref_path_list):
        # set up directory for each reference image
        print('Evaluating on run {}'.format(i))
        filename = os.path.basename(ref_path).replace('.png', '')
        base_dir = os.path.join(outdir, f'{filename}')
        os.makedirs(base_dir, exist_ok=True)
        gt_dir = os.path.join(base_dir, 'ground_truth')
        os.makedirs(gt_dir, exist_ok=True)
        ob_dir = os.path.join(base_dir, 'observations')
        os.makedirs(ob_dir, exist_ok=True)
        gen_dir = os.path.join(base_dir, 'generated')
        os.makedirs(gen_dir, exist_ok=True)

        # load the reference image
        # logger.info(f'Prcessing {ref_path}, ({i+1}/{num_samples})')
        ref_img = load_raw_image(ref_path)  # (C, H, W), np.uint8
        ref_img = normalize_img(ref_img)    # (C, H, W), np.float32
        ref_img = torch.from_numpy(ref_img).to(torch.float32).to(device)    # (C, H, W), torch.float32
        ref_img = ref_img.repeat(eval_batch,1,1,1)
        ref_img = ref_img/2 + 0.5
        psnr, lpips, ssim = [],[],[]
        for j in tqdm(range(args.num_samples//eval_batch)):
            gen_img = [torch.from_numpy(normalize_img(load_raw_image(os.path.join(gen_dir, "sample-{}.png".format(k))))).to(torch.float32).to(device) for k in range(j*eval_batch, (j+1)*eval_batch)]
            gen_img = torch.stack(gen_img) /2 + 0.5
            # psnr.append(psnr_fn(ref_img,gen_img).squeeze(-1))
            lpips.append(lpips_fn(ref_img,gen_img).squeeze(-1))
            # ssim.append(ssim_fn(ref_img,gen_img).squeeze(-1))
            # psnr.append(psnr_fn(ref_img,gen_img))
            # lpips.append(lpips_fn(ref_img,gen_img))
            # ssim.append(ssim_fn(ref_img,gen_img))
            # print(psnr)
            # print(lpips)
            # print(ssim)
        # psnr_all.append(torch.stack(psnr))
        lpips_all.append(torch.stack(lpips))
        # ssim_all.append(torch.stack(ssim))
        # psnr_all.append((psnr))
        # lpips_all.append((lpips))
        # ssim_all.append((ssim))
        
    # psnr_all = torch.stack(psnr_all, dim=0)
    lpips_all = torch.stack(lpips_all, dim=0)
    # lpips_all = psnr_all
    # ssim_all = torch.stack(ssim_all, dim=0)
    # print(psnr_all)
    # print(psnr_all)
    # print("Average PSNR:{:.3f} ({:.3f}), LPIPS:{:.3f} ({:.3f}), SSIM:{:.3f} ({:.3f})"\
    #       .format(psnr_all.mean().item(), psnr_all.mean(-1).std().item(), lpips_all.mean().item(), \
    #               lpips_all.mean(-1).std().item(), ssim_all.mean().item(), ssim_all.mean(-1).std().item()))
    # print("Best PSNR:{:.3f} ({:.3f}), LPIPS:{:.3f} ({:.3f}), SSIM:{:.3f} ({:.3f})"\
    #       .format(psnr_all.max().mean().item(), psnr_all.std().item(), lpips_all.min().mean().item(), \
    #               lpips_all.std().item(), ssim_all.max().mean().item(), ssim_all.std().item()))
    print(f'Average LPIPS: {lpips_all.mean().item()}')

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = ArgumentParser()
    parser.add_argument("--outdir", type=str, default="exp/ffhq256-inpaint")
    parser.add_argument('--config', type=str, default='configs/ffhq256_inpaint.yml')
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--eval_batch", type=int, default=16)
    args = parser.parse_args()
    main(args)