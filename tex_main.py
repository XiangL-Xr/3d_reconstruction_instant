# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-08

import argparse
import os
import torch

from datasets.tex_dataset import MeshDatasetBase
from meshRenderer.network import meshNetwork
from meshRenderer.provider import meshDataset
from meshRenderer.renderer import Trainer
from meshRenderer.utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data')
    parser.add_argument('--case', default='bitong_138')
    parser.add_argument('--img_wh', default=[1024, 1024])
    parser.add_argument('--workspace', type=str, default='./exp_workspace')
    parser.add_argument('--gpu', default=1)
    parser.add_argument('--iters', default=5000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--refine_steps_ratio", type=float, action="append", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    parser.add_argument('--bound', type=float, default=1,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.8, 
                        help="scale camera location into box[-bound, bound]^3, -1 means automatically determine based on camera poses..")
    parser.add_argument('--dt_gamma', type=float, default=0, 
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--max_steps', type=int, default=1024, 
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--background', type=str, default='random', choices=['white', 'random'], help="training background mode")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--pos_gradient_boost', type=float, default=1, help="nvdiffrast option")
    
    ## mesh render stage
    parser.add_argument('--fp16', default=True)
    parser.add_argument('--preload', default=True)
    parser.add_argument('--ssaa', type=int, default=2, help="super sampling anti-aliasing ratio")
    parser.add_argument('--texture_size', type=int, default=2048, help="exported texture resolution")

    args = parser.parse_args()

    # set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    
    ## convert ratio to steps
    args.refine_steps = [int(round(x * args.iters)) for x in args.refine_steps_ratio]
    
    ## generate seed
    seed_everything(args.seed)
    
    ## load model
    model = meshNetwork(args)
    
    criterion = torch.nn.SmoothL1Loss(reduction='none')
    optimizer = lambda model: torch.optim.Adam(model.get_params(args.lr), eps=1e-15)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = MeshDatasetBase(args, device=device)
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=1, 
                                               collate_fn=train_data.collate_fn,
                                               shuffle=True, 
                                               num_workers=0)
    # train_loader = meshDataset(args, device=device).dataloader()
    
    max_epoch = np.ceil(args.iters / len(train_loader)).astype(np.int32)
    save_interval = max(1, max_epoch // 50)
    # eval_interval = max(1, max_epoch // 10)
    print(f'[INFO] max_epoch {max_epoch}, save every {save_interval}.')
    
    ## scheduler and trainer
    scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / args.iters, 1))
    trainer = Trainer(args, model,
                      device = device,
                      workspace = args.workspace,
                      optimizer = optimizer,
                      criterion = criterion,
                      fp16 = args.fp16,
                      lr_scheduler = scheduler,
                      scheduler_update_every_step = True,
                      save_interval = save_interval)
    
    
    ## optimize material
    trainer.metrics = [PSNRMeter(),]
    trainer.train(train_loader, max_epoch)
   
    ## saved output mesh and texture
    save_path = os.path.join(args.workspace, args.case, 'mesh_export')
    trainer.m_export(save_path=save_path, resolution=args.texture_size)


if __name__ == '__main__':
    main()
