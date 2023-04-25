# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-03-21

import os
import random
import numpy as np
import torch

from packaging import version as pver

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def create_dodecahedron_cameras(radius=1, center=np.array([0, 0, 0])):
    vertices = np.array([
        -0.57735,  -0.57735,  0.57735,
        0.934172,  0.356822,  0,
        0.934172,  -0.356822,  0,
        -0.934172,  0.356822,  0,
        -0.934172,  -0.356822,  0,
        0,  0.934172,  0.356822,
        0,  0.934172,  -0.356822,
        0.356822,  0,  -0.934172,
        -0.356822,  0,  -0.934172,
        0,  -0.934172,  -0.356822,
        0,  -0.934172,  0.356822,
        0.356822,  0,  0.934172,
        -0.356822,  0,  0.934172,
        0.57735,  0.57735,  -0.57735,
        0.57735,  0.57735,  0.57735,
        -0.57735,  0.57735,  -0.57735,
        -0.57735,  0.57735,  0.57735,
        0.57735,  -0.57735,  -0.57735,
        0.57735,  -0.57735,  0.57735,
        -0.57735,  -0.57735,  -0.57735,
        ]).reshape((-1,3), order="C")
    
    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center
    
    # forward is simple, notice that it is in fact the inversion of camera direction!
    forward_vector = normalize(vertices - center)
    # pick a temp up_vector, usually [0, 1, 0]
    up_vector = np.array([0, 1, 0], dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    # cross(up, forward) --> right
    right_vector = normalize(np.cross(up_vector, forward_vector, axis=1))
    # rectify up_vector, by cross(forward, right) --> up
    up_vector = normalize(np.cross(forward_vector, right_vector, axis=1))
    
    ## construct c2w
    poses = np.eye(4, dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=1)
    poses[:, :3, 3] = vertices
    
    return poses

@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''
    device = poses.device
    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]
    
    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5
    
    results = {}
    inds = torch.arange(H*W, device=device)
    
    zs = -torch.ones_like(i)    # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy
    directions = torch.stack((xs, ys, zs), dim=-1)    # [N, 3]
    rays_d = (directions.unsqueeze(1) @ poses[:, :3, :3].transpose(-1, -2)).squeeze(1)
    rays_o = poses[:, :3, 3].expand_as(rays_d)        # [N, 3]
    
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    
    return results

@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)

@torch.jit.script 
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1])
    y = x.permute(0, 3, 1, 2)      # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:     # Minification, pervious size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    
    return y.permute(0, 2, 3, 1).contiguous()             # NCHW -> NHWC


def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]


def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0
    
    def clear(self):
        self.V = 0
        self.N = 0
    
    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)
        
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        self.V += psnr
        self.N += 1
        
        return psnr
    
    def measure(self):
        return self.V / self.N
    
    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)
        
    def report(self):
        return (f'PSNR = {self.measure():.6f}')