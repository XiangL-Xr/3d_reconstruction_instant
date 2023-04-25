# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-07

import os
import torch
import numpy as np
import imageio
import nvdiffrast.torch as dr

## Define vector operations ====================================================================
def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps))                           # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)


## Image scaling ===============================================================================
def scale_img_hwc(x : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    # Trying to magnify image in one dimension and minify in the other
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1])
    y = x.permute(0, 3, 1, 2)                                        # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:                # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:                                                            # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    
    return y.permute(0, 2, 3, 1).contiguous()                        # NCHW -> NHWC

def avg_pool_nhwc(x  : torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2)                                        # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    
    return y.permute(0, 2, 3, 1).contiguous()                        # NCHW -> NHWC


## Image save / load helper ====================================================================
def save_image(fn, x : np.ndarray):
    try:
        if os.path.splitext(fn)[1] == ".png":
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3) # Low compression for faster saving
        else:
            imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))
    except:
        print("WARNING: FAILED to save image %s" % fn)

def save_image_raw(fn, x : np.ndarray):
    try:
        imageio.imwrite(fn, x)
    except:
        print("WARNING: FAILED to save image %s" % fn)

def load_image(fn) -> np.ndarray:
    img = load_image_raw(fn)
    if img.dtype == np.float32:   # HDR image
        return img
    else:                         # LDR image
        return img.astype(np.float32) / 255

def load_image_raw(fn) -> np.ndarray:
    return imageio.imread(fn)


## sRGB color transforms ====================================================================
def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    
    return out

## Cubmap utility functions =================================================================
def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    
    return torch.stack((rx, ry, rz), dim=-1)


def time2text(x):
    if x > 3600:
        return "%.2f hours" % (x / 3600)
    elif x > 60:
        return "%.2f mins" % (x / 60)
    else:
        return "%.2f seconds" % x
    
def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)
        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
        
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'),
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'))
    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi     = torch.sin(gx * np.pi), torch.cos(gx * np.pi)
    
    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)
    
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]

def mse2psnr(mse):
    return -10. / np.log(10.) * np.log(mse)

def psnr2mse(psnr):
    return np.exp(-0.1 * np.log(10.) * psnr)