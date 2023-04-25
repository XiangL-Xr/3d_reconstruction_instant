# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-07

import os
import torch
import torch.nn as nn
import numpy as np
import nvdiffrast.torch as dr

from torchinfo import summary
import tinycudann as tcnn
from . import utils

class MLPTexture3D(nn.Module):
    def __init__(self, AABB, channels, min_max=None):
        super(MLPTexture3D, self).__init__()
        self.AABB = AABB
        self.channels = channels
        self.min_max = min_max
        self.internal_dims = 32
        self.hidden = 2
        
        ## setup positional encoding
        desired_resolution = 4096
        base_grid_resolution = 16
        gradient_scaling = 128.0
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels - 1))
        
        enc_cfg = {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale": per_level_scale
        }
        
        self.encoder = tcnn.Encoding(3, enc_cfg)
        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling))
        
        mlp_cfg = {
            "n_input_dims": self.encoder.n_output_dims,
            "n_output_dims": self.channels,
            "n_hidden_layers": self.hidden,
            "n_neurons": self.internal_dims
        }
        
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print("=> Encoder output: %d dims" % (self.encoder.n_output_dims))
        
        # print('=> print mlptexture model structure ...')
        # print(summary(self.net, input_size=(4, 32)))
        # return self.net
    
    # Sample texture at a given location
    def sample(self, texc):
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        
        p_enc = self.encoder(_texc.contiguous())
        out = self.net.forward(p_enc)
        
        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, ...] - self.min_max[0][None, ...]) + self.min_max[0][None, :]
        
        return out.view(*texc.shape[:-1], self.channels)              ## Remap to [n, h, w, c]
    
    
class _MLP(nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        self.cfg = cfg
        self._mlp_net()
        
    def _mlp_net(self):                                               ## Setup MLP NET
        net = (nn.Linear(self.cfg['n_input_dims'], self.cfg['n_neurons'], bias=False), nn.ReLU())
        for i in range(self.cfg['n_hidden_layers'] - 1):
            net = net + (nn.Linear(self.cfg['n_neurons'], self.cfg['n_neurons'], bias=False), nn.ReLU())
        net = net + (nn.Linear(self.cfg['n_neurons'], self.cfg['n_output_dims'], bias=False), )
        
        self.net = nn.Sequential(*net).cuda()
        self.net.apply(self._init_weights)
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale))
    
    def forward(self, x):
        return self.net(x.to(torch.float32))
    
    @staticmethod
    def _init_weights(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)   


## Utility functions for loading / storing a texture ================================================================================
class Texture2D(nn.Module):
    # Initializes a texture from image data
    # Input can be constant value(1D array) or texture(3D array) or mip hierarchy(list of 3d arrays)
    def __init__(self, init, min_max=None):
        super(Texture2D, self).__init__()
        if isinstance(init, np.ndarray):
            init = torch.tensor(init, dtype=torch.float32, device='cuda')
        elif isinstance(init, list) and len(init) == 1:
            init = init[0]
        
        if isinstance(init, list):
            self.data = list(torch.nn.Parameter(mip.clone().detach(), requires_grad=True) for mip in init)
        elif len(init.shape) == 4:
            self.data = torch.nn.Parameter(init.clone().detach(), requires_grad=True)
        elif len(init.shape) == 3:
            self.data = torch.nn.Parameter(init[None, ...].clone().detach(), requires_grad=True)
        elif len(init.shape) == 1:
            self.data = torch.nn.Parameter(init[None, None, None, :].clone().detach(), requires_grad=True)  # Convert constant to 1x1 tensor
        else:
            assert False, "Invalid texture object !"
        
        self.min_max = min_max
    
    # Filtered(trilinear) sample texture at a given location
    def sample(self, texc, texc_deriv, filter_mode='linear-mipmap-linear'):
        if isinstance(self.data, list):
            out = dr.texture(self.data[0], texc, texc_deriv, mip=self.data[1:], filter_mode=filter_mode)
        else:
            if self.data.shape[1] > 1 and self.data.shape[2] > 1:
                mips = [self.data]
                while mips[-1].shape[1] > 1 and mips[-1].shape[2] > 1:
                    mips += [texture2d_mip.apply(mips[-1])]
                out = dr.texture(mips[0], texc, texc_deriv, mip=mips[1:], filter_mode=filter_mode)
            else:
                out = dr.texture(self.data, texc, texc_deriv, filter_mode=filter_mode)
        
        return out       
    
    def getRes(self):
        return self.getMips()[0].shape[1:3]
    
    def getChannels(self):
        return self.getMips()[0].shape[3]
    
    def getMips(self):
        if isinstance(self.data, list):
            return self.data
        else:
            return [self.data]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        if self.min_max is not None:
            for mip in self.getMips():
                for i in range(mip.shape[-1]):
                    mip[..., i].clamp_(min=self.min_max[0][i], max=self.min_max[1][i])
    
    def normalize_(self):
        with torch.no_grad():
            for mip in self.getMips():
                mip = utils.safe_normalize(mip)

## Smooth pooling / mip computation with linear gradient upscaling
class texture2d_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, texture):
        return utils.avg_pool_nhwc(texture, (2, 2))

    @staticmethod
    def backward(ctx, dout):
        gy, gx = torch.meshgrid(torch.linspace(0.0 + 0.25 / dout.shape[1], 1.0 - 0.25 / dout.shape[1], dout.shape[1] * 2, device='cuda'),
                                torch.linspace(0.0 + 0.25 / dout.shape[2], 1.0 - 0.25 / dout.shape[2], dout.shape[2] * 2, device='cuda'))
        uv = torch.stack((gx, gy), dim=-1)
        
        return dr.texture(dout * 0.25, uv[None, ...].contiguous(), filter_mode='linear', boundary_mode='clamp')


def save_texture2D(fn, tex, lambda_fn=None):
    if isinstance(tex.data, list):
        for i, mip in enumerate(tex.data):
            _save_mip2D(fn, mip[0, ...], i, lambda_fn)
    else:
        _save_mip2D(fn, tex.data[0, ...], None, lambda_fn)

def _save_mip2D(fn, mip, mipidx, lambda_fn):
    if lambda_fn is not None:
        data = lambda_fn(mip).detach().cpu().numpy()
    else:
        data = mip.detach().cpu().numpy()
    
    if mipidx is None:
        utils.save_image(fn, data)
    else:
        base, ext = os.path.splitext(fn)
        utils.save_image(base + ("_%d" % mipidx) + ext, data)

def srgb_to_rgb(texture):
    return Texture2D(list(utils.srgb_to_rgb(mip) for mip in texture.getMips()))

def rgb_to_srgb(texture):
    return Texture2D(list(utils.rgb_to_srgb(mip) for mip in texture.getMips()))