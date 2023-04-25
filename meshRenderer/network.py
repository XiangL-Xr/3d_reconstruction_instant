# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-03-21

import torch
import torch.nn as nn
import torch.nn.functional as F

from .renderer import MeshRenderer


def get_encoder(encoding, level_dim=2, desired_resolution=2048, interpolation='linear', **kwargs):
    cfg = {
        "input_dim":  3,
        "multires":   6,     # freq
        "degree":     4,     # SH
        "num_levels": 16,
        "base_resolution": 16,
        "log2_hashmap_size": 19,
        "align_corners": False
    }
    if encoding == 'None':
        return lambda x, **kwargs: x, cfg["input_dim"]
    
    elif encoding == 'hashgrid':
        from .gridencoder import GridEncoder
        encoder = GridEncoder(
            input_dim = cfg["input_dim"],
            num_levels = cfg["num_levels"],
            level_dim = level_dim,
            base_resolution = cfg["base_resolution"],
            log2_hashmap_size = cfg["log2_hashmap_size"],
            desired_resolution= desired_resolution, 
            gridtype= 'hash',
            align_corners = cfg['align_corners'],
            interpolation = interpolation
        )
    
    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, hashgrid]')
    
    return encoder, encoder.output_dim

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        
        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, 
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))
        
        self.net = nn.ModuleList(net)
        
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)

        return x
  

class meshNetwork(MeshRenderer):
    def __init__(self, args, specular_dim=3,):
        super().__init__(args)
    
        ## sigma and feature network
        # self.encoder, self.in_dim_density = get_encoder('hasgrid', level_dim=1, desired_resolution=2048*self.bound, interpolation='smoothstep')
        self.encoder_color, self.in_dim_color = get_encoder('hashgrid', level_dim=2, desired_resolution=2048*self.bound, interpolation='linear')
        # self.sigma_net = MLP(self.in_dim_density, 1, 32, 2, bias=False)
        
        ## color network
        self.encoder_dir, self.in_dim_dir = get_encoder('None')
        self.color_net = MLP(self.in_dim_color + self.ind_dim, 3 + specular_dim, 64, 3, bias=False)
        self.specular_net = MLP(specular_dim + self.in_dim_dir, 3, 32, 2, bias=False)
    
    def forward(self, x, d, c=None, shading='full'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # c: [1/N, individual_dim]
        
        # sigma = self.density(x)['sigma']
        color, specular = self.rgb(x, d, c, shading)
        
        return color, specular
    
    def geo_feat(self, x, c=None):
        h = self.encoder_color(x, bound=self.bound)
        if c is not None:
            h = torch.cat([h, c.repeat(x.shape[0], 1) if c.shape[0] == 1 else c], dim=-1)
        h = self.color_net(h)
        geo_feat = torch.sigmoid(h)
        
        return geo_feat
    
    
    def rgb(self, x, d, c=None, shading='full'):
        # color
        geo_feat = self.geo_feat(x, c)
        diffuse = geo_feat[..., :3]
        
        d = self.encoder_dir(d)
        
        specular = self.specular_net(torch.cat([d, geo_feat[..., 3:]], dim=-1))
        specular = torch.sigmoid(specular)
        color = (specular + diffuse).clamp(0, 1)      # specular + albedo
    
        return color, specular
    
    
    def get_params(self, lr):
        
        # params = super().get_params(lr)
        params = []
        
        params.extend([
            # {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.encoder_color.parameters(), 'lr': lr},
            # {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
            {'params': self.specular_net.parameters(), 'lr': lr},
        ])
        
        return params
