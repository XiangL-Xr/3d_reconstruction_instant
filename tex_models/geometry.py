# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-07

import torch
import tex_models.mesh as mesh
import trimesh
import xatlas
import numpy as np

from . import mlptexture
from .render import render


class Tex_Geometry():
    def __init__(self, base_mesh_path):
        super(Tex_Geometry, self).__init__()
        
        self.init_geom = trimesh.load(base_mesh_path)
        self.verts = torch.tensor(self.init_geom.vertices, dtype=torch.float32, device='cuda')
        self.faces = torch.tensor(self.init_geom.faces, dtype=torch.long, device='cuda')
        print("=> Base mesh has %d triangles and %d vertices." % (self.faces.shape[0], self.verts.shape[0]))
        
        # Params setting
        self.texture_res = [1024, 1024]
        self.kd_min      = [0.03, 0.03, 0.03]
        self.kd_max      = [0.80, 0.80, 0.80]
        self.ks_min      = [0, 0.08, 0]             # Limits for ks
        self.ks_max      = [0.0, 1.0, 1.0]
        self.nrm_min     = [-1.0, -1.0,  0.0]       # Limits for normal map
        self.nrm_max     = [ 1.0,  1.0,  1.0]
        
        
    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
       
    def initial_material(self):
        self.kd_min, self.kd_max   = torch.tensor(self.kd_min, dtype=torch.float32, device='cuda'),  torch.tensor(self.kd_max, dtype=torch.float32, device='cuda')
        self.ks_min, self.ks_max   = torch.tensor(self.ks_min, dtype=torch.float32, device='cuda'),  torch.tensor(self.ks_max, dtype=torch.float32, device='cuda')
        self.nrm_min, self.nrm_max = torch.tensor(self.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(self.nrm_max, dtype=torch.float32, device='cuda')
        
        mlp_min = torch.cat((self.kd_min, self.ks_min, self.nrm_min), dim=0)
        mlp_max = torch.cat((self.kd_max, self.ks_max, self.nrm_max), dim=0)
        
        mlp_map_out = mlptexture.MLPTexture3D(self.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        print(mlp_map_out)
        
        opt_mat = mesh.Material({'kd_ks_normal' : mlp_map_out})
        opt_mat['bsdf'] = 'pbr'
        self.mat = opt_mat

    ## Define UV-map by xatlas algorithm
    @torch.no_grad()
    def xatlas_uvmap(self):
        print('=> start mesh to xatlas uvmap ... ')
        v_pos = self.verts.detach().cpu().numpy()
        t_pos_idx = self.faces.detach().cpu().numpy()
        
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
        
        uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
        uvs_idx = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')
        self.uv_mesh = mesh.Mesh(self.verts, self.faces, v_tex=uvs, t_tex_idx=uvs_idx, material=self.mat)
    
    def get_TexMat(self, glctx):
        mask, kd, ks, normal = render.render_uv(glctx, self.uv_mesh, self.texture_res, self.mat['kd_ks_normal'])
        self.uv_mesh.material = mesh.Material({
                'bsdf'    : self.mat['bsdf'],
                'kd'      : mlptexture.Texture2D(kd, min_max=[self.kd_min, self.kd_max]),
                'ks'      : mlptexture.Texture2D(ks, min_max=[self.ks_min, self.ks_max]),
                'normal'  : mlptexture.Texture2D(normal, min_max=[self.nrm_min, self.nrm_max])   
        })
        
        return self.uv_mesh
        