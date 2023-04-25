# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-07

import os
import torch
from . import utils
from . import mlptexture

## Define Base Mesh =====================================================================================
class Mesh:
    def __init__(self, v_pos=None, t_pos_idx=None, v_tex=None, t_tex_idx=None, v_nrm=None, t_nrm_idx=None, v_tng=None, t_tng_idx=None, material=None, base=None):
        self.v_pos = v_pos
        self.v_tex = v_tex
        self.v_nrm = v_nrm
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx
        self.t_tex_idx = t_tex_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tng_idx = t_tng_idx
        self.material = material
        
        if base is not None:
            self.copy_none(base)
        
    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        
        if self.v_tex is None:
            self.v_tex = other.v_tex
            
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
            
        if self.v_tng is None:
            self.v_tng = other.v_tng
        
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
            
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        
        if self.material is None:
            self.material = other.material
    
    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone().detach()
        
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone().detach()
        
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone().detach()
        
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone().detach()
            
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone().detach()
        
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone().detach()
        
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone().detach()
        
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone().detach()
        
        return out

## Simple smooth vertex normal computation ===========================================================
def auto_normals(imesh):
    i0 = imesh.t_pos_idx[:, 0]
    i1 = imesh.t_pos_idx[:, 1]
    i2 = imesh.t_pos_idx[:, 2]
    
    v0 = imesh.v_pos[i0, :]
    v1 = imesh.v_pos[i1, :]
    v2 = imesh.v_pos[i2, :]
    
    face_normals = torch.cross(v1-v0, v2-v0)
    # Splat face normals to vertices
    v_nrm = torch.zeros_like(imesh.v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)
    
    # Normalize, replace zero normals with some default value
    v_nrm = torch.where(utils.dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_nrm = utils.safe_normalize(v_nrm)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))
    
    return Mesh(v_nrm=v_nrm, t_nrm_idx=imesh.t_pos_idx, base=imesh)


## Compute tangent space from texture map coordinates ================================================
def compute_tangents(imesh):
    vn_idx = [None] * 3
    pos    = [None] * 3
    tex    = [None] * 3
    
    for i in range(0, 3):
        pos[i] = imesh.v_pos[imesh.t_pos_idx[:, i]]
        tex[i] = imesh.v_tex[imesh.t_tex_idx[:, i]]
        vn_idx[i] = imesh.t_nrm_idx[:, i]
    
    tangents = torch.zeros_like(imesh.v_nrm)
    tansum   = torch.zeros_like(imesh.v_nrm)
    
    # compute tangent space for each triangle
    uve1 = tex[1] - tex[0]
    uve2 = tex[2] - tex[0]
    pe1  = pos[1] - pos[0]
    pe2  = pos[2] - pos[0]
    
    nom  = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
    dnom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
    
    # avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(dnom > 0.0, torch.clamp(dnom, min=1e-6), torch.clamp(dnom, max=-1e-6))
    
    # update all 3 vertices
    for i in range(0, 3):
        idx = vn_idx[i][:, None].repeat(1, 3)
        tangents.scatter_add_(0, idx, tang)                     # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(0, idx, torch.ones_like(tang))      # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum
    
    # normalize and make sure tangent is perpendicular to normal
    tangents = utils.safe_normalize(tangents)
    tangents = utils.safe_normalize(tangents - utils.dot(tangents, imesh.v_nrm) * imesh.v_nrm)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))
    
    return Mesh(v_tng=tangents, t_tng_idx=imesh.t_nrm_idx, base=imesh)



## Define Material ===================================================================================
class Material(torch.nn.Module):
    def __init__(self, mat_dict):
        super(Material, self).__init__()
        self.mat_keys = set()
        for key in mat_dict.keys():
            self.mat_keys.add(key)
            self[key] = mat_dict[key]
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        self.mat_keys.add(key)
        setattr(self, key, val)
    
    def keys(self):
        return self.mat_keys

@torch.no_grad()
def save_mtl(fn, material):
    folder = os.path.dirname(fn)
    with open(fn, "w") as ff:
        ff.write('newmtl defaultMat\n')
        if material is not None:
            ff.write('bsdf    %s\n' % material['bsdf'])
            if 'kd' in material.keys():
                ff.write('map_Kd texture_kd.png\n')
                mlptexture.save_texture2D(os.path.join(folder, 'texture_kd.png'), mlptexture.rgb_to_srgb(material['kd']))
            if 'ks' in material.keys():
                ff.write('map_Ks texture_ks.png\n')
                mlptexture.save_texture2D(os.path.join(folder, 'texture_ks.png'), material['ks'])
            if 'normal' in material.keys():
                ff.write('bump texture_n.png\n')
                mlptexture.save_texture2D(os.path.join(folder, 'texture_n.png'), material['normal'], lambda_fn=lambda x:(utils.safe_normalize(x)+1) * 0.5)
        else:
            ff.write('Kd 1 1 1\n')
            ff.write('Ks 0 0 0\n')
            ff.write('Ka 0 0 0\n')
            ff.write('Tf 1 1 1\n')
            ff.write('Ni 1\n')
            ff.write('Ns 0\n')


## Save mesh object to objfile =======================================================================
def write_obj(folder, mesh, save_material=True):
    obj_file = os.path.join(folder, 'mesh.obj')
    print("Writing mesh: ", obj_file, " ...")
    with open(obj_file, 'w') as ff:
        ff.write("mtllib mesh.mtl\n")
        ff.write("g default\n")

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None
        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None
        
        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            ff.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                ff.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))
        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                ff.write('vn {} {} {} \n'.format(v[0], v[1], v[2]))
        
        ## faces
        ff.write("s 1 \n")
        ff.write("g pMesh1 \n")
        ff.write("usemtl defaultMat \n")
        
        ## write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            ff.write("f ")
            for j in range(3):
                ff.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1),
                                        '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            ff.write("\n")
    
    if save_material:
        mtl_file = os.path.join(folder, 'mesh.mtl')
        print("Writing material: ", mtl_file)
        save_mtl(mtl_file, mesh.material)
    

## get Mesh from output material ===============================================================
class CustomMesh():
    def __init__(self, initial_guess):
        super(CustomMesh, self).__init__()
        
        self.initial_guess = initial_guess
        self.mesh          = initial_guess.clone()
        print("=> Base mesh has %d triangles and %d vertices." % (self.mesh.t_pos_idx.shape[0], self.mesh.v_pos.shape[0]))

    def getMesh(self, material):
        self.mesh.material = material
        imesh = Mesh(base=self.mesh)

        # Compute normals and tangent space
        imesh = auto_normals(imesh)
        omesh = compute_tangents(imesh)
            
        return omesh

def getMesh(geometry, material):
    geometry.material = material
    imesh = Mesh(base=geometry)

    # Compute normals and tangent space
    imesh = auto_normals(imesh)
    omesh = compute_tangents(imesh)
            
    return omesh