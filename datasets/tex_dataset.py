import os
import glob
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from tex_models import utils
from meshRenderer.utils import get_rays
from torch.utils.data import DataLoader


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    pts = pts[valid]
    center = pts.mean(0)
    return center, pts

def normalize_poses(poses, pts):
    center, pts = get_center(pts)

    z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    Rc = torch.stack([x, y, z], dim=1)
    tc = center.reshape(3, 1)

    R, t = Rc.T, -Rc.T @ tc

    poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
    inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)

    poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)
    scale = poses_norm[...,3].norm(p=2, dim=-1).min()
    poses_norm[...,3] /= scale

    pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
    pts = pts / scale

    return poses_norm, pts

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None,:]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:,2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        return{
            'mv'         : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mtx_in'     : torch.cat(list([item['mtx_in'] for item in batch]), dim=0),
            'view_pos'   : torch.cat(list([item['view_pos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp'        : iter_spp,
            'img'        : torch.cat(list([item['img'] for item in batch]), dim=0)
        }
    
    def collate_fn(self, batch):
        iter_H, iter_W = batch[0]['H'], batch[0]['W']
        return {
            'mvp'    : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'rays_o' : torch.cat(list([item['rays_o'] for item in batch]), dim=0),
            'rays_d' : torch.cat(list([item['rays_d'] for item in batch]), dim=0),
            'images' : torch.cat(list([item['images'] for item in batch]), dim=0),
            'H'      : iter_H,
            'W'      : iter_W,
        }


class ColmapDatasetBase(Dataset):
    def __init__(self, args, examples=None):
        self.img_wh       = args.img_wh
        self.spp          = args.spp
        self.examples     = examples
        
        self.base_dir = os.path.join(args.root_dir, args.case)
        self.cam_near_far = [0.1, 1000.0]
        self.setup()
    
    def setup(self):
        camdata = read_cameras_binary(os.path.join(self.base_dir, 'sparse/0/cameras.bin'))
        H = int(camdata[1].height)
        W = int(camdata[1].width)
        F = camdata[1].params[0]

        w, h = self.img_wh
        assert round(W / w * h) == H
        self.w, self.h = w, h
        self.factor = w / W

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * self.factor
            cx = camdata[1].params[1] * self.factor
            cy = camdata[1].params[2] * self.factor
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.factor
            fy = camdata[1].params[1] * self.factor
            cx = camdata[1].params[2] * self.factor
            cy = camdata[1].params[3] * self.factor
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")

        imdata = read_images_binary(os.path.join(self.base_dir, 'sparse/0/images.bin'))
        names  = [imdata[k].name for k in imdata]
        perms  = np.argsort(names)
        
        all_c2w = []
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        for i, d in enumerate(imdata.values()):
            R = d.qvec2rotmat()
            t = d.tvec.reshape([3, 1])
            # c2w = torch.from_numpy(np.concatenate([np.concatenate([R.T, -R.T@t], axis=1), bottom], axis=0)).float() # modify by @lixiang
            c2w = np.concatenate([np.concatenate([R.T, -R.T@t], axis=1), bottom], axis=0)                             # modify by @lixiang
            # c2w[:,1:3] *= -1. # COLMAP => OpenGL   
            all_c2w.append(c2w)
        
        # all_c2w = np.linalg.inv(np.stack(all_w2c, 0))
        self.all_c2w = torch.from_numpy(np.stack(all_c2w, 0)).float()

        pts3d = read_points3d_binary(os.path.join(self.base_dir, 'sparse/0/points3D.bin'))
        pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()                                    # modify by @lixiang
        # pts3d = np.array([pts3d[k].xyz for k in pts3d])                                                            # modify by @lixiang            
 
        self.all_c2w, pts3d = normalize_poses(self.all_c2w[:, :3, :4], pts3d)
        
        hwf   = np.array([H, W, F]).reshape([3, 1])
        
        poses = self.all_c2w[:, :3, :4].numpy().transpose([1, 2, 0])
        poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)
        poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :],
                                poses[:, 3:4, :], poses[:, 4:5, :]], 1)
        
        ## save and load poses
        poses_arr = []
        for i in perms:
            poses_arr.append(poses[..., i].ravel())
            
        i_poses = np.array(poses_arr).reshape([-1, 3, 5]).transpose([1, 2, 0])
        i_poses = np.concatenate([i_poses[:, 1:2, :], -i_poses[:, 0:1, :], i_poses[:, 2:, :]], 1)       # Taken from nerf, swizzles from LLFF to expected coordinate system
        i_poses = np.moveaxis(i_poses, -1, 0).astype(np.float32)
        
        lcol        = np.array([0, 0, 0, 1], dtype=np.float32)[None, None, :].repeat(i_poses.shape[0], 0)
        self.imvs   = torch.tensor(np.concatenate((i_poses[:, :, 0:4], lcol), axis=1), dtype=torch.float32)
        self.aspect = self.w / self.h
        self.fovy   = focal_length_to_fovy(i_poses[:, 2, 4], i_poses[:, 0, 4])
        
        self.preloaded_data = []
        for i in range(self.imvs.shape[0]):
            self.preloaded_data += [self._parse_frame(i)]
        
    def _parse_frame(self, idx):
        # Load image and mask data
        all_images = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) 
                      if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        all_masks  = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "masks", "*"))) 
                      if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        # assert len(all_images) == self.imvs.shape[0] and len(all_masks) == self.imvs.shape[0]
        
        img   = _load_img(all_images[idx])
        mask  = _load_mask(all_masks[idx])
        img_alpha = torch.cat((img, mask[..., 0:1]), dim=-1)
        
        ## setup transforms
        proj   = prespective(self.fovy[idx, ...], self.aspect, self.cam_near_far[0], self.cam_near_far[1])
        mv     = torch.linalg.inv(self.imvs[idx, ...])
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img_alpha[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...]

    def __len__(self):
        return self.imvs.shape[0] if self.examples is None else self.examples

    def __getitem__(self, itr):
        img_alpha, mv, mtx_in, view_pos = self.preloaded_data[itr % self.imvs.shape[0]]
        return {
            'mv'         : mv,
            'mtx_in'     : mtx_in,
            'view_pos'   : view_pos,
            'resolution' : self.img_wh,
            'spp'        : self.spp,
            'img'        : img_alpha
        }


def focal_length_to_fovy(focal_length, sensor_height):
    return 2 * np.arctan(0.5 * sensor_height / focal_length)

def prespective(fovy=0.7854, aspect=1.0, n=0.05, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

def _load_mask(fn):
    img = torch.tensor(utils.load_image(fn), dtype=torch.float32)
    if len(img.shape) == 2:
        img = img[..., None].repeat(1, 1, 3)
    return img

def _load_img(fn):
    img = utils.load_image_raw(fn)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = utils.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img


class MeshDatasetBase(Dataset):
    def __init__(self, args, device, examples=None):
        self.img_wh       = args.img_wh
        self.device       = device
        self.examples     = examples
        
        self.base_dir = os.path.join(args.root_dir, args.case)
        self.cam_near_far = [0.05, 1000.0]
        self.downscale = 1
        self.bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        self.setup()
    
    def setup(self):
        camdata = read_cameras_binary(os.path.join(self.base_dir, 'sparse/0/cameras.bin'))
        H = int(camdata[1].height)
        W = int(camdata[1].width)
        F = camdata[1].params[0]

        w, h = self.img_wh
        assert round(W / w * h) == H
        self.w, self.h = w, h
        self.factor = w / W

        imdata = read_images_binary(os.path.join(self.base_dir, 'sparse/0/images.bin'))
        names  = [imdata[k].name for k in imdata]
        perms  = np.argsort(names)
        
        all_c2w = []
        for i, d in enumerate(imdata.values()):
            R = d.qvec2rotmat()
            t = d.tvec.reshape([3, 1])
            # c2w = torch.from_numpy(np.concatenate([np.concatenate([R.T, -R.T@t], axis=1), bottom], axis=0)).float() # modify by @lixiang
            c2w = np.concatenate([np.concatenate([R.T, -R.T@t], axis=1), self.bottom], axis=0)                             # modify by @lixiang   shape: [4, 4]
            # c2w[:,1:3] *= -1. # COLMAP => OpenGL   
            all_c2w.append(c2w)                                                                                       
        
        # all_c2w = np.linalg.inv(np.stack(all_w2c, 0))
        self.all_c2w = torch.from_numpy(np.stack(all_c2w, 0)).float()                                                # shape: [N, 4, 4]

        pts3d = read_points3d_binary(os.path.join(self.base_dir, 'sparse/0/points3D.bin'))
        pts3d = torch.from_numpy(np.array([pts3d[k].xyz for k in pts3d])).float()                                    # modify by @lixiang
        # pts3d = np.array([pts3d[k].xyz for k in pts3d])                                                            # modify by @lixiang            
 
        self.all_c2w, pts3d = normalize_poses(self.all_c2w[:, :3, :4], pts3d)                                        # self.all_c2w [N, 3, 4]
        
        hwf   = np.array([H, W, F]).reshape([3, 1])
        
        # define intrinsics
        cx, cy = self.w / 2.0, self.h / 2.0
        fl_x = fl_y = F
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        
        poses = self.all_c2w[:, :3, :4].numpy().transpose([1, 2, 0])
        poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)
        poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :],
                                poses[:, 3:4, :], poses[:, 4:5, :]], 1)
        
        ## save and load poses
        poses_arr = []
        for i in perms:
            poses_arr.append(poses[..., i].ravel())
            
        i_poses = np.array(poses_arr).reshape([-1, 3, 5]).transpose([1, 2, 0])
        i_poses = np.concatenate([i_poses[:, 1:2, :], -i_poses[:, 0:1, :], i_poses[:, 2:, :]], 1)       # Taken from nerf, swizzles from LLFF to expected coordinate system
        i_poses = np.moveaxis(i_poses, -1, 0).astype(np.float32)
        
        lcol        = np.array([0, 0, 0, 1], dtype=np.float32)[None, None, :].repeat(i_poses.shape[0], 0)
        self.imvs   = torch.tensor(np.concatenate((i_poses[:, :, 0:4], lcol), axis=1), dtype=torch.float32)
        self.aspect = self.w / self.h
        self.fovy   = focal_length_to_fovy(i_poses[:, 2, 4], i_poses[:, 0, 4])
        
        self.preloaded_data = []
        for i in range(self.imvs.shape[0]):
            self.preloaded_data += [self._parse_frame(i)]
        
    def _parse_frame(self, idx):
        # Load image and mask data
        all_images = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) 
                      if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        all_masks  = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "masks", "*"))) 
                      if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        # assert len(all_images) == self.imvs.shape[0] and len(all_masks) == self.imvs.shape[0]
        
        img   = _load_img(all_images[idx])
        mask  = _load_mask(all_masks[idx])
        img_alpha = torch.cat((img, mask[..., 0:1]), dim=-1)
        
        ## setup transforms
        proj   = prespective(self.fovy[idx, ...], self.aspect, self.cam_near_far[0], self.cam_near_far[1])
        mv     = torch.linalg.inv(self.imvs[idx, ...])
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img_alpha[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...]

    def __len__(self):
        return self.imvs.shape[0]
    
    def __getitem__(self, index):
        img_alpha, mv, mtx_in, view_pos = self.preloaded_data[index % self.imvs.shape[0]]
        
        results = {'H': self.h, 'W': self.w}
        num_rays = -1
        poses = self.all_c2w[index]                                             # [N, 3, 4]
        poses = np.concatenate([poses, self.bottom], axis=0)                    # [4, 4]
        poses = torch.from_numpy(poses.astype(np.float32)).unsqueeze(0).to(self.device)            # [N, 4, 4]
        
        rays = get_rays(poses, self.intrinsics, self.h, self.w, num_rays)
        
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']
        
        mvp = mtx_in.to(self.device)
        results['mvp'] = mvp
        
        images = img_alpha.to(self.device)
        img_C = img_alpha.shape[-1]
        images = images.view(-1, img_C)
        results['images'] = images
        
        return results