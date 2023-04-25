# /usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-03-21

import os
import json
import numpy as np
import tqdm
import cv2
import torch

from .utils import create_dodecahedron_cameras, get_rays
from torch.utils.data import DataLoader

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    pose[:3, 3] = pose[:3, 3] * scale + np.array(offset)
    pose = pose.astype(np.float32)
    
    return pose

class meshDataset:
    def __init__(self, args, device, type='train', n_test=10):
        super().__init__()
        
        self.opt = args
        self.device = device
        self.type = type
        
        self.root_path = args.path
        self.preload = args.preload  # preload data into GPU
        self.scale = args.scale      # camera radius scale to make sure camera are inside the bounding box.
        self.bound = args.bound      # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = args.fp16        # if preload, load into fp16.
        
        self.downscale = 1
        self.offset = [0, 0, 0]      # camera offset
        
        if self.scale == -1:
            print(f'[WARN] --data_format nerf cannot auto-choose --scale, use 1 as default.')
            self.scale = 1
        
        ## auto-detect transforms.json
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap'
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender'
        else:
            raise NotImplementedError(f'[meshDataset] Cannot find transforms*.json under {self.root_path}')
        
        ## load compatible fromat data
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            with open(os.path.join(self.root_path, f'transforms_{self.type}.json'), 'r') as f:
                transform = json.load(f)
        
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')
        
        ## load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // self.downscale
            self.W = int(transform['w']) // self.downscale
        else:
            self.H = self.W = None
        
        ## read images
        frames = np.array(transform["frames"])
        if 'time' in frames[0]:
            frames = np.array([f for f in frames is f['time'] == 0])
            print(f'[INFO] selecting time == 0 frames: {len(transform["frames"])} --> {len(frames)}')
        

        self.poses = []
        self.images = []
        for f in tqdm.tqdm(frames, desc=f'Loading {type} data...'):
            f_path = os.path.join(self.root_path, f['file_path'])
            if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                f_path += '.png'
            
            if not os.path.exists(f_path):
                print(f'[WARN] {f_path} not exists!')
                continue
            
            pose = np.array(f['transform_matrix'], dtype=np.float32)    # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
            
            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)            # [H, W, 3] to [H, W, 4]
            if self.H is None or self.W is None:
                self.H = image.shape[0] // self.downscale
                self.W = image.shape[1] // self.downscale
                
            ## add support for the alpha channel as a mask.
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            
            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            
            self.poses.append(pose)
            self.images.append(image)
        
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))   # [N , 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0).astype(np.uint8)) # [N, H, W, C]
        
        ## calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        
        ## load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / self.downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / self.downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            fl_x = self.W / (2*np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2*np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Faild to load focal length, please check the transforms.json!')
        
        cx = (transform['cx'] / self.downscale) if 'cx' in transform else (self.W / 2.0)
        cy = (transform['cy'] / self.downscale) if 'cy' in transform else (self.H / 2.0)
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        
        ## prespective projection matrix
        self.near = 0.05
        self.far  = 1000
        y = self.H / (2.0 * fl_y)
        aspect = self.W / self.H
        self.projection = np.array([[1/(y*aspect), 0, 0, 0],
                                    [0, -1/y, 0, 0],
                                    [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                                    [0, 0, -1, 0]], dtype=np.float32)
        self.projection = torch.from_numpy(self.projection)
        self.mvps = self.projection.unsqueeze(0) @ torch.inverse(self.poses)
        
        ## tmp: dodecahedron_cameras for mesh visibility test
        dodecahedron_poses = create_dodecahedron_cameras()
        ## visualize_poses(dodecahedron_poses, bound=self.opt.bound, points=self.pts3d)
        self.dodecahedron_poses = torch.from_numpy(dodecahedron_poses.astype(np.float32)) # [N, 4, 4]
        self.dodecahedron_mvps  = self.projection.unsqueeze(0) @ torch.inverse(self.dodecahedron_poses)
        
        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                self.images = self.images.to(self.device)
            
            self.projection = self.projection.to(self.device)
            self.mvps = self.mvps.to(self.device)
    
    
    def collate(self, index):
        B = len(index)         # a list of length 1
        results = {'H': self.H, 'W': self.W}
        num_rays = -1
        
        poses = self.poses[index].to(self.device)    # [N, 4, 4]
        rays  = get_rays(poses, self.intrinsics, self.H, self.W, num_rays)
        
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']
        results['index']  = index
        
        mvp = self.mvps[index].to(self.device)
        results['mvp'] = mvp
        if self.images is not None:
            images = self.images[index].squeeze(0).float().to(self.device) / 255   # [H, W, 3/4]
            C = self.images.shape[-1]
            images = images.view(-1, C)

            results['images'] = images
        
        return results
    
    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=True, num_workers=0)
        loader._data = self
        loader.has_gt = self.images is not None
        
        return loader
           