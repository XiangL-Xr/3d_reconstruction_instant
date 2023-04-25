# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-03-14

import os, sys
import torch
import trimesh
import xatlas
import cv2
import json
import math
import time
import tqdm

import nvdiffrast.torch as dr
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX

from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.neighbors import NearestNeighbors
from rich.console import Console

from .utils import *


class MeshRenderer(torch.nn.Module):
    def __init__(self, args):
        super(MeshRenderer, self).__init__()

        self.opt = args
        self.root_dir = args.root_dir
        self.case = args.case
        self.grid_size = 128
        self.min_near = 0.05
        self.density_thresh = 10
        self.ind_dim = 0
        self.ind_num = 500
        # self.ind_codes = None
        # self.cuda_ray = True
        # self.trainable_density_grid = False
        
        ## bound for ray marching (world space)
        # self.real_bound = args.bound
        
        ## bound for grid querying
        self.bound = args.bound
        
        ## define texture training
        self.glctx = dr.RasterizeGLContext(output_db=False)
        # self.glctx = dr.RasterizeCudaContext()
        
        ## sequentially load cascaded meshes
        vertices  = []
        triangles = []
        
        mesh_folder = os.path.join(self.root_dir, self.case, 'base_mesh')
        mesh_file = max([os.path.join(mesh_folder, d) for d in os.listdir(mesh_folder)], key=os.path.getmtime)
        mesh = trimesh.load(mesh_file, force='mesh',
                            skip_material=True, process=False)
        print(f'[INFO] loaded init mesh: {mesh.vertices.shape}, {mesh.faces.shape}')
        
        vertices.append(mesh.vertices)
        triangles.append(mesh.faces)

        vertices = np.concatenate(vertices, axis=0)
        triangles = np.concatenate(triangles, axis=0)
        
        self.vertices = torch.from_numpy(vertices).float().cuda()
        self.triangles = torch.from_numpy(triangles).int().cuda()
        
    
    def get_params(self, lr):
        params = []
        # if self.ind_codes is not None:
        #     params.append({'params': self.ind_codes, 'lr': self.opt.lr * 0.1, 'weight_decay': 0})
        
        # if self.glctx is not None:
        #     params.append({'params': self.vertices_offsets, 'lr': self.opt.lr_vert, 'weight_decay': 0})
        
        return params
    
    
    def forward(self, x, d):
        raise NotImplementedError()
    
    def density(self, x):
        raise NotImplementedError()
    
    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()
    
    
    @torch.no_grad()
    def export_mesh(self, path, h0=2048, w0=2048, png_compression_level=3):
        device = self.vertices.device
        
        def _export_obj(v, f, h0, w0, ssaa=1):
            v_np = v.cpu().numpy()       # [N, 3]
            f_np = f.cpu().numpy()       # [M, 3]
            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
            
            ## unwrap uvs
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0 # max_iterations是一个整数，表示Atlas对象在生成纹理映射时最大的迭代次数。将max_iterations设置为0意味着在生成纹理映射时不进行任何迭代，即只进行一次简单的映射计算。这样可以减少映射时间，但可能会牺牲映射质量。
            pack_options = xatlas.PackOptions()
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            vmapping, ft_np, vt_np = atlas[0]        # [N], [M, 3], [N, 2]
            
            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
            
            ## render uv maps
            uv = vt * 2.0 - 1.0        # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]
            
            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0
            
            ## 光栅化，将三角网格渲染为二维图像，并返回二维图像的深度和颜色信息(r, g, b, a)
            rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
            ## 插值操作，对栅格化得到的深度图rast进行三维插值，计算每个像素点的三维空间坐标(x, y, z)
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)                # [1, h, w, 3]
            ## 插值操作，得到每个像素点是否在三角形面片内部的掩码信息
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]
            
            ## masked query
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)
            
            feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)
            if mask.any():
                xyzs = xyzs[mask]  # [M, 3]
                # batched inference to avoid OOM
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        all_feats.append(self.geo_feat(xyzs[head:tail]).float())
                    head += 640000
                
                feats[mask] = torch.cat(all_feats, dim=0)
            
            feats = feats.view(h, w, -1)       # 6 channels
            mask  = mask.view(h, w)
            
            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().numpy()
            feats = (feats * 255).astype(np.uint8)
            
            ## NN search as a queer antialiasing ...
            mask = mask.cpu().numpy()
            
            inpaint_region = binary_dilation(mask, iterations=3)
            inpaint_region[mask] = 0
            
            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=2)
            search_region[not_search_region] = 0
            
            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)
            
            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)
            
            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]
            
            # do ssaa after the NN search, in numpy
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR)   # albedo
            feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGB2BGR)   # visibility features
            
            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
                feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)
            
            cv2.imwrite(os.path.join(path, f'feat0.png'), feats0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            cv2.imwrite(os.path.join(path, f'feat1.png'), feats1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            
            # save obj(v, vt, f /)
            obj_file = os.path.join(path, f'mesh.obj')
            mtl_file = os.path.join(path, f'mesh.mtl')
            
            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:
                fp.write(f'mtllib mesh.mtl \n')
                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
                
                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n')
                
                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl defaultMat \n')
                for i in range(len(f_np)):
                    fp.write(f"f {f_np[i,0]+1}/{ft_np[i,0]+1} {f_np[i,1]+1}/{ft_np[i,1]+1} {f_np[i, 2]+1}/{ft_np[i,2]+1} \n")
            
            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl defaultMat \n')
                fp.write(f'Ka 1 1 1 \n')
                fp.write(f'Kd 1 1 1 \n')
                fp.write(f'Ks 0 0 0 \n')
                fp.write(f'Tr 1 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0 \n')
                fp.write(f'map_Kd feat0.png \n')
                fp.write(f'map_Ks feat1.png \n')
        
        v = self.vertices.detach()
        f = self.triangles.detach()
        
        _export_obj(v, f, h0, w0, self.opt.ssaa)
        # half the texture resolution for remote area
        if h0 > 2048 and w0 > 2048:
            h0 // 2
            w0 // 2
        
        ## save mlp as json
        params = dict(self.specular_net.named_parameters())
        mlp = {}
        for k, p in params.items():
            p_np = p.detach().cpu().numpy().T
            print(f'[INFO] writing MLP param {k}: {p_np.shape}')
            mlp[k] = p_np.tolist()
        
        # mlp['bound'] = self.bound
        
        mlp_file = os.path.join(path, f'mlp.json')
        with open(mlp_file, 'w') as fp:
            json.dump(mlp, fp, indent=2)
    
    def render_mesh(self, rays_o, rays_d, mvp, h0, w0, index=None, bg_color=None, shading='full', **kwargs):
        prefix = rays_d.shape[:-1]
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_d.shape[0]         # N = B * N, in fact
        device = rays_d.device
        
        ## do super-sampling
        if self.opt.ssaa > 1:
            h = int(h0 * self.opt.ssaa)
            w = int(w0 * self.opt.ssaa)
            
            # interpolate rays_d when ssaa > 1 ...
            dirs = rays_d.view(h0, w0, 3)
            dirs = scale_img_hwc(dirs, (h, w), mag='nearest').view(-1, 3).contiguous()
        else:
            h, w, = h0, w0
            dirs = rays_d.contiguous()
        
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        if bg_color is None:
            bg_color = 1           # mix background color
        
        if torch.is_tensor(bg_color) and len(bg_color.shape) == 2:
            bg_color = bg_color.view(h0, w0, 3)                    # [N, 3] to [h, w, 3]
        
        ind_code = None
        results = {}
        # vertices = self.vertices + self.vertices_offsets    # [N, 3]
        vertices = self.vertices
        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0)   # [1, N, 4]
        
        rast, _ = dr.rasterize(self.glctx, vertices_clip, self.triangles, (h, w))
        xyzs, _ = dr.interpolate(vertices.unsqueeze(0), rast, self.triangles)                          # [1, H, W, 3]
        mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, self.triangles)  # [1, H, W, 1]
        mask_flatten = (mask > 0).view(-1)
        xyzs = xyzs.view(-1, 3)
        
        rgbs = torch.zeros(h * w, 3, device=device, dtype=torch.float32)
        if mask_flatten.any():
            with torch.cuda.amp.autocast(enabled=True):
                mask_rgbs, masked_specular = self.rgb(xyzs[mask_flatten].detach(), dirs[mask_flatten], ind_code, shading)
            
            rgbs[mask_flatten] = mask_rgbs.float()
        
        rgbs = rgbs.view(1, h, w, 3)
        alphas = mask.float().detach()
        
        alphas = dr.antialias(alphas, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        rgbs   = dr.antialias(rgbs,   rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        
        image = alphas * rgbs
        depth = alphas * rast[0, :, :, [2]]
        T = 1 - alphas
        
        ## trig_id for updating trig errors
        # trig_id = rast[0, :, :, -1] - 1     # [h, w]
        
        ## ssaa
        if self.opt.ssaa > 1:
            image = scale_img_hwc(image, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            T = scale_img_hwc(T, (h0, w0))
            # trig_id = scale_img_hw(trig_id.float(), (h0, w0), mag='nearest', min='nearest').long()
        
        # self.triangles_errors_id = trig_id
        
        image = image + T * bg_color
        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        
        results['depth'] = depth
        results['image'] = image
        
        return results
                


class Trainer(object):
    def __init__(self,
                 args,
                 model,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 metrics=[],
                 local_rank=0,
                 device=None,
                 mute=False,
                 fp16=False,
                 save_interval=1,
                 max_keep_ckpt=2,
                 workspace='workspace',
                 best_mode='min',
                 use_loss_as_metric=True,
                 use_tensorboardX=True,
                 scheduler_update_every_step=False,
                 ):
        
        self.opt = args
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.workspace = workspace
        self.ema = None
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.max_keep_ckpt = max_keep_ckpt
        self.save_interval = save_interval
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.name = 'ngp_' + self.opt.case
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        
        self.model = model.to(self.device)
        
        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        
        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)
        else:
            self.optimizer = self.optimizer_fn(self.model)
        
        if lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)
        else:
            self.lr_scheduler = self.lr_scheduler_fn(self.optimizer)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        
        ## variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],
            "checkpoints": [],
            "best_result": None,
        }
        
        ## auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'
        
        ## workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")
            
            self.ckpt_path = os.path.join(self.workspace, self.opt.case, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
        
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        
        self.log(args)
        self.log(self.model)
    
    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()
    
    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()
    
    ### --------------------------------------------
    def train_step(self, data):
        rays_o = data['rays_o']      # [N, 3]
        rays_d = data['rays_d']      # [N, 3]
        
        images = data['images']      # [N, 3/4]
        N, C = images.shape
        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])
        
        if self.opt.background == 'white':
            bg_color = 1
        else:
            bg_color = torch.rand(N, 3, device=self.device)      # [N, 3], pixel-wise random
        
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        
        mvp = data['mvp'].squeeze(0)          # [4, 4]
        H, W = data['H'], data['W']
        outputs = self.model.render_mesh(rays_o, rays_d, mvp, H, W, bg_color=bg_color, shading='full', **vars(self.opt))
        
        pred_rgb = outputs['image']
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)      # [H, W]
        loss = loss.mean()
        
        return pred_rgb, gt_rgb, loss
    
    def train(self, train_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, self.opt.case, "run", self.name))
        
        start_t = time.time()
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loader)
            
            if (self.epoch % self.save_interval == 0 or self.epoch == max_epochs) and self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)
        
        end_t = time.time()
        self.log(f"[INFO] training takes {(end_t - start_t)/60:.6f} minutes.")
        
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()
    
    
    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        total_loss = 0.0
        self.model.train()  
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        self.local_step = 0
        for iter, data in enumerate(loader):
            self.local_step += 1
            self.global_step += 1
            
            self.optimizer.zero_grad()
            
            preds, truths, loss_net = self.train_step(data)
            loss = loss_net
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            
            loss_val = loss_net.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
        
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                pbar.update(loader.batch_size)
                
        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        
        if self.local_rank == 0:
            pbar.close()
        
        self.log(f"==> Finished Epoch {self.epoch}.")
    
    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'
        
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }
        
        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
        
        if not best:
            state['model'] = self.model.state_dict()
            file_path = f"{name}.pth"
            if remove_old:
                self.stats["checkpoints"].append(file_path)
                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)
        
        torch.save(state, os.path.join(self.ckpt_path, file_path))
    
    def m_export(self, save_path=None, resolution=2048):
        if save_path is None:
            save_path = os.path.join(self.workspace, self.opt.case, 'mesh_export')
        
        self.log(f"==> Exporting mesh to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        self.model.export_mesh(save_path, resolution, resolution)
        self.log(f"==> Exported mesh to {save_path}")
        