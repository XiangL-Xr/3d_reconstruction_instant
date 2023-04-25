# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-08

import os
import time
import torch
import nvdiffrast.torch as dr
import numpy as np

from .. import utils
from ..render import renderutils as ru
from .. import light
from .. import mesh


## Helper functions ====================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


## Define UV map =======================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):
    # clip space transform
    uv_clip0 = mesh.v_tex[None, ...] * 2.0 - 1.0
    
    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip0, torch.zeros_like(uv_clip0[..., 0:1]), torch.ones_like(uv_clip0[..., 0:1])), dim=-1)
    
    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)
    
    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())
    
    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10        # Combined kd_ks_normal must be 9 or 10 channels
    perturbed_nrm = all_tex[..., -3:]
    
    return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], utils.safe_normalize(perturbed_nrm)

def render_mesh(glctx, opt_mesh, target, lgt, resolution, spp=1, num_layers=1, background=None, bsdf=None):
    assert opt_mesh.t_pos_idx.shape[0] > 0                          # got empty training triangle mesh  (unrecoverable discontinuity)
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])
    
    full_res = [resolution[0]*spp, resolution[1]*spp]
    # convert numpy arrays to torch tensors
    mtx_in, view_pos = prepare_input_vector(target['mtx_in'], target['view_pos'])
    # clip space transform
    v_pos_clip = ru.xfm_points(opt_mesh.v_pos[None, ...], mtx_in)
    
    # render all layer front-to-back
    layers = []
    with dr.DepthPeeler(glctx, v_pos_clip, opt_mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, opt_mesh, view_pos, lgt, resolution, spp, bsdf), rast)]
    
    # setup background
    if background is not None:
        if spp > 1:
            background = utils.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')
    
    # composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True, v_pos_clip, opt_mesh)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False, v_pos_clip, opt_mesh)
        
        # downscale to framebuffer resolution. use avg pooling
        out_buffers[key] = utils.avg_pool_nhwc(accum, spp) if spp > 1 else accum
    
    return out_buffers
    

def render_layer(rast, rast_deriv, mesh, view_pos, lgt, resolution, spp, bsdf, msaa=True):
    full_res = [resolution[0]*spp, resolution[1]*spp]
    # scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = utils.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = utils.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv
    
    # interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())
    
    # compute geometric normals. we need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = utils.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normals_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normals_indices.int())
    
    # compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _  = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int())
    
    # texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)
    
    # Shade
    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
                    view_pos, lgt, mesh.material, bsdf)

    # scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = utils.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')
    
    return buffers

def shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv,
          view_pos, lgt, material, bsdf=None):
    # texture lookups
    perturbed_nrm = None
    if 'kd_ks_normal' in material:
        # combined texture, used for MLPs because lookups are expensive
        all_tex_jitter = material['kd_ks_normal'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device='cuda'))
        all_tex = material['kd_ks_normal'].sample(gb_pos)
        assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10
        
        kd, ks, perturbed_nrm = all_tex[..., :-6], all_tex[..., -6:-3], all_tex[..., -3:]
        # compute albedo(kd) gradient, used for material regularizer
        kd_grad = torch.sum(torch.abs(all_tex_jitter[..., :-6] - all_tex[..., :-6]), dim=-1, keepdim=True) / 3
    else:
        kd_jitter = material['kd'].sample(gb_texc + torch.normal(mean=0, std=0.005, size=gb_texc.shape, device='cuda'), gb_texc_deriv)
        kd = material['kd'].sample(gb_texc, gb_texc_deriv)
        ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3]     # skip alpha
        if 'normal' in material:
            perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)
        kd_grad = torch.sum(torch.abs(kd_jitter[..., 0:3] - kd[..., 0:3]), dim=-1, keepdim=True) / 3
    
    # separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1])
    kd = kd[..., 0:3]
    
    # normal perturbation & normal bend
    if 'no_perturbed_nrm' in material and material['no_perturbed_nrm']:
        perturbed_nrm = None
    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal,
                                          two_sided_shading=True, opengl=True)
    
    # evaluate BSDF
    assert 'bsdf' in material or bsdf is not None
    bsdf = material['bsdf'] if bsdf is None else bsdf
    if bsdf == 'pbr':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=True)
        else:
            assert False, "invalid light type!"
    elif bsdf == 'diffuse':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=False)
        else:
            assert False, "invalid light type!"
    elif bsdf == 'normal':
        shaded_col = (gb_normal  + 1.0) * 0.5
    elif bsdf == 'tangent':
        shaded_col = (gb_tangent + 1.0) * 0.5
    elif bsdf == 'kd':
        shaded_col = kd
    elif bsdf == 'ks':
        shaded_col = ks
    else:
        assert False, "Invalid BSDF '%s'" % bsdf
    
    buffers = {
        'shaded'      : torch.cat((shaded_col, alpha), dim=-1),
        'kd_grad'     : torch.cat((kd_grad, alpha), dim=-1),
        'occlusion'   : torch.cat((ks[..., :1], alpha), dim=-1)
    }
    
    return buffers
    

class Trainer(torch.nn.Module):
    def __init__(self, glctx, opt_mesh, material, lgt):
        super(Trainer, self).__init__()
        self.glctx = glctx
        self.material = material
        self.light = lgt
        
        # define image losses
        self.img_lg1_loss = createloss(loss_name='logl1')
        self.img_mse_loss = torch.nn.functional.mse_loss
        
        self.params = list(self.material.parameters())
        self.params = self.params + list(self.light.parameters())
        
        self.opt_mesh = mesh.getMesh(opt_mesh, self.material)
    
    def forward(self, target, iter):
        self.light.build_mips()
        self.light.xfm(target['mv'])
        
        return self.optimize_step(target, iter)

    def optimize_step(self, target, iteration, bsdf=None):
        buffers = render_mesh(self.glctx, self.opt_mesh, target, self.light, resolution=target['resolution'], 
                              spp=target['spp'], num_layers=1, background=target['background'], bsdf=bsdf)
        
        # image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        #img_mse_loss = self.img_mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        img_mse_loss = self.img_mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        img_lg1_loss = self.img_lg1_loss(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
        img_loss = img_mse_loss + img_lg1_loss
        
        # albedo(kd) smoothnesss regularizer loss
        reg_loss = torch.tensor([0], dtype=torch.float32, device='cuda')
        reg_loss = reg_loss + torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.003 * min(1.0, iteration / 500)
        
        # light white balance regularizer
        reg_loss = reg_loss + self.light.regularizer() * 0.03
        
        return img_loss, reg_loss
        
class optimize_mesh():
    def __init__(self, args, glctx, opt_mesh, opt_material, lgt, train_data):
        super(optimize_mesh, self).__init__()
        
        self.glctx = glctx
        self.opt_mesh = opt_mesh
        self.material = opt_material
        self.lgt = lgt
        self.train_data = train_data
        
        self.iters = args.iters                # 3000
        self.batch_size = args.batch_size      # 4
        self.base_lr = args.base_lr            # 0.01
        self.warmup_iter = args.warmup_iter    # 100
        self.log_interval = args.log_interval  # 10
        
        self.img_loss_vec = []
        self.reg_loss_vec = []
        self.iter_dur_vec = []
        
        self._init_optimizer()
        
    def _init_optimizer(self):
        
        self.trainer = Trainer(self.glctx, self.opt_mesh, self.material, self.lgt)
        self.optimizer = torch.optim.Adam(self.trainer.params, lr=self.base_lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda it: self.lr_schedule(it))
    
    def lr_schedule(self, iter):
        if iter < self.warmup_iter:
            return iter / self.warmup_iter
        else:
            return max(0.0, 10**(-(iter - self.warmup_iter) * 0.0005))     # Exponential falloff from [1.0, 0.1] over 1/0.0005 iters.
    
    def train_step(self):
        ## load train dataloader
        train_dataloader = torch.utils.data.DataLoader(
                                self.train_data,
                                batch_size = self.batch_size,
                                collate_fn = self.train_data.collate,
                                shuffle = True
                           )
        for iter, target in enumerate(train_dataloader):
            target = prepare_batch(target, 'random')                       # mix randomized background into dataset image
            iter_start_time = time.time()
            self.optimizer.zero_grad()
            
            img_loss, reg_loss = self.trainer(target, iter)
            total_loss = img_loss + reg_loss
            
            self.img_loss_vec.append(img_loss.item())
            self.reg_loss_vec.append(reg_loss.item())
            
            total_loss.backward()
            if hasattr(self.lgt, 'base') and self.lgt.base.grad is not None:
                self.lgt.base.grad *= 64
            if 'kd_ks_normal' in self.material:
                self.material['kd_ks_normal'].encoder.params.grad /= 8.0

            self.optimizer.step()
            self.scheduler.step()
            
            with torch.no_grad():
                if 'kd' in self.material:
                    self.material['kd'].clamp_()
                if 'ks' in self.material:
                    self.material['ks'].clamp_()
                if 'normal' in self.material:
                    self.material['normal'].clamp_()
                    self.material['normal'].normalize_()
                if self.lgt is not None:
                    self.lgt.clamp_(min=0.0)

            torch.cuda.current_stream().synchronize()
            self.iter_dur_vec.append(time.time() - iter_start_time)
            if iter % self.log_interval == 0:
                img_loss_avg = np.mean(np.asarray(self.img_loss_vec[-self.log_interval:]))
                reg_loss_avg = np.mean(np.asarray(self.reg_loss_vec[-self.log_interval:]))
                iter_dur_avg = np.mean(np.asarray(self.iter_dur_vec[-self.log_interval:]))
                
                remaining_time = (self.iters-iter) * iter_dur_avg
                
                print("[--iter-- %04d], img_loss: %.6f, reg_loss: %.6f, lr: %.5f, iter_time=%.2f ms, rem_time=%s " %
                      (iter, img_loss_avg, reg_loss_avg, self.optimizer.param_groups[0]['lr'], iter_dur_avg*1000, utils.time2text(remaining_time)))
        
        total_time = np.sum(np.array(self.iter_dur_vec))
        print('-' * 45)
        print("[--iter-- %04d], total_time=%s " % (iter, utils.time2text(total_time)))
        print('-' * 45)
        
        return self.opt_mesh, self.material


class Validate():
    def __init__(self, args, glctx, geometry, material, light, val_data, display):
        super(Validate, self).__init__()
        
        self.glctx = glctx
        self.geometry = geometry
        self.material = material
        self.light = light
        self.val_data = val_data
        self.display = display
        
        self.background = args.background
        self.display_res = args.img_wh
        self.val_dir = os.path.join(args.out_dir, args.case, 'validate')
        os.makedirs(self.val_dir, exist_ok=True)
        
        self.img_mse_loss = torch.nn.functional.mse_loss
        self.mse_values  = []
        self.psnr_values = []
    
    def steup(self):
        ## load val dataloader
        val_dataloader = torch.utils.data.DataLoader(
                                self.val_data,
                                batch_size = 1,
                                collate_fn = self.val_data.collate
                           )
        with open(os.path.join(self.val_dir, 'metrics.txt'), 'w') as fout:
            fout.write('ID, MSE, PSNR\n')
            print('=> Runing validation ...')
            for iter, target in enumerate(val_dataloader):
                target = prepare_batch(target, self.background)
                result_img, result_dict = self.validate_step(target)
                
                ## compute metrics
                opt = torch.clamp(result_dict['opt'], 0.0, 1.0)
                ref = torch.clamp(result_dict['ref'], 0.0, 1.0)
                
                v_mse = self.img_mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
                v_psnr = utils.mse2psnr(v_mse)
                self.mse_values.append(float(v_mse))
                self.psnr_values.append(float(v_psnr))
                
                line = "%d, %1.8f, %1.8f\n" % (iter, v_mse, v_psnr)
                fout.write(str(line))
                
                ## save images
                for k in result_dict.keys():
                    np_img = result_dict[k].detach().cpu().numpy()
                    utils.save_image(self.val_dir + '/' + ('render_%03d_%s.png' % (iter, k)), np_img)
                
                ## save result image
                res_img = result_img.detach().cpu().numpy()
                utils.save_image(self.val_dir + '/' + ('render_%03d_res.png' % iter), res_img)
            
            avg_mse = np.mean(np.array(self.mse_values))
            avg_psnr = np.mean(np.array(self.psnr_values))
            line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
            fout.write(str(line))
            print('------------------------')
            print("## MSE,     PSNR ##")
            print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
            print('------------------------')                
    
    def validate_step(self, target):
        result_dict = {}
        with torch.no_grad():
            self.light.build_mips()
            self.light.xfm(target['mv'])
            
            opt_mesh = mesh.getMesh(self.geometry, self.material)
            buffers = render_mesh(self.glctx, opt_mesh, target, self.light, resolution=target['resolution'],
                                  spp=target['spp'], num_layers=1, background=target['background'], bsdf=None)
            
            result_dict['ref'] = utils.rgb_to_srgb(target['img'][..., 0:3])[0]
            result_dict['opt'] = utils.rgb_to_srgb(buffers['shaded'][..., 0:3])[0]
            result_img = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)
            if self.display is not None:
                for layer in self.display:
                    if 'latlong' in layer and layer['latlong']:
                        if isinstance(self.light, light.EnvironmentLight):
                            result_dict['ligth_img'] = utils.cubemap_to_latlong(self.light.base, self.display_res)
                        result_img = torch.cat([result_img, result_dict['ligth_img']], axis=1)
                    elif 'relight' in layer:
                        if not isinstance(layer['relight'], light.EnvironmentLight):
                            layer['relight'] = light.load_env(layer['relight'])
                        img = render_mesh(self.glctx, opt_mesh, target, layer['relight'], resolution=target['resolution'],
                                          spp=target['spp'], num_layers=1, background=target['background'], bsdf=None)
                        result_dict['relight'] = utils.rgb_to_srgb(img[..., 0:3])[0]
                        result_img = torch.cat([result_img, result_dict['relight']], axis=1)
                    elif 'bsdf' in layer:
                        buffers = render_mesh(self.glctx, opt_mesh, target, self.light, resolution=target['resolution'],
                                              spp=target['spp'], num_layers=1, background=target['background'], bsdf=layer['bsdf'])
                        if layer['bsdf'] == 'kd':
                            result_dict[layer['bsdf']] = utils.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                        elif layer['bsdf'] == 'normal':
                            result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                        else:
                            result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                        result_img = torch.cat([result_img, result_dict[layer['bsdf']]], axis=1)
            
            return result_img, result_dict


def createloss(loss_name=None):
    if loss_name == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif loss_name == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif loss_name == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    else:
        assert False

def prepare_input_vector(mtx_x, pos_x):
    mtx_x = torch.tensor(mtx_x, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_x) else mtx_x
    pos_x = torch.tensor(pos_x, dtype=torch.float32, device='cuda') if not torch.is_tensor(pos_x) else pos_x
    pos_x = pos_x[:, None, None, :] if len(pos_x.shape) == 2 else pos_x
    
    return mtx_x, pos_x

def composite_buffer(key, layers, background, antialias, v_pos_clip, mesh):
    accum = background
    for buffers, rast in reversed(layers):
        alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
        accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
        if antialias:
            accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
    
    return accum

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4
    if bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type
    
    target['mv'] = target['mv'].cuda()
    target['mtx_in'] = target['mtx_in'].cuda()
    target['view_pos'] = target['view_pos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background
    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)
    
    return target