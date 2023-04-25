# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-08

import argparse
import os
import nvdiffrast.torch as dr

from datasets import tex_dataset
from tex_models import mesh
from tex_models.geometry import Tex_Geometry
from tex_models.render import render
from tex_models import light


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data')
    parser.add_argument('--case', default='bitong_138')
    parser.add_argument('--out_dir', default='./Exp_out')
    parser.add_argument('--iters', default=2500)
    parser.add_argument('--base_lr', default=0.01)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--img_wh', default=[1024, 1024])
    parser.add_argument('--use_mask', default=True)
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')
    parser.add_argument('--train_res', default=[1024, 1024])
    parser.add_argument('--spp', default=2)
    parser.add_argument('--warmup_iter', default=100)
    parser.add_argument('--log_interval', default=10)
    parser.add_argument('--background', default='white')
    parser.add_argument('--val_render', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='if true, set logging level to DEBUG')

    args = parser.parse_args()

    # set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    args.base_dir = os.path.join(args.root_dir, args.case)
    
    ## load datasets
    examples = (args.iters+1) * args.batch_size
    train_data = tex_dataset.ColmapDatasetBase(args, examples)
    val_data = tex_dataset.ColmapDatasetBase(args)
    
    ## load light model
    lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    lgt = lgt.clone()
    
    glctx = dr.RasterizeGLContext()
    
    ## load base mesh model
    mesh_folder = os.path.join(args.base_dir, 'base_mesh')
    mesh_file = max([os.path.join(mesh_folder, d) for d in os.listdir(mesh_folder)], key=os.path.getmtime)
    print("=> load base mesh file from {}".format(os.path.join(args.base_dir, 'base_mesh', mesh_file)))
    
    ## texture mesh
    tex_geom = Tex_Geometry(mesh_file)
    tex_geom.initial_material()
    tex_geom.xatlas_uvmap()
    tex_mat = tex_geom.get_TexMat(glctx)
    
    ## optimize material
    m_optimizer = render.optimize_mesh(args, glctx, tex_mat, tex_mat.material, lgt, train_data)
    o_geometry, o_material = m_optimizer.train_step()
    
    ## saved output mesh and texture
    out_mesh = mesh.getMesh(o_geometry, o_material)
    saved_folder = os.path.join(args.out_dir, args.case, "out_mesh")
    os.makedirs(saved_folder, exist_ok=True)
    mesh.write_obj(saved_folder, out_mesh)
    light.save_env_map(os.path.join(saved_folder, "probe.hdr"), lgt)
    
    if args.val_render:
        display = [{"latlong": True}, {"bsdf": "kd"}, {"bsdf": "ks"}, {"bsdf": "normal"}]
        validate = render.Validate(args, glctx, o_geometry, o_material, lgt, val_data, display)
        validate.steup()


if __name__ == '__main__':
    main()
