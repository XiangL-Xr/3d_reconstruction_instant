# !/usr/bin/python3
# coding: utf-8
# @Author: lixiang
# @Date: 2023-02-16

import argparse
import os
import open3d as o3d
import numpy as np
import copy
import pymeshfix

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', default='./data/mesh_path/mesh_name')

args = parser.parse_args()


def euclidean_dist(arr1, arr2):
    return np.sqrt(np.sum(np.square(arr2 - arr1), axis=1))

def verts2remove(i_mesh):
    all_verts = np.asarray(i_mesh.vertices)
    centers = np.zeros([all_verts.shape[0], 3])
    all_dists = euclidean_dist(centers, all_verts)
    
    all_index = np.arange(all_verts.shape[0])
    verts_remove_idx = all_index[all_dists > 0.20]
    
    o_mesh = copy.deepcopy(i_mesh)
    o_mesh.remove_vertices_by_index(verts_remove_idx)
    
    return o_mesh

def faces2remove(i_mesh, flag='largest'):
    print("=> Cluster connected triangles...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (i_mesh.cluster_connected_triangles())
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area) 
    
    if flag == 'largest':
        print("=> modify mesh with largest clusters")
        o_mesh = copy.deepcopy(i_mesh)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        o_mesh.remove_triangles_by_mask(triangles_to_remove)
    else:
        print("=> modify mesh with small clusters removed...")
        o_mesh = copy.deepcopy(i_mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
        o_mesh.remove_triangles_by_mask(triangles_to_remove)
    
    return o_mesh
  
def mesh_filtering(i_mesh, iters=10, f_type='average'):
    if f_type == 'average':
        print('=> filter with average with {} iterations'.format(iters))
        mesh_out = i_mesh.filter_smooth_simple(number_of_iterations=iters)
        mesh_out.compute_vertex_normals()
    elif f_type == 'laplacian':
        print('=> filter with Laplacian with {} iterations'.format(iters))
        mesh_out = i_mesh.filter_smooth_laplacian(number_of_iterations=iters)
        mesh_out.compute_vertex_normals
    elif f_type == 'taubin':
        print('=> filter with Taubin with {} iterations'.format(iters))
        mesh_out = i_mesh.filter_smooth_taubin(number_of_iterations=iters)
        mesh_out.compute_vertex_normals()
    else:
        print('=> please select filter type!')
        
    return mesh_out
        

if __name__ == '__main__':
    print('=' * 80)
    ## load mesh
    mesh_path = args.mesh_path
    mesh_name = os.path.basename(mesh_path)
    if mesh_name.endswith('.ply') or mesh_name.endswith('.obj') or mesh_name.endswith('.OBJ'):
        i_mesh = o3d.io.read_triangle_mesh(mesh_path)
        print("=> Load mesh from {} ...".format(mesh_path))
    else:
        print("=> This endswith file is not supported!")

    ## compute input mesh vertex normals
    i_mesh.compute_vertex_normals()
    print("=> input mesh info: ", i_mesh)
    
    ## remove outliers meshs
    #o_mesh = verts2remove(samp_mesh)
    o_mesh = faces2remove(i_mesh, flag='largest')
    
    m_verts = np.asarray(o_mesh.vertices)
    m_faces = np.asarray(o_mesh.triangles)
    
    if m_faces.shape[0] > 200000:
        samp_mesh = o_mesh.simplify_quadric_decimation(target_number_of_triangles=200000)
    else:
        samp_mesh = o_mesh

    ## compute samp mesh vertex normals
    samp_mesh.compute_vertex_normals()
    print("=> sample mesh info: ", samp_mesh)
    s_verts = np.asarray(samp_mesh.vertices)
    s_faces = np.asarray(samp_mesh.triangles)
    
    ## mesh reqair
    print('=> start mesh repair ... --------------------------------')
    meshfix = pymeshfix.PyTMesh()
    meshfix.load_array(s_verts, s_faces)
    print('=> There are {:d} boundaries in input mesh'.format(meshfix.boundaries()))
    meshfix.fill_small_boundaries()
    print('=> There are {:d} boundaries after mesh repair'.format(meshfix.boundaries()))
    # Clean (removes self intersections)
    meshfix.clean(max_iters=10, inner_loops=3)
    
    fixed_verts, fixed_faces = meshfix.return_arrays()
    fixed_mesh = o3d.geometry.TriangleMesh()
    fixed_mesh.vertices = o3d.utility.Vector3dVector(fixed_verts)
    fixed_mesh.triangles = o3d.utility.Vector3iVector(fixed_faces)
    fixed_mesh.compute_vertex_normals()
    
    ## mesh filtering
    print('=> start mesh filtering ... --------------------------------')
    final_mesh = mesh_filtering(fixed_mesh, iters=2, f_type='average')      # select choice ['average, laplacian, taubin]
    
    ## saved mesh
    remesh_name = mesh_name.split('.')[0] + '_optim.obj'
    o3d.io.write_triangle_mesh(mesh_path.replace(mesh_name, remesh_name), final_mesh)
    print('## All Done! ## ' + '=' * 75)
    