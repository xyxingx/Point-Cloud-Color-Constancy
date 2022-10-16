# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:42:27 2021

check ETH3D-depth and generate point cloud based on it


@author: Xiaoyan
"""
import cv2
import numpy as np
import os
from PIL import Image
from scipy import misc
import time
import array
import struct
from depth_map_utils import *
import open3d as o3d
from matplotlib import pyplot as plt

def pltim(img,name):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(name)

def xshow(filename, nx, nz):
    f = open(filename, "rb")
    pic = np.zeros((nx, nz))
    for i in range(nx):
        for j in range(nz):
            data = f.read(4)
            elem = struct.unpack("f", data)[0]
            pic[i][j] = elem
    f.close()
    return pic




# fnn = 'DSC_6535'

depth_path = 'D:/data/eth3d_de/depth/ground_truth_depth/dslr_images/'
img_path = 'D:/data/eth3d_de/png/'
depth_list = os.listdir(depth_path)
save_path_16 = 'D:/data/eth3d_pcd/pcd_16/'
save_path_64 = 'D:/data/eth3d_pcd/pcd_64/'
save_path_256 = 'D:/data/eth3d_pcd/pcd_256/'
if not os.path.isdir(save_path_16):
    os.makedirs(save_path_16)

if not os.path.isdir(save_path_256):
    os.makedirs(save_path_256)


scale = 256
for fn in depth_list:
    fnn = fn.strip().split('.')[0]
    stime = time.time()
    if not os.path.exists(img_path+fnn+'_16bit.png'):
        print('no img!')
        continue
    depth_map = np.array(xshow(depth_path+fnn+'.JPG',4032,6048),'float32')
    depth_map[depth_map==np.inf]=0
    depth_map_fa = fill_in_fast(depth_map,np.max(depth_map))

    img_raw = np.array(cv2.imread(img_path+fnn+'_16bit.png',-1),'float32')
    img_raw = np.clip(img_raw/16383,0,1)
    ### camera model part, it could be pre-defined for faster generation
    
    fx = 3406.79
    fy = 3404.57
    cx = 3040.861
    cy = 2014.4
    z = depth_map_fa/10
    v_len = int(depth_map.shape[0])
    u_len = int(depth_map.shape[1])
    pcd = np.zeros((v_len,u_len,3))
    yy = np.arange(0,4032)
    yy = np.expand_dims(yy,1).repeat(6048,1)
    xx = np.arange(0,6048)
    xx = np.expand_dims(xx,0).repeat(4032,0)
    pcd[...,0] = (xx-cx)*z/fx
    pcd[...,1] = (yy-cy)*z/fy
    pcd[...,2] = z

    ### point cloud resize part
    pcd_resize = cv2.resize(pcd,(scale,scale))
    pcd_resize1 = pcd_resize.reshape(scale**2,3) 
    pcd_color = cv2.resize(img_raw,(scale,scale))
    pcd_color_reshape = pcd_color.reshape(scale**2,3)
    pcd_cl = o3d.utility.Vector3dVector(pcd_resize1)
    pcd_cc = o3d.geometry.PointCloud(pcd_cl)
    pcd_cc.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_cc2 = np.asarray(pcd_cc.points)
    pcdd1 = np.concatenate((pcd_resize1,pcd_color_reshape),1)

    np.save(save_path_256+fn,pcdd1)

