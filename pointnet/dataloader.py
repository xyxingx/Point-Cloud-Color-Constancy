from __future__ import print_function
import os
import random
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd




class PcdColor(data.Dataset):

    """
    Gets point cloud data (pcd)

    Hyperparameters: 
        datasets: which dataset 
        foldn: fold number (0,1,2)
        sizes: size of point cloud (16,64)

    """
 
    def __init__(self, train=True, datasets='ETH3D', foldn=0, sizes=16):
        if datasets =='ETH3D':
            label_path = './folds/ETH3d_folds.csv'
            self.gt_path = '/data/share/xxy/PCCC/pcd_datas/ETH3d/labels/' #Puts your own data path
            self.data_path = '/data/share/xxy/PCCC/pcd_datas/ETH3d/pcd_'
        elif datasets == 'sRGB':
            label_path = './folds/NYU-v2&DIODE_folds.csv'
            self.gt_path = '/data/share/xxy/PCCC/pcd_datas/QXRE/all_label/'
            self.data_path = '/data/share/xxy/PCCC/pcd_datas/QXRE/pcd'
        elif datasets == 'DepthAWB':
            label_path = './folds/Depth_AWB_folds.csv'
            self.gt_path = '/data/share/xxy/PCCC/pcd_datas/DepthAWB/label/'
            self.data_path = '/data/share/xxy/PCCC/pcd_datas/DepthAWB/pcd'

        df = pd.read_csv(label_path)
        df_train = df[df.foldnum != foldn].reset_index(drop=True)
        df_test = df[df.foldnum == foldn].reset_index(drop=True)
        if train:
            self.file_list = df_train.fn.values.tolist()
        else:
            self.file_list = df_test.fn.values.tolist()
        self.train = train
        self.sizes = sizes
        self.sta_aug = True

    def __getitem__(self, index):
        fn = self.file_list[index]
        fnn = fn.strip().split('.')[0]
        pcd = np.load(self.data_path+str(self.sizes)+'/'+fnn+'.npy')
        illums = np.load(self.gt_path+ fnn + '.npy')
        pcd = np.array(pcd, dtype='float32')
        if self.train:
            # pcd_pos = pcd[...,0:2]
            ha = pcd[..., 0]
            da = pcd[..., 2]
            va = pcd[..., 1]
            pcd_new = np.zeros_like(pcd[...,0:3])
            # print(pcd_new.shape)
            pcd_new[..., 0] = da
            pcd_new[..., 1] = ha
            pcd_new[..., 2] = va
            pcd_new = self.RotateZ(pcd_new)
            # pcd_new = self.RotateY(pcd_new)
            # pcd_new = self.RotateX(pcd_new)
            pcd[...,0] = pcd_new[..., 1]
            pcd[..., 1] = pcd_new[..., 2]
            pcd[..., 2] = pcd_new[..., 0]
            if self.sta_aug:
                pcc_new = pcd[...,3:]
                stax = np.random.normal(1,0.2) 
                pcc_new = pcc_new*stax
                pcd[...,3:] =  pcc_new
        pcd = pcd.transpose(1,0)
        illums = np.array(illums, dtype='float32')
        pcd = torch.from_numpy(pcd.copy())
        illums = torch.from_numpy(illums.copy())
        return pcd, illums, fnn
    
    def RotateZ(self,pcd_new):
        angg = random.gauss(0, 30)
        rotate_ang = angg * np.pi / 180.
        rotate_z = np.asarray([np.cos(rotate_ang), -np.sin(rotate_ang), 0,
                                np.sin(rotate_ang), np.cos(rotate_ang), 0,
                                0, 0, 1]).reshape((3, 3))
        pcd_new = np.dot(pcd_new, rotate_z)
        return pcd_new

    def RotateY(self,pcd_new):
        angg = random.gauss(0, 30)
        rotate_ang = angg * np.pi / 180.
        rotate_y = np.asarray([np.cos(rotate_ang), np.sin(rotate_ang), 0,
                                0, 1, 0,
                                -np.sin(rotate_ang), 0, np.cos(rotate_ang)]).reshape((3, 3))
        pcd_new = np.dot(pcd_new, rotate_y)
        return pcd_new

    def RotateX(self,pcd_new):
        angg = random.gauss(0, 30)
        rotate_ang = angg * np.pi / 180.
        rotate_x = np.asarray([1,0 , 0,
                                0, np.cos(rotate_ang), -np.sin(rotate_ang),
                                np.sin(rotate_ang), 0, np.cos(rotate_ang)]).reshape((3, 3))
        pcd_new = np.dot(pcd_new, rotate_x)
        return pcd_new

    def __len__(self):
        return (len(self.file_list))


if __name__ == '__main__':
    dataset = PcdColor(train=True)
    dataload = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=6)
    for ep in range(10):
        time1 = time.time()
        for i, data in enumerate(dataload):
            img, ill, fn = data
            print(img.shape)
            # print(fn)
