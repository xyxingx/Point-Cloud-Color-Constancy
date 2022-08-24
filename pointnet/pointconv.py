"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import sys
import torch.nn.functional as F
sys.path.append('../')
from utils.pointconv_util import PointConvDensitySetAbstraction

class PointConvDensityClsSsg(nn.Module):
    def __init__(self, num_classes1=1,num_classes2=8):
        super(PointConvDensityClsSsg, self).__init__()
        feature_dim = 3
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(256, num_classes1)
        self.fc4 = nn.Linear(256, num_classes2)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x1 = self.fc3(x)
        # x1 = F.log_softmax(x1, -1)
        x2 = self.fc4(x)
        x2 = F.log_softmax(x2, -1)
        return x1,x2

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    input = torch.randn((8,6,2048))
    label = torch.randn(8,16)
    model = PointConvDensityClsSsg(num_classes1=1,num_classes2=8)
    output1,output2= model(input[:,:3,:],input[:,3:,:])
    print(output1.size())
    print(output2.size())
    # print(output)

