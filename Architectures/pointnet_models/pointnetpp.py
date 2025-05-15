import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet_utils import *
from pickle import load

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channel=3, mlp=[64,128], group_all=False, npoint=256, radius=10, nsample=64):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint # Valor de puntos agrupados
        self.radius = radius # Radio del muestreo
        self.nsample = nsample # Tamaño de muestra
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.in_channel = in_channel
        last_channel = in_channel # Canales de salida
        for out_channel in mlp: # Crea capas de convolución
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1)) # Capa convolución
            self.mlp_bns.append(nn.BatchNorm2d(out_channel)) # normalización de batches
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        # Revisa si es agrupamiento final o intermedio
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            new_points = new_points[:,:,:,:self.in_channel]

        # Aplica convoluciones y normalización
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        # Regresa la permutación y obtiene agragación máx
        new_points = new_points.permute(0, 3, 2, 1)
        new_points = torch.max(new_points, 2)[0]

        return new_xyz, new_points

class PointNetPlusPlus(nn.Module):
    def __init__(self, num_class, dim=128, scale=2, dropout=0.4, normal_channel=False):
        super(PointNetPlusPlus, self).__init__()
        in_channel = 3
        self.normal_channel = normal_channel
        self.dim = dim
        self.sa1 = PointNetSetAbstraction()
        self.sa2 = PointNetSetAbstraction(in_channel=dim)
        self.sa3 = PointNetSetAbstraction(in_channel=dim+3, group_all=True)
        self.ffw1 = nn.Sequential(nn.Linear(dim, scale*dim), nn.ReLU())
        self.drop1 = nn.Dropout(p=dropout)
        self.ffw2 = nn.Sequential(nn.Linear(scale*dim, num_class), nn.Softmax(1))

    def forward(self, xyz):
        B, _, _ = xyz.shape
        # Capas PintNet++
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Classificación por FFW
        x = l3_points.view(B, self.dim)
        h1 = self.drop1(self.ffw1(x))
        out = self.ffw2(h1)
        
        return out[0]


def get_data(file, device='cpu'):
    x, _ = load(open(file, 'rb')) 
    x = torch.tensor(x, dtype=torch.float32).to(device)
    N,C = x.shape
    
    return x.view(1,N,C)

def get_dataset(directory, device='cpu'):
    x = []
    y = []
    for subdir in glob(directory+"/*"):
        files = glob(subdir+'/*')
        x += files
        y += [int(subdir[-1])]*len(files)
		
    return x, torch.tensor(y).to(device)