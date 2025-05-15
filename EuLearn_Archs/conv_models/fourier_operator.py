import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import pool
from utils import LayerNorm

class FeedForward(nn.Module):
	def __init__(self, in_size, out_size, units=100, p=0.1):
		super(FeedForward, self).__init__()
		self.h = nn.Sequential(nn.Linear(in_size, units), nn.Tanh())
		self.out =  nn.Linear(units, out_size)
		self.drop = nn.Dropout(p=p)
	def forward(self, x):
		h = self.h(x)
		h = self.drop(h)
		return self.out(h)

class FourierConv(nn.Module):
	def __init__(self,in_channels, out_channels, max_f=1):
		super(FourierConv, self).__init__()
		self.R = nn.Parameter( torch.rand(in_channels,out_channels,dtype=torch.cfloat) )
		
	def forward(self, x):
		Fv = torch.fft.fftn(x)
		RFv = torch.matmul(Fv,self.R)
		
		return torch.fft.irfftn(RFv, x.size())

class FourierBlock(nn.Module):
	def __init__(self,in_channels, out_channels):
		super(FourierBlock,self).__init__()
		self.conv = FourierConv(in_channels, out_channels)
		self.linear = nn.Linear(in_channels, out_channels)
		self.gelu = nn.GELU()
		
	def forward(self,x):
		v_l = self.linear(x)
		v_f = self.conv(x)
		h = self.gelu(v_l + v_f)
		
		return h #self.Q(h)
	
class FNOLayer(nn.Module):
	"""Fourier Neural Operator Layer"""
	def __init__(self, da, du, dv, dropout=0.5):
		super(FNOLayer, self).__init__()
		self.P = FeedForward(da,dv) #nn.Linear(da, dv)
		self.Q = FeedForward(dv,du) #nn.Linear(dv, du)
		self.four1 = FourierBlock(dv, dv)
		self.four2 = FourierBlock(dv, dv)
		self.four3 = FourierBlock(dv, dv)
		self.drop1 = nn.Dropout(p=dropout)
		self.drop2 = nn.Dropout(p=dropout)
		self.drop3 = nn.Dropout(p=dropout)
		
	def forward(self, x):
		v = self.P(x)
		h = self.four1(v)
		h = self.drop1(h)
		h = self.four2(h)
		h = self.drop2(h)
		h = self.four3(h)
		h = self.drop3(h)
		u = self.Q(h)
		
		return u

class FNO(nn.Module):
    def __init__(self, input_size=(71,71), output_size=2, d_model=64, dropout=0.5, scale=2):
        super(FNO, self).__init__()
        self.fft = FNOLayer(input_size[0], input_size[1], d_model) # FNO Layers
        self.fft2 = FNOLayer(input_size[0], input_size[1], d_model) 
        self.norm = LayerNorm(input_size[1]) # Normalized layers
        self.norm2 = LayerNorm(input_size[1])
        self.drop0 = nn.Dropout(p=dropout) # Dropouts
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)
        self.ffw1 = nn.Sequential(nn.Linear(d_model, scale*d_model), nn.ReLU()) # Output layers
        self.ffw2 = nn.Sequential(nn.Linear(scale*d_model, output_size), nn.Softmax(0))
    
    def forward(self, x):  
        # First FNO layer
        self.h1 = self.fft(x)
        h1 = self.norm(self.h1)
        h1 = self.drop0(h1)
        h1 = pool.global_mean_pool(h1, batch=None)
        # Second FNO layer
        self.h = self.fft2(h1)
        h = self.norm2(self.h)
        h = self.drop1(h)
        h = pool.global_max_pool(h, batch=None)
        # Pooling
        h_pool = torch.flatten(h)
        h_pool = self.drop2(h_pool)
        # Output layers
        self.h_ = self.ffw1(h_pool)
        self.h_ = self.drop3(self.h_)
        out = self.ffw2(self.h_)

        return out