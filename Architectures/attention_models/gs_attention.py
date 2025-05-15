import torch
import torch.nn as nn
from utils import LayerNorm
from torch_geometric.nn import pool
from math import sqrt    
    
class Attention(nn.Module):
    """Common attention module"""
    def __init__(self, input_size, d_model, bias=True):
        super(Attention, self).__init__()
        self.V  = nn.Linear(input_size, d_model, bias=bias)
        
        self.d_model = d_model
        
    def forward(self, x, adj):       
        value = self.V(x)
        
        scores = torch.matmul(x, x.T)/sqrt(self.d_model)
        scores = scores.masked_fill(adj == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim = -1)
        Vs = torch.matmul(p_attn, value)
        
        return Vs, p_attn    
        
        
class GraphAttention(nn.Module):
    """Complete model"""
    def __init__(self, input_size=3, output_size=3, d_model=128, dropout=0.5, scale=3):
        super(GraphAttention, self).__init__()
        self.att = Attention(input_size, d_model)
        self.drop0 = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        self.att2 = Attention(d_model, 2*d_model)
        self.drop2 = nn.Dropout(p=dropout)
        self.norm2 = LayerNorm(2*d_model)
        
        self.ffw1 = nn.Sequential(nn.Linear(2*d_model, scale*d_model), nn.ReLU())
        self.drop1 = nn.Dropout(p=dropout)
        self.ffw2 = nn.Sequential(nn.Linear(scale*d_model,output_size), nn.Softmax(0))
        
    def forward(self, x, adj):
        h_att, self.att_matrix = self.att(x, adj)
        h1 = self.norm(h_att)
        h1 = self.drop0(h1.relu())
        
        h_att2, self.att_matrix2 = self.att2(h1, adj)
        h = self.norm2(h_att2)
        h = self.drop0(h)
        
        h = pool.global_max_pool(h, batch=None)
        self.h_ = torch.flatten(h, start_dim=0)
        
        h1 = self.ffw1(self.h_)
        h1 = self.drop1(h1)
        out = self.ffw2(h1)       
        
        return out
