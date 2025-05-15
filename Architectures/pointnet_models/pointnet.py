import torch
import torch.nn as nn
from torch_geometric.nn import pool, MessagePassing
from torch_cluster import knn_graph

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.proj = nn.Linear(in_channels, 128, bias=False)
        self.mlp = nn.Sequential(nn.Linear(128 + 3, 2*out_channels), nn.ReLU(), nn.Linear(2*out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        input = pos_j - pos_i  # Compute spatial relation with difference

        if h_j is not None: # check if there is a representation
            h_j = self.proj(h_j)
            input = torch.cat([h_j, input], dim=-1)
        out = self.mlp(input)  # Apply our final MLP.
        
        return out

class PointNet(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=3, scale=2, dropout=0.3, layers=2):
        super().__init__()
        self.conv1 = PointNetLayer(in_channels, d_model)
        self.conv2 = nn.ModuleList( [PointNetLayer((i+1)*d_model, (i+2)*d_model) for i in range(layers)] )
        
        self.drop0 = nn.Dropout(p=dropout)
        self.drop1 = nn.ModuleList( [nn.Dropout(p=dropout) for i in range(layers)] )
        self.drop3 = nn.Dropout(p=dropout)
        
        self.ffw = nn.Sequential(nn.Linear((layers+1)*d_model, scale*d_model), nn.ReLU()) 
        self.classifier = nn.Sequential(nn.Linear(scale*d_model, out_channels), nn.Softmax(dim=0))
        
    def forward(self, x, adj, batch=False):
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        
        h = self.conv1(h=x, pos=x, edge_index=edge_index) # Input layer
        h = self.drop0(h.relu())

        for j, conv in enumerate(self.conv2): # Secundary layers
            h = conv(h=h, pos=x, edge_index=edge_index)
            h = self.drop1[j](h.relu())
        
        h_pool = pool.global_max_pool(h, batch=None) # max pooling
        h_pool = torch.flatten(h_pool) # flattenning

        self.h_ = self.drop3( self.ffw(h_pool) )
        
        return self.classifier(self.h_)
