import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer add and normalization"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class NoamOptimizer:
    def __init__(self, parameters, d_model, warmup=40000, lr=0, eps=1e-9, decay=0.01, betas=(0.9, 0.99)):
        #optimizer
        self.optimizer = torch.optim.Adam(parameters, lr=lr, betas=betas, eps=eps, weight_decay=decay)
        self._step = 0
        self.warmup = warmup
        self.model_size = d_model
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self):
        step = self._step
        lr_step = self.model_size**(-0.5) * min(step**(-0.5), step*self.warmup**(-1.5))
        return lr_step

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def visualize(data, adjacency):
    import plotly.graph_objects as go
    from networkx import from_numpy_array
    
    G = from_numpy_array(adjacency.numpy())
    x_edges=[]
    y_edges=[]
    z_edges=[]
    for edge in G.edges:
        x_coords = [data[edge[0]][0],data[edge[1]][0],None]
        x_edges += x_coords
        y_coords = [data[edge[0]][1],data[edge[1]][1],None]
        y_edges += y_coords
        z_coords = [data[edge[0]][2],data[edge[1]][2],None]
        z_edges += z_coords

    trace_edges = go.Scatter3d(x=x_edges, y=y_edges, z=z_edges,mode='lines',
                               line=dict(color='black', width=2),hoverinfo='none')
    trace_nodes = go.Scatter3d(x=data[:,0], y=data[:,1], z=data[:,2],mode='markers',
                               marker=dict(symbol='circle',size=2,colorscale=['lightgreen','magenta'],line=dict(color='black', width=0.5)),
                               hoverinfo='text')

    data = [trace_edges, trace_nodes]
    fig = go.Figure(data=data)
    fig.show()
