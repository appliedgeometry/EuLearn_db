import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx


def get_graph_feature(x, k=20, idx=None, device='cpu'):
    batch_size = 1 #x.size(0)
    num_points = x.size(1)
    x = x.view(batch_size, -1, num_points)
    #print(x.shape)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device(device)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class DGCNN(nn.Module):
    def __init__(self, k, emb_dims=512, dropout=0.3, output_channels=11):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256,emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=dropout)
        self.out = nn.Sequential( nn.Linear(256, output_channels), nn.Softmax(1) )

    def forward(self, x):
        batch_size = 1 #x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k).transpose(1,2)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k).transpose(1,2)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x3, k=self.k).transpose(1,2)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        x = self.conv5(x4)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu( (self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu( (self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.out(x)
        
        return x