import torch
import numpy as np
import trimesh
import pickle
from glob import glob
from networkx import adjacency_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_field(genus, clase, permute, size):
    """Generate point cloud from dataset"""
    x, y = [], []
    if permute:
        files = np.random.permutation( glob(genus+'/*_sf.txt')[:size] )
    else:
        files = glob(genus+'/*_sf.txt')
    for filename in files: # Open files with suffix _sf, i.e. scalar fields.
        field = np.loadtxt(filename)
        aux_file=open(filename,'r')
        grid_size=eval(aux_file.readline().split(":")[1])
        aux_file.close()
        field = field.reshape(grid_size)
        x.append( torch.Tensor(field).to(device) )
        y.append(clase)
        
    return x, torch.tensor(y).to(device)

def get_point_cloud(genus, clase, permute=False, size=100):
    x, y = [], []
    if permute:
        files = np.random.permutation( glob(genus+'/*.stl')[:size] )
    else:
        files = glob(genus+'/*.stl')
    for filename in files:
        knot = trimesh.load_mesh(filename)
        adj = torch.Tensor( adjacency_matrix(knot.vertex_adjacency_graph).todense() ).to(device)
        vertex = torch.Tensor( knot.vertices ).to(device)
        x.append( (vertex, adj) )
        y.append(clase)
       
    return x,torch.tensor(y).to(device)

def get_data(directories, clases, permute, size, type):
    x = []
    y = []
    for dir_x,genus in zip(directories, clases):
        if type == 'sc_field':
            xi, yi  = get_field(dir_x, genus, permute, size)
        elif type == 'pointcloud':
            xi, yi  = get_point_cloud(dir_x, genus, permute, size)
        else:
            print('Not valid data type')
        x += xi
        y += [yi]
        
    return x, torch.cat(tuple(y)).to(device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directories, clases=[0,1,2], permute=False, size=300, type='sc_field'):
        self.x, self.y = get_data(directories, clases, permute, size, type)

    def __len__(self): #size
        return len(self.x)

    def __getitem__(self, idx): #get an item of the data set
        return self.x[idx], self.y[idx]
        

def get_surfaces(folder, genus, size):
    if size == False:
    	files = glob(folder+'/*.stl')
    else:	
    	files = glob(folder+'/*.stl')[:size]

    x,y = [], []
    for file in files:
        with open(file, 'rb') as f: 
            x_r, adj_r = pickle.load(f) 
            x.append((torch.tensor(x_r, dtype=torch.float32).to(device), torch.Tensor(adj_r).to(device)))
            y.append(genus)
            
    return x, y
    
def get_sampled_pointclouds(directory, size=10):
	x = []
	y = []
	for subdir in glob(directory+"/*"):
		xi, yi = get_surfaces(subdir, genus=int(subdir[-1]), size=size)
		x += xi
		y += yi
		
	return x, torch.tensor(y).to(device)
