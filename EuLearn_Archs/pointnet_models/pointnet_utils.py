import torch
import torch.nn as nn

def square_distance(src:torch.tensor, dst:torch.tensor) -> torch.tensor:
    """ Regresa matriz de distancias NxM para N puntos con M por comparar por cada batch"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst** 2, -1).view(B, 1, M)
    return dist


def index_points(points:torch.tensor, idx:int) -> torch.tensor:
    """Regresa puntos indexados según idx"""
    device = points.device # Device
    B = points.shape[0] # Núm de batches
    view_shape = list(idx.shape) # Tamañi de idx
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # Re ajusta tamaños
    new_points = points[batch_indices, idx, :] # toma puntos de idx
    return new_points


def farthest_point_sample(xyz:torch.tensor, npoint:int) -> torch.tensor:
    """ Muestreo de los puntos mejor distribuidos en los datos de la nube de puntos"""
    device = xyz.device # device
    B, N, C = xyz.shape # tamaños Batch; Number of points; Channels
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # Inicia centroides en 0
    distance = torch.ones(B, N).to(device)*1e10 # Inicia distancias con valores altos 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # Inicia farthest de manera aleatoria
    batch_indices = torch.arange(B, dtype=torch.long).to(device) # Índice por batch
    for i in range(npoint):
        centroids[:, i] = farthest # Asigna a los centrodides los puntos farthest actuales
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3) # Asigna centroide con base en farthest
        dist = torch.sum((xyz - centroid)**2, -1) # Obtiene distancias cuadradas
        mask = dist < distance # Enmascara elementos con distancia mayor a la actual
        distance[mask] = dist[mask] # Guarda distancia actual con base en máscara
        farthest = torch.max(distance, -1)[1] # Obtiene farthest con el máximo de las distancias
    return centroids


def query_ball_point(radius:float, nsample:int, xyz:torch.tensor, new_xyz:torch.tensor) -> torch.tensor:
    """ Regresa índices de puntos (tantos como en new_xyz) con nsample agrupados con base a radius"""
    device = xyz.device # device
    B, N, C = xyz.shape # tamaño 
    _, S, _ = new_xyz.shape # número de puntos en new
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz) # Obtiene distancias entre new y los datos de entrada x
    # Agrupa puntos con base a la distancia de un radio^2 determinado
    group_idx[sqrdists > radius**2] = N 
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask] # Retorna esos puntos
    return group_idx


def sample_and_group(npoint:int, radius:float, nsample:int, xyz:torch.tensor, points:torch.tensor, returnfps=False):
    """Regresa sampleo agrupado de los puntos con los métodos anteriores"""
    B, N, C = xyz.shape # tamaño
    S = npoint # S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # Aplica farthest
    new_xyz = index_points(xyz, fps_idx) # Obtiene puntos con índices de farthest
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # Aplica query ball
    grouped_xyz = index_points(xyz, idx) # Regresa puntos con índices anteriores
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # Normaliza restando los farthest points

    if points is not None:
        grouped_points = index_points(points, idx) # Indexa puntos con base en idx
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # Concatena
    else:
        new_points = grouped_xyz_norm # En otro caso, ignora los puntos
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """Samplea y agrupa puntos"""
    device = xyz.device # Device
    B, N, C = xyz.shape # Tamaños
    new_xyz = torch.zeros(B, 1, C).to(device) # inicializa en 0s
    grouped_xyz = xyz.view(B, 1, N, C) # Inicializa agregando dimensión a x
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1) #Concatenta grouped y points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points