import torch
import numpy as np

class Sphere():
    def __init__(self, config):
        self.coord_path = config['target_network_input']['sphere_points_path']
        self.edge_path = config['target_network_input']['sphere_edges_path']
        self.sphere_coord = None # (x,y,z)
        self.sphere_edges = None # (x_1, x_2)

    def load_object(self):
        if self.sphere_coord != None:
            print("Sfera juz jest zaciagnieta!!!")
            return 
        
        if self.sphere_edges != None:
            print("Edge juz sa zaciagniete!!!")
            return
        
        self.sphere_coord = np.fromfile(self.coord_path, float, -1, ' ')
        self.sphere_coord = self.sphere_coord.reshape(4096, 3)
        self.sphere_edges = np.fromfile(self.edge_path, int, -1, ' ')
        self.sphere_edges = self.sphere_edges.reshape(int(self.sphere_edges.size/2), 2)

    
    def get_sphere_coord(self):
        if self.sphere_coord == None:
            self.load_object()
        
        return torch.from_numpy(self.sphere_coord).float()
    
    def get_sphere_edges(self):
        if self.sphere_edges == None:
            self.load_object()
        
        return torch.from_numpy(self.sphere_edges).int()