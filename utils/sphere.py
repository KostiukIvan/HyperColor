import torch
import numpy as np
import random

class Sphere():
    def __init__(self, config):
        self.coord_path = config['target_network_input']['static_sphere']['sphere_points_path']
        self.edge_path = config['target_network_input']['static_sphere']['sphere_edges_path']
        self.real_sphere_coord = None # (x,y,z)
        self.shaked_sphere_coord = None # (x,y,z)
        self.sphere_edges = None # (x_1, x_2)
        self.config = config

    def load(self):
        if self.real_sphere_coord != None:
            print("Sfera juz jest zaciagnieta!!!")
            return 
        
        if self.sphere_edges != None:
            print("Edge juz sa zaciagniete!!!")
            return
        
        self.real_sphere_coord = np.fromfile(self.coord_path, float, -1, ' ')

        if len(self.real_sphere_coord) != self.config['n_points'] * 3:
            print("Error, probaly objects number of points is not equal to sphere number of points")
            return

        self.real_sphere_coord = self.real_sphere_coord.reshape(self.config['n_points'], 3) # (self.config['n_points'], 3)
        self.sphere_edges = np.fromfile(self.edge_path, int, -1, ' ')
        self.sphere_edges = self.sphere_edges.reshape(int(self.sphere_edges.size/2), 2)

        self.real_sphere_coord = torch.from_numpy(self.real_sphere_coord).float()
        self.sphere_edges = torch.from_numpy(self.sphere_edges).int()
        self.shaked_sphere_coord = self.real_sphere_coord

    
    def get_sphere_coord(self):
        if self.real_sphere_coord == None:
            self.load()

        if self.config['target_network_input']['static_sphere']['shake']['enable']:
            return self.shake()
        
        return self.real_sphere_coord
    
    def get_sphere_edges(self):
        if self.sphere_edges == None:
            self.load()
        
        return self.sphere_edges

    def shake(self):
        r = 1 # radius
        z_coord = self.real_sphere_coord[:,2]
        delta = self.config['target_network_input']['static_sphere']['shake']['delta']
        self.shaked_sphere_coord = torch.zeros([self.config['n_points'], 2])
    
        # setting x and y coordinate with some noise
        self.shaked_sphere_coord = self.real_sphere_coord[:,0:2].add( torch.zeros([self.config['n_points'], 2]).uniform_(-delta, delta) )
        
        # preventing lenth of pouint higher than radius
        self.shaked_sphere_coord[self.shaked_sphere_coord > 1] = 1 - delta
        self.shaked_sphere_coord[self.shaked_sphere_coord < -1] = -1 + delta

        z_coord_rec = (r * torch.ones([self.config['n_points']]))**2 - self.shaked_sphere_coord[:,0]**2 - self.shaked_sphere_coord[:,1]**2
        z_coord_rec[z_coord_rec < 0] = delta
        z_coord_rec = torch.sqrt(z_coord_rec)
        z_coord_rec[z_coord < 0] = -z_coord_rec[z_coord < 0]
        z_coord_rec = z_coord_rec.reshape(self.config['n_points'],1)  
                                                  
        self.shaked_sphere_coord = torch.cat((self.shaked_sphere_coord, z_coord_rec),1)

        # preventing lenth of points higher than radius
        self.shaked_sphere_coord[self.shaked_sphere_coord > 1] = 1 - delta
        self.shaked_sphere_coord[self.shaked_sphere_coord < -1] = -1 + delta
 
        return self.shaked_sphere_coord

