import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_edge_distance, chamfer_distance, mesh_edge_loss
from utils.util import CombinedLossType

class CombinedLoss(nn.Module):

    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.count = 0
        self.config = config
    def forward(self, gts_X, pred_X, gts_normals, S_mesh = None, change_loss_func = False):

        if change_loss_func:
            mesh = Meshes(verts=[b for b in pred_X], \
                            faces=list(map(lambda x: x[1] ,(map(lambda x: x.get_mesh_verts_faces(0), S_mesh))))) # x[1] = face

            points = Pointclouds(points = [g for g in gts_X], normals = [g_n for g_n in gts_normals])

            return point_mesh_edge_distance(mesh, points)
        else:
            loss, _ = chamfer_distance(gts_X, pred_X)
            return loss

    def forward_(self, gts_X, pred_X, gts_normals, S_mesh = None, change_loss_func = False):
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
            ftype = torch.cuda.FloatTensor
        else:
            dtype = torch.LongTensor
            ftype = torch.float32

        # data preparation
        gts_points = gts_X[:, :, :3].type(ftype) # [2, 4096, 3]
        preds_points = pred_X[:, :, :3].type(ftype) # [2, 4096, 3]
        loss = torch.tensor(0.0).type(ftype)


        if change_loss_func:
            if self.config['target_network_input']['loss']['change_to']['champher']:
                pass
            if self.config['target_network_input']['loss']['change_to']['colors']:
                gts_colors = gts_X[:, :, 3:].type(ftype) 
                preds_colors = pred_X[:, :, 3:].type(ftype) 
                pass
            if self.config['target_network_input']['loss']['change_to']['edges']:
                pass
            if self.config['target_network_input']['loss']['change_to']['normals']:
                pass
            
           
        else:
            if self.config['target_network_input']['loss']['default']['champher']:
                pass
            if self.config['target_network_input']['loss']['default']['colors']:
                gts_colors = gts_X[:, :, 3:].type(ftype) 
                preds_colors = pred_X[:, :, 3:].type(ftype) 
                pass
            if self.config['target_network_input']['loss']['default']['edges']:
                pass
            if self.config['target_network_input']['loss']['default']['normals']:
                pass
            
          


    def _champher_loss():
        pass
    def _color_loss():
        pass
    def _edges_and_normals_loss():
        pass

