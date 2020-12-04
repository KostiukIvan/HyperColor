import torch
import torch.nn as nn
import numpy as np
import sys
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency, point_mesh_edge_distance, point_mesh_face_distance
from utils.util import CombinedLossType

class CombinedLoss(nn.Module):

    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.count = 0
        self.config = config
        self.colors_alpha = 1
        #self.log = log

        
    def forward(self, gts_X, pred_X, train_colors):
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
            ftype = torch.cuda.FloatTensor
        else:
            dtype = torch.LongTensor
            ftype = torch.float32

        loss = torch.tensor(0.0).type(ftype)
        if train_colors: # [2, 4096, 6]
            '''
            champher_loss, _ = chamfer_distance(gts_X, pred_X)
            loss +=  champher_loss * 900
            '''
            # gts_colors = gts_X[:, :, 3:6].type(ftype) # [2, 4096, 3]
            # preds_colors = pred_X[:, :, 3:6].type(ftype) # [2, 4096, 3]
            origin_colors= pred_X[:, :, 3:6].type(ftype)
            pred_colors = pred_X[:, :, :3].type(ftype)

            MSE = torch.nn.MSELoss()
            colors_loss = MSE(origin_colors, pred_colors)
            loss +=  colors_loss 

        else: # [2, 4096, 3]
            gts_points = gts_X[:, :, :3].type(ftype) # [2, 4096, 3]
            preds_points = pred_X[:, :, :3].type(ftype) # [2, 4096, 3]
            champher_loss, _ = chamfer_distance(gts_points, preds_points)
            loss +=  champher_loss * 900

        return loss

        

       

