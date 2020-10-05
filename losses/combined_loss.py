import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency, point_mesh_edge_distance, point_mesh_face_distance
from utils.util import CombinedLossType

class CombinedLoss(nn.Module):

    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.count = 0
        self.config = config

        
    def forward(self, gts_X, pred_X, gts_normals, S_mesh = None, change_loss_func = False):
        losses = []

        if change_loss_func:
            if self.config['target_network_input']['loss']['change_to']['chamfer_distance']:
                losses.append(CombinedLossType.chamfer_distance)

            if self.config['target_network_input']['loss']['change_to']['mesh_edge_loss']:
                losses.append(CombinedLossType.mesh_edge_loss)

            if self.config['target_network_input']['loss']['change_to']['mesh_laplacian_smoothing']:
                losses.append(CombinedLossType.mesh_laplacian_smoothing)

            if self.config['target_network_input']['loss']['change_to']['mesh_normal_consistency']:
                losses.append(CombinedLossType.mesh_normal_consistency)

            if self.config['target_network_input']['loss']['change_to']['point_mesh_edge_distance']:
                losses.append(CombinedLossType.point_mesh_edge_distance)

            if self.config['target_network_input']['loss']['change_to']['point_mesh_face_distance']:
                losses.append(CombinedLossType.point_mesh_face_distance)

            if self.config['target_network_input']['loss']['change_to']['colors']:
                losses.append(CombinedLossType.colors)
        else:
            if self.config['target_network_input']['loss']['default']['chamfer_distance']:
                losses.append(CombinedLossType.chamfer_distance)

            if self.config['target_network_input']['loss']['default']['mesh_edge_loss']:
                losses.append(CombinedLossType.mesh_edge_loss)

            if self.config['target_network_input']['loss']['default']['mesh_laplacian_smoothing']:
                losses.append(CombinedLossType.mesh_laplacian_smoothing)

            if self.config['target_network_input']['loss']['default']['mesh_normal_consistency']:
                losses.append(CombinedLossType.mesh_normal_consistency)

            if self.config['target_network_input']['loss']['default']['point_mesh_edge_distance']:
                losses.append(CombinedLossType.point_mesh_edge_distance)

            if self.config['target_network_input']['loss']['default']['point_mesh_face_distance']:
                losses.append(CombinedLossType.point_mesh_face_distance)

            if self.config['target_network_input']['loss']['default']['colors']:
                losses.append(CombinedLossType.colors)

        return self.forward_with_spec_losses(gts_X, pred_X, gts_normals, S_mesh, losses)



    def forward_with_spec_losses(self, gts_X, pred_X, gts_normals, S_mesh = None, losses = []):
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
            ftype = torch.cuda.FloatTensor
        else:
            dtype = torch.LongTensor
            ftype = torch.float32

        if losses == []:
            print("You didn't choose any loss function!!!")
            return torch.tensor(0.0).type(ftype)

        gts_points = gts_X[:, :, :3].type(ftype) # [2, 4096, 3]
        preds_points = pred_X[:, :, :3].type(ftype) # [2, 4096, 3]

        loss = torch.tensor(0.0).type(ftype)

        if CombinedLossType.chamfer_distance in losses:
            champher_loss, _ = chamfer_distance(gts_points, preds_points)
            loss += champher_loss*2000

        if ([CombinedLossType.mesh_edge_loss, CombinedLossType.mesh_laplacian_smoothing, CombinedLossType.mesh_normal_consistency, 
                CombinedLossType.point_mesh_edge_distance, CombinedLossType.point_mesh_face_distance ] or losses) != []:

            pred_meshes = Meshes(verts=[b for b in preds_points], \
                            faces=list(map(lambda x: x[1], (map(lambda x: x.get_mesh_verts_faces(0), S_mesh))))) # x[1] = face

            gts_point_clouds = Pointclouds(points = [g for g in gts_X], normals = [g_n for g_n in gts_normals])

            if CombinedLossType.mesh_edge_loss in losses:
                loss += mesh_edge_loss(pred_meshes)

            if CombinedLossType.mesh_laplacian_smoothing in losses:
                loss += mesh_laplacian_smoothing(pred_meshes)
            
            if CombinedLossType.mesh_normal_consistency in losses:
                loss += mesh_normal_consistency(pred_meshes)

            if CombinedLossType.point_mesh_edge_distance in losses:
                loss += point_mesh_edge_distance(pred_meshes, gts_point_clouds) 

            if CombinedLossType.point_mesh_face_distance in losses:
                loss += point_mesh_face_distance(pred_meshes, gts_point_clouds) 

        return loss




    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1)) # (bs, num_points_x, num_points_x)
        yy = torch.bmm(y, y.transpose(2, 1)) # (bs, num_points_y, num_points_y)
        zz = torch.bmm(x, y.transpose(2, 1)) # (bs, num_points_x, num_points_y)
        ########################################
        # torch.bmm((a,b,c), (a,b,c).transpose(2,1))
        # torch.bmm((a,b,c), (a,c,b)) = (a, b, b)
        ########################################
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))                                     # (bs, num_points_y, num_points_x)

        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz) # (bs, num_points_x, num_points_y)
        P = rx.transpose(2, 1) + ry - 2 * zz # (bs, num_points_x, num_points_y) + (bs, num_points_x, num_points_y) - 2 * (bs, num_points_x, num_points_y)
        return P
