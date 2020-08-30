import torch
import torch.nn as nn
import numpy as np


class CombinedLoss(nn.Module):

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, gts, preds, gts_normals, sphere_edges = None):
        
        if sphere_edges != None:
            return self.forward_with_edges(gts, preds, gts_normals, sphere_edges)
        else:
            return self.forward_with_nearest_neighbour(gts, preds, gts_normals)

    def unit(self, tensor):
    	return torch.nn.functional.normalize(tensor, dim=1, p=2) # l_2 normalization


    def forward_with_edges(self, gts, preds, gts_normals, sphere_edges):
        process_colors = False
        if preds.shape[2] > 3:
            process_colors = True

        if self.use_cuda:
            dtype = torch.cuda.LongTensor
            ftype = torch.cuda.FloatTensor
            itype = torch.cuda.IntTensor
        else:
            dtype = torch.LongTensor
            ftype = torch.float64
            itype = torch.int32

        
        # data preparation
        gts_points = gts[:, :, :3] # [2, 4096, 3]
        if process_colors:
            gts_colors = gts[:, :, 3:] # [2, 4096, 3]

        preds_points = preds[:, :, :3] # [2, 4096, 3]
        if process_colors:
            preds_colors = preds[:, :, 3:] # [2, 4096, 3]
        
        color_loss = torch.tensor(0.0).type(ftype)
        edge_loss = torch.tensor(0.0).type(ftype)
        normal_loss = torch.tensor(0.0).type(ftype)
        champfer_loss = torch.tensor(0.0).type(ftype)
        sphere_edges = sphere_edges.type(dtype)

        # ============== color loss ==============

        # ============== edge loss ==============
        edge_start = preds_points[:, sphere_edges[:,0], :]
        edge_end = preds_points[:, sphere_edges[:,1], :]
        edge = torch.sub(edge_start, edge_end)
        edge_length = torch.sum(torch.sqrt(edge))
        edge_loss = torch.mean(edge_length)*3000

        # ============ champfer loss ============
        P = self.batch_pairwise_dist(gts_points, preds_points)
        bt_num_idx = torch.arange(0, P.size(0)).type(dtype)
        dist_first_to_second_0, idx_first_to_second_0 = torch.min(P, 1) 
        dist_second_to_first_0, idx_second_to_first_0 = torch.min(P, 2)

        champfer_loss = torch.sum(dist_first_to_second_0) + torch.sum(dist_second_to_first_0)   

        # ============= normal loss =============
        edge = edge.type(dtype)
        normal = torch.stack([gts_normals[b, torch.squeeze(idx_first_to_second_0[b, :]), : ] for b in bt_num_idx]) # change  idx_first_to_second_0 to idx_second_to_first_0
        normal = normal[:, sphere_edges[:,0], : ] 

        edge = edge.type(ftype)
        normal = normal.type(ftype)
        cosine = torch.abs(torch.sum(torch.matmul(self.unit(normal),torch.transpose(self.unit(edge), dim0 = 1, dim1 = 2))))
        #cosine = torch.abs(torch.sum(torch.matmul(self.unit(normal), self.unit(edge))))
        normal_loss = torch.mean(cosine) * 50000
        
        result = color_loss + champfer_loss + normal_loss*10.0
        
        return result.type(dtype)
                

    def forward_with_nearest_neighbour(self, gts, preds, gts_normals):
        process_colors = False
        if preds.shape[2] > 3:
            process_colors = True

        if self.use_cuda:
            dtype = torch.cuda.LongTensor
            ftype = torch.cuda.FloatTensor
        else:
            dtype = torch.LongTensor
            ftype = torch.float64
        
        gts_points = gts[:, :, :3] # [2, 4096, 3]
        if process_colors:
            gts_colors = gts[:, :, 3:] # [2, 4096, 3]

        preds_points = preds[:, :, :3] # [2, 4096, 3]
        if process_colors:
            preds_colors = preds[:, :, 3:] # [2, 4096, 3]

        # ============ color loss ============
        color_loss = torch.tensor(1.0).type(dtype)

        # ============ champfer loss ============
        #       First               Second              0. 1. 2. 3. - nearest points for  0` caluculated by using champher distance
        #  |^^^^^^^^^^^^^^^^^| |^^^^^^^^^^^^^^^^^| 
        #  |                 | |        2.       |
        #  |                 | |                 |
        #  |      0`*        | |  1.    0.       |
        #  |                 | |                 |
        #  |                 | |        3.       |
        #  |_________________| |_________________| 
        #
        #
        #       First               Second              n - normal for 0`
        #  |^^^^^^^^^^^^^^^^^| |^^^^^^^^^^^^^^^^^|      0.-1. - first edge
        #  |    n \          | |        2.       |
        #  |       \         | |        |        |
        #  |      0`*        | |  1.----0.       |
        #  |                 | |        |        |
        #  |                 | |        3.       |
        #  |_________________| |_________________| 
        P = self.batch_pairwise_dist(gts_points, preds_points)
        
        max_value = torch.tensor(torch.max(P)).type(torch.float)
        bt_num_idx = torch.arange(0, P.size(0)).type(dtype)
        pnt_num_idx = torch.arange(0, P.size(2)).type(dtype)
        
        dist_first_to_second_0, idx_first_to_second_0 = torch.min(P, 1) 
        dist_second_to_first_0, idx_second_to_first_0 = torch.min(P, 2)
        
        # changing value to max 
        for i in bt_num_idx:
            P[i, idx_first_to_second_0[i, :], pnt_num_idx] = max_value
            P[i, pnt_num_idx, idx_second_to_first_0[i, :]] = max_value

        dist_first_to_second_1, idx_first_to_second_1 = torch.min(P, 1) 
        dist_second_to_first_1, idx_second_to_first_1 = torch.min(P, 2) 

        # changing value to max 
        for i in bt_num_idx:
            P[i, idx_first_to_second_1[i, :], pnt_num_idx] = max_value
            P[i, pnt_num_idx, idx_second_to_first_1[i, :]] = max_value

        dist_first_to_second_2, idx_first_to_second_2 = torch.min(P, 1) 
        dist_second_to_first_2, idx_second_to_first_2 = torch.min(P, 2) 

        # changing value to max 
        for i in bt_num_idx:
            P[i, idx_first_to_second_2[i, :], pnt_num_idx] = max_value
            P[i, pnt_num_idx, idx_second_to_first_2[i, :]] = max_value

        dist_first_to_second_3, idx_first_to_second_3 = torch.min(P, 1) 
        dist_second_to_first_3, idx_second_to_first_3 = torch.min(P, 2) 

        champfer_loss = torch.sum(dist_first_to_second_0) + torch.sum(dist_second_to_first_0)

        # ============ normal loss ============
        normal_loss = torch.tensor(0.0).type(ftype)
        for i in bt_num_idx:
            edge_1 = preds_points[i, idx_first_to_second_0[i,:]] - preds_points[i, idx_first_to_second_1[i,:]]
            edge_2 = preds_points[i, idx_first_to_second_0[i,:]] - preds_points[i, idx_first_to_second_2[i,:]]
            edge_3 = preds_points[i, idx_first_to_second_0[i,:]] - preds_points[i, idx_first_to_second_3[i,:]]
            
            normal_loss += torch.abs(torch.sum(edge_1 * gts_normals[i]))
            normal_loss += torch.abs(torch.sum(edge_2 * gts_normals[i]))
            normal_loss += torch.abs(torch.sum(edge_3 * gts_normals[i]))
        

        result = color_loss + champfer_loss + normal_loss*10.0
        
        return result.type(dtype)

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


