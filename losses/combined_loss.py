import torch
from torch import nn
from torch.nn.functional import normalize
import logging

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.logger = logging.getLogger('aae')

    def forward(self, preds, gts, normals, edges):
        #import pdb; pdb.set_trace()
        pred_points, true_points = preds, gts

        # champfer loss
        P = self.batch_pairwise_dist(true_points, pred_points)
        mins, _ = torch.min(P, 1)
        # NOTE - using mean instead of sum, otherwise loss parts have too high amplitude
        loss_1 = torch.mean(mins)
        sum = torch.sum(mins)
        mins, nearest_true_indicies = torch.min(P, 2)
        loss_2 = torch.mean(mins)
        sum += torch.sum(mins)
        champfer_loss = loss_1 + loss_2
        champfer_sum = sum

        #import pdb; pdb.set_trace()

        #egde loss
        edge_starts = pred_points[:, edges[:, 0], :]
        edge_ends = pred_points[:, edges[:, 1], :]
        edge_vectors = torch.sub(edge_starts, edge_ends)
        edge_vectors_squared = torch.sum(torch.mul(edge_vectors, edge_vectors), dim=2)

        edge_loss = torch.mean(edge_vectors_squared)


        #import pdb; pdb.set_trace()
        #normal loss
        # not working #normals_to_nearest_true = normals.gather(2, nearest_true_indicies)
        # not working #normals_to_nearest_true = normals[:, nearest_true_indicies, :]
        normals_to_nearest_true = torch.stack([normals[batch_index, nearest_true_indicies[batch_index], :] for batch_index in range(normals.shape[0])])
        # points that are connected via edge are considered as neighbours
        normals_to_point_neighbour = normals_to_nearest_true[:, edges[:, 0], :]
        point_neighbour_vectors = edge_vectors

        normalized_normals = normalize(input=normals_to_point_neighbour, dim=1, p=2).type(torch.cuda.DoubleTensor)
        normalized_point_neighbour_vectors = normalize(input=point_neighbour_vectors, dim=1, p=2).type(torch.cuda.DoubleTensor)

        #computing cosine of angle between vectors (point, neighbour) and normal to that point
        cosines = torch.abs(torch.sum(torch.mul(normalized_normals, normalized_point_neighbour_vectors), dim=2))

        normal_cosine_loss = torch.mean(cosines)

        #import pdb; pdb.set_trace()
        # scaling loss parts, so they have similar impact
        champfer_loss *= 30_000
        edge_loss *= 240
        normal_cosine_loss *= 200_000

        info = f'champfer - {champfer_loss}, edge - {edge_loss}, normal - {normal_cosine_loss}'
        self.logger.info(info)
        print(info)

        return champfer_loss + edge_loss + normal_cosine_loss

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P