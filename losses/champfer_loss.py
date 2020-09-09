import torch
import torch.nn as nn
import numpy as np


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        # pred_points, pred_colors = preds[:, :, :3], preds[:, :, 3:]
        # true_points, true_colors = gts[:, :, :3], gts[:, :, 3:]

        #import pdb; pdb.set_trace()

        pred_points = preds
        true_points = gts

        # champfer loss
        P = self.batch_pairwise_dist(true_points, pred_points)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        champfer_loss = loss_1 + loss_2

        del P
        del mins
        del loss_1
        del loss_2

        # #normal loss
        # # 1) - find 3 nearest neighbours
        # generated_points_distances = self.batch_pairwise_dist(pred_points, pred_points)

        # sorted, indicies = generated_points_distances, torch.cuda.LongTensor()
        # torch.sort(generated_points_distances, dim=1, out=(sorted, indicies))

        # del sorted
        # del generated_points_distances

        # neighbours_indicies = indicies[1:num_neighbours + 1]

        # del indicies

        # # 2) find mapping (generated, neighbour[i]) -> normal to nearest true of generated
        # #   2.1) generated -> normal to nearest true
        # normals_to_nearest_true = torch.gather(input=normals, dim=0, index=nearest_true_indicies)
        # #   2.2) arange mapping (generated, neighbour[i]) -> normal
        # generated_neighbour_to_normal_mapping = torch.gather(normals_to_nearest_true, dim=0, index=torch.LongTensor(np.repeat(np.arange(num_points), num_neighbours)))
        # #   2.3) create aranged normals
        # aranged_normals = torch.arange(input=normals, dim=1, index=generated_neighbour_to_normal_mapping)

        # # 3) find differences (generated - neighbour[i])
        # generated_to_neighbour_mapping = [neighbours_indicies[point_ind][neighbour_ind] for point_ind in range(num_points) for neighbour_ind in range(num_neighbours)]
        # differences = torch.gather(input=pred_points, dim=1, index=torch.LongTensor(np.repeat(np.arange(num_points), num_neighbours))) - torch.gather(input=pred_points, dim=1, index=torch.LongTensor(np.repeat(generated_to_neighbour_mapping)))

        # cosine = torch.abs(torch.sum(torch.mul(differences.transpose(2,1), aranged_normals), dim=1))
        # normal_loss = torch.mean(cosine) * 0.5

        # #color loss
        # # TODO: implement distance between RGB colors batchwise
        # color_loss = 0

        return champfer_loss

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
