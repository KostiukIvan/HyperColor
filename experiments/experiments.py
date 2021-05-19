import argparse
import json
import logging
import pickle
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from models import aae

from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed, get_weights_dir
from utils.points import generate_points

from sklearn.neighbors import KNeighborsClassifier
import skimage.color as colors
import pandas as pd
import math

cudnn.benchmark = True


def main(config):
    set_seed(config['seed'])

    results_dir = prepare_results_dir(config, config['arch'], 'experiments',
                                      dirs_to_create=['interpolations', 'sphere', 'points_interpolation',
                                                      'different_number_points', 'fixed', 'reconstruction',
                                                      'sphere_triangles', 'sphere_triangles_interpolation'])
    weights_path = get_weights_dir(config)
    epoch = find_latest_epoch(weights_path)

    if not epoch:
        print("Invalid 'weights_path' in configuration")
        exit(1)

    setup_logging(results_dir)
    global log
    log = logging.getLogger('aae')

    if not exists(join(results_dir, 'experiment_config.json')):
        with open(join(results_dir, 'experiment_config.json'), mode='w') as f:
            json.dump(config, f)

    device = cuda_setup(config['cuda'], config['gpu'])
    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
    elif dataset_name == "custom":
        # import pdb; pdb.set_trace()
        from datasets.customDataset import CustomDataset
        dataset = CustomDataset(root_dir=config['data_dir'],
                                classes=config['classes'], split='test', config=config)
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']), len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=12, drop_last=True,
                                   pin_memory=True)

    #
    # Models
    #
    hyper_network_p = aae.PointsHyperNetwork(config, device).to(device)
    hyper_network_cp = aae.ColorsAndPointsHyperNetwork(config, device).to(device)
    encoder_p = aae.PointsEncoder(config).to(device)
    encoder_cp = aae.ColorsAndPointsEncoder(config).to(device)

    if config['reconstruction_loss'].lower() == 'chamfer':
        from losses.champfer_loss import ChamferLoss
        reconstruction_loss = ChamferLoss().to(device)
    elif config['reconstruction_loss'].lower() == 'earth_mover':
        # from utils.metrics import earth_mover_distance
        # reconstruction_loss = earth_mover_distance
        from losses.earth_mover_distance import EMD
        reconstruction_loss = EMD().to(device)
    elif config['reconstruction_loss'].lower() == 'combined':
        from losses.combined_loss import CombinedLoss
        reconstruction_loss = CombinedLoss(experiment=True).to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    log.info("Weights for epoch: %s" % epoch)

    log.info("Loading weights...")
    hyper_network_p.load_state_dict(torch.load(
        join(weights_path, f'{epoch:05}_G_P.pth')))
    hyper_network_cp.load_state_dict(torch.load(
        join(weights_path, f'{epoch:05}_G_CP.pth')))

    encoder_p.load_state_dict(torch.load(
        join(weights_path, f'{epoch:05}_E_P.pth')))
    encoder_cp.load_state_dict(torch.load(
        join(weights_path, f'{epoch:05}_E_CP.pth')))
   

    hyper_network_p.eval()
    hyper_network_cp.eval()
    encoder_p.eval()
    encoder_cp.eval()

    total_loss_colors = []
    total_loss_points = []
    total_loss_colors_encoder = []
    total_loss_points_encoder = []
    total_codes_p = []
    total_codes_cp = []
    x = []

    with torch.no_grad():
        for i, point_data in enumerate(points_dataloader, 0):
            #if i > 50:
            #  break
            
            if dataset_name == "custom":
                X = torch.cat((point_data['points'], point_data['colors']), dim=2)
                X = X.to(device, dtype=torch.float)

            else: 
                X, _ = point_data
                X = X.to(device, dtype=torch.float)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3 or X.size(-1) == 6 or X.size(-1) == 7:
                X.transpose_(X.dim() - 2, X.dim() - 1)
            
            x.append(X)
            codes_p, mu_p, logvar_p = encoder_p(X[:,:3,:])
            codes_cp, mu_cp, logvar_cp = encoder_cp(X)
            total_codes_p.append(codes_p)
            total_codes_cp.append(codes_cp)

            # test
            ########################################################################################

            ########################################################################################
            target_networks_weights_p = hyper_network_p(codes_p)
            target_networks_weights_cp = hyper_network_cp(codes_cp)
            
            X_rec = torch.zeros(torch.cat([X, X[:,:3,:]], dim=1).shape).to(device) # [b, 9, 4096]
            for j, target_network_weights in enumerate(zip(target_networks_weights_p, target_networks_weights_cp)):

                target_network_p = aae.TargetNetwork(config, target_network_weights[0]).to(device)
                target_network_cp = aae.ColorsAndPointsTargetNetwork(config, target_network_weights[1]).to(device)

                if not config['target_network_input']['constant'] or target_network_input is None:     
                    target_network_input = generate_points(config=config, epoch=epoch, size=(X.shape[2], 3)).to(device)

                pred_points = target_network_p(target_network_input.to(device, dtype=torch.float)) # [4096, 3]
                pred_colors = target_network_cp(target_network_input.to(device, dtype=torch.float)) # [4096, 3]

                points_kneighbors = pred_points
                clf = KNeighborsClassifier(1)
                clf.fit(torch.transpose(X[j][:3], 0 , 1).cpu().numpy(), np.ones(len(torch.transpose(X[j][:3], 0 , 1))))
                nearest_points = clf.kneighbors(points_kneighbors.detach().cpu().numpy(), return_distance=False)
                origin_colors = torch.transpose(X[j][3:6], 0, 1)[nearest_points].squeeze()
                
                origin_colors = torch.transpose(origin_colors, 0, 1) # [3, 4096]
                pred_colors = torch.transpose(pred_colors, 0, 1) # [3, 4096]
                pred_points = torch.transpose(pred_points, 0, 1) # [3, 4096]

                X_rec[j] = torch.cat([pred_points, pred_colors, origin_colors], dim=0) # [B,6,N]


            if config['reconstruction_loss'].lower() == 'combined': 
                loss_colors = reconstruction_loss(X.permute(0, 2, 1),
                                            X_rec.permute(0, 2, 1),
                                            True)
                loss_points = reconstruction_loss(X.permute(0, 2, 1),
                                            X_rec.permute(0, 2, 1),
                                            False)
                
            else:
                loss_e = torch.mean(
                    config['reconstruction_coef'] *
                    reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                        X_rec.permute(0, 2, 1) + 0.5))

            loss_colors_encoder = 0.5 * (torch.exp(logvar_cp) + torch.pow(mu_cp, 2) - 1 - logvar_cp).sum()
            loss_points_encoder = 0.5 * (torch.exp(logvar_p) + torch.pow(mu_p, 2) - 1 - logvar_p).sum()

            total_loss_colors.append(loss_colors.item())
            total_loss_points.append(loss_points.item())
            total_loss_colors_encoder.append(loss_colors_encoder.item())
            total_loss_points_encoder.append(loss_points_encoder.item())


        x = torch.cat(x)
        total_codes_p = torch.cat(total_codes_p)
        total_codes_cp = torch.cat(total_codes_cp)

        log.info(
            f'Number of interations : {i} \n'
            f'\tRec loss points: mean={np.mean(total_loss_points) :.10f} std={np.std(total_loss_points) :.10f} \n'
            f'\tRec loss colors: mean={np.mean(total_loss_colors) :.4f} std={np.std(total_loss_colors) :.4f} \n'
            f'\tRec loss points encoder: mean={np.mean(total_loss_points_encoder) :.4f} std={np.std(total_loss_points_encoder) :.4f} \n'
            f'\tRec loss colors encoder: mean={np.mean(total_loss_colors_encoder) :.4f} std={np.std(total_loss_colors_encoder) :.4f} \n'
            f'\tPoints encoder mean={torch.mean(total_codes_p.mean(dim=1)) :.4f} std={torch.mean(total_codes_p.std(dim=1)) :.4f} \n'
            f'\tColors encoder mean={torch.mean(total_codes_cp.mean(dim=1)) :.4f} std={torch.mean(total_codes_cp.std(dim=1)) :.4f} \n'

        )

        if config['experiments']['interpolation']['execute']:
            interpolation(x, encoder, hyper_network_points, device, results_dir, epoch,
                          config['experiments']['interpolation']['amount'],
                          config['experiments']['interpolation']['transitions'])

        if config['experiments']['interpolation_between_two_points']['execute']:
            interpolation_between_two_points(encoder, hyper_network_points, device, x, results_dir, epoch,
                                             config['experiments']['interpolation_between_two_points']['amount'],
                                             config['experiments']['interpolation_between_two_points']['image_points'],
                                             config['experiments']['interpolation_between_two_points']['transitions'])

        if config['experiments']['reconstruction']['execute']:
            reconstruction(encoder_p, encoder_cp, hyper_network_p, hyper_network_cp, device, x, results_dir, epoch,
                           config['experiments']['reconstruction']['amount'])

        if config['experiments']['sphere']['execute']:
            sphere(encoder, hyper_network_points, device, x, results_dir, epoch,
                   config['experiments']['sphere']['amount'], config['experiments']['sphere']['image_points'],
                   config['experiments']['sphere']['start'], config['experiments']['sphere']['end'],
                   config['experiments']['sphere']['transitions'])

        if config['experiments']['sphere_triangles']['execute']:
            sphere_triangles(encoder_p, encoder_cp, hyper_network_p, hyper_network_cp, device, x, results_dir,
                             config['experiments']['sphere_triangles']['amount'],
                             config['experiments']['sphere_triangles']['method'],
                             config['experiments']['sphere_triangles']['depth'],
                             config['experiments']['sphere_triangles']['start'],
                             config['experiments']['sphere_triangles']['end'],
                             config['experiments']['sphere_triangles']['transitions'],
                             epoch)

        if config['experiments']['sphere_triangles_interpolation']['execute']:
            sphere_triangles_interpolation(encoder_p, encoder_cp, hyper_network_p, hyper_network_cp, device, x, results_dir,
                                           config['experiments']['sphere_triangles_interpolation']['objects_amount'],
                                           config['experiments']['sphere_triangles_interpolation']['colors_amount'],
                                           config['experiments']['sphere_triangles_interpolation']['method'],
                                           config['experiments']['sphere_triangles_interpolation']['depth'],
                                           config['experiments']['sphere_triangles_interpolation']['coefficient'],
                                           config['experiments']['sphere_triangles_interpolation']['transitions'])

        if config['experiments']['different_number_of_points']['execute']:
            different_number_of_points(encoder, hyper_network_points, x, device, results_dir, epoch,
                                       config['experiments']['different_number_of_points']['amount'],
                                       config['experiments']['different_number_of_points']['image_points'])

        if config['experiments']['fixed']['execute']:
            fixed(hyper_network_p, hyper_network_cp, device, results_dir, epoch, config['experiments']['fixed']['amount'],
                  config['z_size'], 
                  config['experiments']['fixed']['points']['mean'],
                  config['experiments']['fixed']['points']['std'], 
                  torch.mean(total_codes_cp.mean(dim=1)), #config['experiments']['fixed']['colors']['mean'],
                  config['experiments']['fixed']['colors']['std'], 
                  (x.shape[1], x.shape[2]),
                  config['experiments']['fixed']['triangulation']['execute'],
                  config['experiments']['fixed']['triangulation']['method'],
                  config['experiments']['fixed']['triangulation']['depth'])


def interpolation(x, encoder, hyper_network, device, results_dir, epoch, amount=5, transitions=10):
    log.info(f'Interpolations')

    for k in range(amount):
        x_a = x[None, 2 * k, :, :]
        x_b = x[None, 2 * k + 1, :, :]

        with torch.no_grad():
            z_a, mu_a, var_a = encoder(x_a)
            z_b, mu_b, var_b = encoder(x_b)

        for j, alpha in enumerate(np.linspace(0, 1, transitions)):
            z_int = (1 - alpha) * z_a + alpha * z_b  # interpolate in the latent space
            weights_int = hyper_network(z_int)  # decode the interpolated sample

            target_network = aae.TargetNetwork(config, weights_int[0])
            target_network_input = generate_points(config=config, epoch=epoch, size=(x.shape[2], x.shape[1]))
            x_int = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

            np.save(join(results_dir, 'interpolations', f'{k}_{j}_target_network_input'), np.array(target_network_input))
            np.save(join(results_dir, 'interpolations', f'{k}_{j}_interpolation'), np.array(x_int))

            fig = plot_3d_point_cloud(x_int[0], x_int[1], x_int[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'interpolations', f'{k}_{j}_interpolation.png'))
            plt.close(fig)


def interpolation_between_two_points(encoder, hyper_network, device, x, results_dir, epoch, amount=30,
                                     image_points=1000, transitions=21):
    log.info("Interpolations between two points")
    x = x[:amount]

    z_a, _, _ = encoder(x)
    weights_int = hyper_network(z_a)
    for k in range(amount):
        target_network = aae.TargetNetwork(config, weights_int[k])
        target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, x.shape[1]))
        x_a = target_network_input[torch.argmin(target_network_input, dim=0)[2]][None, :]
        x_b = target_network_input[torch.argmax(target_network_input, dim=0)[2]][None, :]

        x_rec = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()
        x_int = torch.zeros(transitions, x.shape[1])
        for j, alpha in enumerate(np.linspace(0, 1, transitions)):
            z_int = (1 - alpha) * x_a + alpha * x_b  # interpolate point
            x_int[j] = target_network(z_int.to(device))

        x_int = torch.transpose(x_int, 0, 1).cpu().numpy()

        np.save(join(results_dir, 'points_interpolation', f'{k}_target_network_input'), np.array(target_network_input))
        np.save(join(results_dir, 'points_interpolation', f'{k}_reconstruction'), np.array(x_rec))
        np.save(join(results_dir, 'points_interpolation', f'{k}_points_interpolation'), np.array(x_int))

        fig = plot_3d_point_cloud(x_rec[0], x_rec[1], x_rec[2], in_u_sphere=True,
                                  show=False, x1=x_int[0], y1=x_int[1], z1=x_int[2])
        fig.savefig(join(results_dir, 'points_interpolation', f'{k}_points_interpolation.png'))
        plt.close(fig)



class CodeBookInstance(object):
    def __init__(self, codeword, point, normalized):
        self.codeword = codeword
        self.point = point
        self.normalized = normalized


class CodeBook(object):
    def __init__(self, vec_length, l1_norm_val, stream_len, std):
        assert l1_norm_val > 0
        assert vec_length > 0

        self.book = []
        self.vec_length = vec_length
        self.l1_norm_val = l1_norm_val
        self.stream_len = stream_len
        self.std = std
        self._create_codebook()

    def _create_codebook(self):
        p = [-self.l1_norm_val] + [0] * (self.vec_length - 1)
        while True:
            if sum(abs(x) for x in p) == self.l1_norm_val:
                l2_norm = math.sqrt(sum(x ** 2 for x in p)) 
                normalized = tuple(self.std * x / l2_norm for x in p)
                #print(len(self.book), p, normalized)
                cb_instance = CodeBookInstance(len(self.book), tuple(p), normalized)
                self.book.append(cb_instance)

            index = np.nonzero(p)[-1][-1]
            if p[index] > 0:
                left_index = index - 1
                if p[left_index] == 0:
                    p[index] = -(p[index])
                    p[index] += 1
                    p[left_index] += 1
                else:
                    p[index] = -(p[index])
                    p[index] += -1 if p[left_index] < 0 else 1
                    p[left_index] += 1
            else:
                if index >= self.vec_length - 1:
                    p[index] = -(p[index])
                else:
                    p[index] += 1
                    p[index + 1] -= 1

            if p[0] == self.l1_norm_val:
                break

    def find_nearest_pvq_code(self, value):
        assert len(value) == self.vec_length, f"{len(value)}, expected {self.vec_length}"
        ret = None
        min_dist = None
        for i in range(len(self.book)):
            q = self.book[i].normalized
            dist = math.sqrt(sum(abs(q[j] - value[j]) ** 2 for j in range(len(value))))
            if min_dist is None or dist < min_dist:
                ret = self.book[i]
                min_dist = dist

        return ret, min_dist

    def encode_sequence(self, latent):
        assert len(latent) >= self.vec_length
        latent = list(latent)
        ret = []
        for i in range(0, len(latent), self.vec_length):
            value = latent[i : (i + self.vec_length)]
            if i + self.vec_length >= len(latent):
                zeros_len = (i + self.vec_length) - len(latent)
                value = latent[i: len(latent)] + [0] * zeros_len
            node, min_dist = self.find_nearest_pvq_code(value)
            ret.append(node.codeword)
        return ret

    def decode_sequence(self, stream):
        ret = []
        for idx in stream:
            #x = list(filter(lambda x: x.codeword == idx, self.book))[0]
            x = self.book[idx]
            ret.extend(list(x.normalized))

        return ret[:self.stream_len]


def reconstruction(encoder_p, encoder_cp, hyper_network_points, hyper_network_colors, device, x, results_dir, epoch, amount=5):
    log.info("Reconstruction")
    x = x[:amount]

    z_a, _, _ = encoder_p(x[:,:3,:])
    z_b, _, _ = encoder_cp(x)

    weights_points_rec = hyper_network_points(z_a)
    weights_colors_rec = hyper_network_colors(z_b)
    x = x.cpu().numpy()
    for c in range(amount):
        target_network_points = aae.TargetNetwork(config, weights_points_rec[c])
        target_network_colors = aae.ColorsAndPointsTargetNetwork(config, weights_colors_rec[c])

        target_network_input = generate_points(config=config, epoch=epoch, size=(x.shape[2], 3)).to(device)

        x_points_rec = torch.transpose(target_network_points(target_network_input.to(device)), 0, 1).cpu().numpy()
        x_colors_rec = torch.transpose(target_network_colors(target_network_input.to(device)), 0, 1).cpu().numpy()
        x_colors_rec = colors.lab2xyz(x_colors_rec.transpose()).transpose()

        np.save(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_target_network_input'), np.array(target_network_input.detach().cpu().numpy()))
        np.save(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_real'), np.array(x[c]))
        np.save(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_reconstructed_points'), np.array(x_points_rec))
        np.save(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_reconstructed_colors'), np.array(x_colors_rec))


        fig = plot_3d_point_cloud(x_points_rec[0], x_points_rec[1], x_points_rec[2], C=x_colors_rec.transpose(), in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_reconstructed.png'))
        plt.close(fig)

        fig = plot_3d_point_cloud(x[c][0], x[c][1], x[c][2], C=x[c][3:6].transpose(), in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_real.png'))
        plt.close(fig)

    new_z_a = torch.zeros_like(z_a)
    new_z_b = torch.zeros_like(z_b)
    
    cb_a = CodeBook(3, 32, 2048, std=0.0005)
    for i, latent in enumerate(z_a):
        stream = cb_a.encode_sequence(latent.cpu().numpy())
        new_latent = cb_a.decode_sequence(stream)
        new_z_a[i] = torch.tensor(new_latent)

    cb_b = CodeBook(3, 32, 2048, std=0.0217)
    for i, latent in enumerate(z_b):
        stream = cb_b.encode_sequence(latent.cpu().numpy())
        new_latent = cb_b.decode_sequence(stream)
        new_z_b[i] = torch.tensor(new_latent)

    weights_points_rec = hyper_network_points(new_z_a)
    weights_colors_rec = hyper_network_colors(new_z_b)

    for c in range(amount):
        target_network_points = aae.TargetNetwork(config, weights_points_rec[c])
        target_network_colors = aae.ColorsAndPointsTargetNetwork(config, weights_colors_rec[c])

        target_network_input = generate_points(config=config, epoch=epoch, size=(x.shape[2], 3)).to(device)

        x_points_rec = torch.transpose(target_network_points(target_network_input.to(device)), 0, 1).cpu().numpy()
        x_colors_rec = torch.transpose(target_network_colors(target_network_input.to(device)), 0, 1).cpu().numpy()
        x_colors_rec = colors.lab2xyz(x_colors_rec.transpose()).transpose()

        np.save(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_target_network_input'), np.array(target_network_input.detach().cpu().numpy()))
        np.save(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_real'), np.array(x[c]))
        np.save(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_reconstructed_points'), np.array(x_points_rec))
        np.save(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_reconstructed_colors'), np.array(x_colors_rec))


        fig = plot_3d_point_cloud(x_points_rec[0], x_points_rec[1], x_points_rec[2], C=x_colors_rec.transpose(), in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_reconstructed.png'))
        plt.close(fig)

        fig = plot_3d_point_cloud(x[c][0], x[c][1], x[c][2], C=x[c][3:6].transpose(), in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_real.png'))
        plt.close(fig)


    


def sphere(encoder, hyper_network, device, x, results_dir, epoch, amount=10, image_points=10240, start=2.0, end=4.0,
           transitions=21):
    log.info("Sphere")
    x = x[:amount]

    z_a, _, _ = encoder(x)
    weights_sphere = hyper_network(z_a)
    x = x.cpu().numpy()
    for k in range(amount):
        target_network = aae.TargetNetwork(config, weights_sphere[k])
        target_network_input = generate_points(config=config, epoch=epoch, size=(image_points, x.shape[1]),
                                               normalize_points=False)
        x_rec = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

        np.save(join(results_dir, 'sphere', f'{k}_real'), np.array(x[k]))
        np.save(join(results_dir, 'sphere', f'{k}_point_cloud_before_normalization'),
                np.array(target_network_input))
        np.save(join(results_dir, 'sphere', f'{k}_reconstruction'), np.array(x_rec))

        target_network_input = target_network_input / torch.norm(target_network_input, dim=1).view(-1, 1)
        np.save(join(results_dir, 'sphere', f'{k}_point_cloud_after_normalization'),
                np.array(target_network_input))

        for coeff in np.linspace(start, end, num=transitions):
            coeff = round(coeff, 1)
            x_sphere = torch.transpose(target_network(target_network_input.to(device) * coeff), 0, 1).cpu().numpy()

            np.save(join(results_dir, 'sphere',
                         f'{k}_output_from_target_network_for_point_cloud_after_normalization_coefficient_{coeff}'),
                    np.array(x_sphere))

            fig = plot_3d_point_cloud(x_sphere[0], x_sphere[1], x_sphere[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'sphere', f'{k}_{coeff}_sphere.png'))
            plt.close(fig)

        fig = plot_3d_point_cloud(x[k][0], x[k][1], x[k][2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'sphere', f'{k}_real.png'))
        plt.close(fig)


def sphere_triangles(encoder_p, encoder_cp, hyper_network_points, hyper_network_colors, device, x, \
    results_dir, amount, method, depth, start, end, transitions, epoch):
    from utils.sphere_triangles import generate
    log.info("Sphere triangles")
    x = x[:amount]

    z_a, _, _ = encoder_p(x[:,:3,:])
    z_b, _, _ = encoder_cp(x)
    weights_points_rec = hyper_network_points(z_a)
    weights_colors_rec = hyper_network_colors(z_b)
    x = x.cpu().numpy()

    for p in range(amount):
        for c in range(amount):
            target_network_points = aae.TargetNetwork(config, weights_points_rec[p])
            target_network_colors = aae.ColorsAndPointsTargetNetwork(config, weights_colors_rec[c])

            target_network_input, triangulation = generate(method, depth)
            
            x_points_rec = torch.transpose(target_network_points(target_network_input.to(device)), 0, 1).cpu().numpy()
            x_colors_rec = torch.transpose(target_network_colors(target_network_input.to(device)), 0, 1).cpu().numpy()
            x_colors_rec = colors.lab2xyz(x_colors_rec.transpose()).transpose()
            
            np.save(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_real'), np.array(x[p]))
            np.save(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_point_cloud'), np.array(target_network_input.cpu().numpy()))
            np.save(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_reconstructed_points'), np.array(x_points_rec))
            np.save(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_reconstructed_colors'), np.array(x_colors_rec))

            with open(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_triangulation.pickle'), 'wb') as triangulation_file:
                pickle.dump(triangulation, triangulation_file)

            fig = plot_3d_point_cloud(x_points_rec[0], x_points_rec[1], x_points_rec[2], C=x_colors_rec.transpose(), in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_reconstructed.png'))
            plt.close(fig)

            for coefficient in np.linspace(start, end, num=transitions):
                coefficient = round(coefficient, 3)

                target_network_input_coefficient = target_network_input * coefficient
                x_sphere = torch.transpose(target_network_points(target_network_input_coefficient.to(device)), 0, 1).cpu().numpy()
                x_sphere_color = torch.transpose(target_network_colors(target_network_input_coefficient.to(device)), 0, 1).cpu().numpy()
                x_sphere_color = colors.lab2xyz(x_sphere_color.transpose()).transpose()

                np.save(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_point_cloud_coefficient_{coefficient}'),
                        np.array(target_network_input_coefficient))
                np.save(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_reconstruction_coefficient_{coefficient}'), x_sphere)
                np.save(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_reconstruction_coefficient_for_colors_{coefficient}'), x_sphere_color)

                fig = plot_3d_point_cloud(x_sphere[0], x_sphere[1], x_sphere[2], C=x_sphere_color.transpose(), in_u_sphere=True, show=False)
                fig.savefig(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_{coefficient}_reconstructed.png'))
                plt.close(fig)

            fig = plot_3d_point_cloud(x[p][0], x[p][1], x[p][2], C=x[c][3:6].transpose(), in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'sphere_triangles', f'p_{p}_c_{c}_real.png'))
            plt.close(fig)


def sphere_triangles_interpolation(encoder_p, encoder_cp, hyper_network_p, hyper_network_cp, device, x, results_dir, objects_amount, colors_amount, method, 
                            depth, coefficient, transitions):
    from utils.sphere_triangles import generate
    log.info("Sphere triangles interpolation")

    for p in range(objects_amount):
        x_a_p = x[None, 2 * p, :, :]
        x_b_p = x[None, 2 * p + 1, :, :]
        for c in range(colors_amount):
            x_a_cp = x[None, 2 * c, :, :]
            x_b_cp = x[None, 2 * c + 1, :, :]

            with torch.no_grad():
                z_a_p, mu_a_p, var_a_p = encoder_p(x_a_p[:,:3,:])
                z_b_p, mu_b_p, var_b_p = encoder_p(x_b_p[:,:3,:])
                z_a_cp, mu_a_cp, var_a_cp = encoder_cp(x_a_cp)
                z_b_cp, mu_b_cp, var_b_cp = encoder_cp(x_b_cp)

            for j, alpha in enumerate(np.linspace(0, 1, transitions)):
                z_int_p = (1 - alpha) * z_a_p + alpha * z_b_p  # interpolate in the latent space
                weights_int_p = hyper_network_p(z_int_p)  # decode the interpolated sample

                z_int_cp = (1 - alpha) * z_a_cp + alpha * z_b_cp  # interpolate in the latent space
                weights_int_cp = hyper_network_cp(z_int_cp)  # decode the interpolated sample

                target_network_points = aae.TargetNetwork(config, weights_int_p[0])
                target_network_colors = aae.ColorsAndPointsTargetNetwork(config, weights_int_cp[0])

                target_network_input, triangulation = generate(method, depth)
                x_int_p = torch.transpose(target_network_points(target_network_input.to(device)), 0, 1).cpu().numpy()
                x_int_cp = torch.transpose(target_network_colors(target_network_input.to(device)), 0, 1).cpu().numpy()
                x_int_cp = colors.lab2xyz(x_int_cp.transpose()).transpose()

                np.save(join(results_dir, 'sphere_triangles_interpolation', f'p_{p}_c_{c}_{j}_point_cloud'),
                        np.array(target_network_input))
                np.save(join(results_dir, 'sphere_triangles_interpolation', f'p_{p}_c_{c}_{j}_points_interpolation'), x_int_p)
                np.save(join(results_dir, 'sphere_triangles_interpolation', f'p_{p}_c_{c}_{j}_colors_interpolation'), x_int_cp)

                with open(join(results_dir, 'sphere_triangles_interpolation', f'p_{p}_c_{c}_{j}_triangulation.pickle'),
                        'wb') as triangulation_file:
                    pickle.dump(triangulation, triangulation_file)

                fig = plot_3d_point_cloud(x_int_p[0], x_int_p[1], x_int_p[2], C=x_int_cp.transpose(), in_u_sphere=True, show=False)
                fig.savefig(join(results_dir, 'sphere_triangles_interpolation', f'p_{p}_c_{c}_{j}_interpolation.png'))
                plt.close(fig)

                target_network_input_coefficient = target_network_input * coefficient
                x_int_coeff_p = torch.transpose(target_network_points(target_network_input_coefficient.to(device)), 0, 1).cpu().numpy()
                x_int_coeff_cp = torch.transpose(target_network_colors(target_network_input_coefficient.to(device)), 0, 1).cpu().numpy()
                x_int_coeff_cp = colors.lab2xyz(x_int_coeff_cp.transpose()).transpose()

                np.save(join(results_dir,
                            'sphere_triangles_interpolation', f'p_{p}_c_{c}_{j}_point_cloud_coefficient_{coefficient}'),
                        np.array(target_network_input_coefficient))
                np.save(join(results_dir, 'sphere_triangles_interpolation',
                            f'p_{p}_c_{c}_{j}_points_interpolation_coefficient_{coefficient}'), x_int_coeff_p)
                np.save(join(results_dir, 'sphere_triangles_interpolation',
                            f'p_{p}_c_{c}_{j}_colors_interpolation_coefficient_{coefficient}'), x_int_coeff_cp)

                fig = plot_3d_point_cloud(x_int_coeff_p[0], x_int_coeff_p[1], x_int_coeff_p[2], C=x_int_coeff_cp.transpose(), in_u_sphere=True, show=False)
                fig.savefig(join(results_dir, 'sphere_triangles_interpolation', f'p_{p}_c_{c}_{j}_{coefficient}_interpolation.png'))
                plt.close(fig)


def different_number_of_points(encoder, hyper_network, x, device, results_dir, epoch, amount=5,
                               number_of_points_list=(10, 100, 1000, 2048, 10000)):
    log.info("Different number of points")
    x = x[:amount]

    latent, _, _ = encoder(x)
    weights_diff = hyper_network(latent)
    x = x.cpu().numpy()
    for k in range(amount):
        np.save(join(results_dir, 'different_number_points', f'{k}_real'), np.array(x[k]))
        fig = plot_3d_point_cloud(x[k][0], x[k][1], x[k][2], in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'different_number_points', f'{k}_real.png'))
        plt.close(fig)

        target_network = aae.TargetNetwork(config, weights_diff[k])

        for number_of_points in number_of_points_list:
            target_network_input = generate_points(config=config, epoch=epoch, size=(number_of_points, x.shape[1]))
            x_diff = torch.transpose(target_network(target_network_input.to(device)), 0, 1).cpu().numpy()

            np.save(join(results_dir, 'different_number_points', f'{k}_target_network_input'),
                    np.array(target_network_input))
            np.save(join(results_dir, 'different_number_points', f'{k}_{number_of_points}'), np.array(x_diff))

            fig = plot_3d_point_cloud(x_diff[0], x_diff[1], x_diff[2], in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'different_number_points', f'{k}_{number_of_points}.png'))
            plt.close(fig)


def fixed(hyper_network_points, hyper_network_colors, device, results_dir, epoch, fixed_number, z_size, fixed_points_mean, fixed_points_std,
         fixed_colors_mean, fixed_colors_std, x_shape, triangulation, method, depth):
    log.info("Fixed")

    fixed_points_noise = torch.zeros(fixed_number, z_size).normal_(mean=fixed_points_mean, std=fixed_points_std).to(device)
    fixed_colors_noise = torch.zeros(fixed_number, z_size).normal_(mean=fixed_colors_mean, std=fixed_colors_std).to(device)
    weights_points_fixed = hyper_network_points(fixed_points_noise)
    weights_colors_fixed = hyper_network_colors(fixed_colors_noise)

    for p, weights_points in enumerate(weights_points_fixed):
        for c, weights_color in enumerate(weights_colors_fixed):

            target_network_points = aae.TargetNetwork(config, weights_points).to(device)
            target_network_colors = aae.ColorsAndPointsTargetNetwork(config, weights_color).to(device)

            target_network_input = generate_points(config=config, epoch=epoch, size=(x_shape[1], 3))
            fixed_points_rec = torch.transpose(target_network_points(target_network_input.to(device)), 0, 1).cpu().numpy()
            fixed_colors_rec = torch.transpose(target_network_colors(target_network_input.to(device)), 0, 1).cpu().numpy()
            fixed_colors_rec = colors.lab2xyz(fixed_colors_rec.transpose()).transpose()
            
            np.save(join(results_dir, 'fixed', f'p_{p}_c_{c}_target_network_input'), np.array(target_network_input))
            np.save(join(results_dir, 'fixed', f'p_{p}_c_{c}_fixed_reconstruction_points'), fixed_points_rec)
            np.save(join(results_dir, 'fixed', f'p_{p}_c_{c}_fixed_reconstruction_colors'), fixed_colors_rec)

            fig = plot_3d_point_cloud(fixed_points_rec[0], fixed_points_rec[1], fixed_points_rec[2], C=fixed_colors_rec.transpose(), in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'fixed', f'p_{p}_c_{c}_fixed_reconstructed.png'))
            plt.close(fig)

            if triangulation:
                from utils.sphere_triangles import generate

                target_network_input, triangulation = generate(method, depth)

                with open(join(results_dir, 'fixed', f'p_{p}_c_{c}_triangulation.pickle'), 'wb') as triangulation_file:
                    pickle.dump(triangulation, triangulation_file)

                fixed_points_rec = torch.transpose(target_network_points(target_network_input.to(device)), 0, 1).cpu().numpy()
                fixed_colors_rec = torch.transpose(target_network_colors(target_network_input.to(device)), 0, 1).cpu().numpy()
                fixed_colors_rec = colors.lab2xyz(fixed_colors_rec.transpose()).transpose()

                np.save(join(results_dir, 'fixed', f'p_{p}_c_{c}_target_network_input_triangulation'),
                        np.array(target_network_input))
                np.save(join(results_dir, 'fixed', f'p_{p}_c_{c}_fixed_points_reconstruction_triangulation'), fixed_points_rec)
                np.save(join(results_dir, 'fixed', f'p_{p}_c_{c}_fixed_colors_reconstruction_triangulation'), fixed_colors_rec)

                fig = plot_3d_point_cloud(fixed_points_rec[0], fixed_points_rec[1], fixed_points_rec[2], C=fixed_colors_rec.transpose(), in_u_sphere=True, show=False)
                fig.savefig(join(results_dir, 'fixed', f'{p}_fixed_reconstructed_triangulation.png'))
                plt.close(fig)

            np.save(join(results_dir, 'fixed', f'p_{p}_c_{c}_fixed_points_noise'), np.array(fixed_points_noise[p].cpu()))
            np.save(join(results_dir, 'fixed', f'p_{p}_c_{c}_fixed_colors_noise'), np.array(fixed_colors_noise[c].cpu()))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
