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
from code_book import *

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
    log.info(f"Device variable: {device} ")

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

    points_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=12, drop_last=True,
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
            if i > 15:
              break
            
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

        if config['experiments']['reconstruction']['execute']:
            reconstruction(encoder_p, encoder_cp, hyper_network_p, hyper_network_cp, device, x, results_dir, epoch,
                           config['experiments']['reconstruction']['amount'])



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
    
    encoder = Encoder()
    decoder = Decoder()
    #for i, latent in enumerate(z_a):
    #    stream = encoder(list(latent.cpu().numpy()))
    #    new_latent = decoder(stream)
    #    new_z_a[i] = torch.tensor(new_latent)

    
    for i, latent in enumerate(z_b):
        stream = encoder(list(latent.cpu().numpy()))
        print(len(stream))
        new_latent = decoder(stream)
        new_z_b[i] = torch.tensor(new_latent)

    weights_points_rec = hyper_network_points(z_a)
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
        fig.savefig(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_reconstructed_C.png'))
        plt.close(fig)

        fig = plot_3d_point_cloud(x[c][0], x[c][1], x[c][2], C=x[c][3:6].transpose(), in_u_sphere=True, show=False)
        fig.savefig(join(results_dir, 'reconstruction', f'p_{c}_c_{c}_real_C.png'))
        plt.close(fig)



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

