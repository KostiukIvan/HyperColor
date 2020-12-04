import argparse
import json
import logging
from datetime import datetime
import shutil
from itertools import chain
from os.path import join, exists
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from models import aae
from utils.pcutil import plot_3d_point_cloud
from utils.util import find_latest_epoch, prepare_results_dir, cuda_setup, setup_logging, set_seed
from utils.points import generate_points
from datasets.meshsDataset import Mesh
from sklearn.neighbors import KNeighborsClassifier

cudnn.benchmark = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def main(config):
    set_seed(config['seed'])

    results_dir = prepare_results_dir(config, 'vae', 'training')
    starting_epoch = find_latest_epoch(results_dir) + 1

    if not exists(join(results_dir, 'config.json')):
        with open(join(results_dir, 'config.json'), mode='w') as f:
            json.dump(config, f)

    setup_logging(results_dir)
    log = logging.getLogger('vae')

    device = cuda_setup(config['cuda'], config['gpu'])
    log.info(f'Device variable: {device}')
    if device.type == 'cuda':
        log.info(f'Current CUDA device: {torch.cuda.current_device()}')

    weights_path = join(results_dir, 'weights')
    metrics_path = join(results_dir, 'metrics')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'],
                                  classes=config['classes'])
    elif dataset_name == "photogrammetry":
        from datasets.photogrammetry import PhotogrammetryDataset
        dataset = PhotogrammetryDataset(root_dir=config['data_dir'],
                                 classes=config['classes'], config=config)

    elif dataset_name == "custom":
        # import pdb; pdb.set_trace()
        from datasets.customDataset import CustomDataset
        dataset = CustomDataset(root_dir=config['data_dir'],
                                classes=config['classes'], config=config)

        # Load spheres and faces
        #meshes = Mesh(config["meshs_of_sphere_dir"], config["n_points"])

    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')

    log.info("Selected {} classes. Loaded {} samples.".format(
        'all' if not config['classes'] else ','.join(config['classes']),
        len(dataset)))

    points_dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                   shuffle=config['shuffle'],
                                   num_workers=config['num_workers'], drop_last=True,
                                   pin_memory=True)

    #
    # Models
    #
    hyper_network_points = aae.HyperNetwork(config, device).to(device)
    hyper_network_colors = aae.HyperNetwork(config, device).to(device)
    encoder = aae.Encoder(config).to(device)

    hyper_network_points.apply(weights_init)
    hyper_network_colors.apply(weights_init)
    encoder.apply(weights_init)

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
        reconstruction_loss = CombinedLoss(config).to(device)

    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    #
    # Optimizers
    #
    e_hn_optimizer_points = getattr(optim, config['optimizer']['E_HN']['type'])
    e_hn_optimizer_points = e_hn_optimizer_points(chain(encoder.parameters(), hyper_network_points.parameters()),
                                    **config['optimizer']['E_HN']['hyperparams'])

    e_hn_optimizer_colors = getattr(optim, config['optimizer']['E_HN']['type'])
    e_hn_optimizer_colors = e_hn_optimizer_colors(chain(hyper_network_colors.parameters()),
                                    **config['optimizer']['E_HN']['hyperparams'])

    log.info("Starting epoch: %s" % starting_epoch)
    if starting_epoch > 1:
        log.info("Loading weights...")
        hyper_network_points.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_GP.pth')))
        hyper_network_colors.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_GC.pth')))

        encoder.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_E.pth')))

        e_hn_optimizer_points.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EGoP.pth')))
        e_hn_optimizer_colors.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EGoC.pth')))

        log.info("Loading losses...")
        losses_e = np.load(join(metrics_path, f'{starting_epoch - 1:05}_E.npy')).tolist()
        losses_kld = np.load(join(metrics_path, f'{starting_epoch - 1:05}_KLD.npy')).tolist()
        losses_eg = np.load(join(metrics_path, f'{starting_epoch - 1:05}_EG.npy')).tolist()
    else:
        log.info("First epoch")
        losses_e = []
        losses_kld = []
        losses_eg = []

    if config['target_network_input']['normalization']['enable']:
        normalization_type = config['target_network_input']['normalization']['type']
        assert normalization_type == 'progressive', 'Invalid normalization type'

    target_network_input = None
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        log.debug("Epoch: %s" % epoch)

        train_colors = False
        if epoch > config['target_network_input']['loss']['change_to']['after_epoch']:
            train_colors = True

        if train_colors:
            hyper_network_colors.train()
            hyper_network_points.eval()
            encoder.eval()
        else:
            hyper_network_points.train()
            encoder.train()
            hyper_network_colors.eval()

        total_loss_all = 0.0
        total_loss_r = 0.0
        total_loss_kld = 0.0
        for i, point_data in enumerate(points_dataloader, 1):

            if dataset_name == "custom":
                X = torch.cat((point_data['points'], point_data['colors']), dim=2)
                X = X.to(device, dtype=torch.float)
                X_normals = point_data['normals'].to(device, dtype=torch.float)

            else: 
                X, _ = point_data
                X = X.to(device, dtype=torch.float)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3 or X.size(-1) == 6 or X.size(-1) == 7:
                X.transpose_(X.dim() - 2, X.dim() - 1)
            
            codes, mu, logvar = encoder(X)

            if train_colors:
                target_networks_weights_colors = hyper_network_colors(codes)
                target_networks_weights_points = hyper_network_points(codes)
                
                X_rec = torch.zeros(torch.cat([X, X[:,:3,:]], dim=1).shape).to(device) # [b, 9, 4096]
                for j, target_network_weights in enumerate(zip(target_networks_weights_points, target_networks_weights_colors)):

                    target_network_points = aae.TargetNetwork(config, target_network_weights[0]).to(device)
                    target_network_colors = aae.TargetNetwork(config, target_network_weights[1]).to(device)

                    if not config['target_network_input']['constant'] or target_network_input is None:     
                        target_network_input = generate_points(config=config, epoch=epoch, size=(X.shape[2], 3)).to(device)

                    pred_points = target_network_points(target_network_input.to(device, dtype=torch.float)) # [4096, 3]
                    pred_colors = target_network_colors(target_network_input.to(device, dtype=torch.float)) # [4096, 3]

                    points_kneighbors = pred_points
                    clf = KNeighborsClassifier(1)
                    clf.fit(torch.transpose(X[j][:3], 0 , 1).cpu().numpy(), np.ones(len(torch.transpose(X[j][:3], 0 , 1))))
                    nearest_points = clf.kneighbors(points_kneighbors.detach().cpu().numpy(), return_distance=False)
                    origin_colors = torch.transpose(X[j][3:6], 0, 1)[nearest_points].squeeze()
                    
                    origin_colors = torch.transpose(origin_colors, 0, 1) # [3, 4096]
                    pred_colors = torch.transpose(pred_colors, 0, 1) # [3, 4096]
                    pred_points = torch.transpose(pred_points, 0, 1) # [3, 4096]

                    X_rec[j] = torch.cat([pred_points, pred_colors, origin_colors], dim=0) # [B,6,N]

            else:
                target_networks_weights_points = hyper_network_points(codes)
                X_rec = torch.zeros(X[:,:3,:].shape).to(device)
                for j, target_network_weights_points in enumerate(target_networks_weights_points):
                    target_network_points = aae.TargetNetwork(config, target_network_weights_points).to(device)

                    if not config['target_network_input']['constant'] or target_network_input is None:     
                        target_network_input = generate_points(config=config, epoch=epoch, size=(X.shape[2], 3)).to(device)
                    X_rec[j] = torch.transpose(target_network_points(target_network_input.to(device, dtype=torch.float)), 0, 1)

        
            if config['reconstruction_loss'].lower() == 'combined': 
                loss_r = reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                            X_rec.permute(0, 2, 1) + 0.5,
                                            train_colors)
                
            else:
                loss_r = torch.mean(
                    config['reconstruction_coef'] *
                    reconstruction_loss(X.permute(0, 2, 1) + 0.5,
                                        X_rec.permute(0, 2, 1) + 0.5))
            if train_colors:
                loss_all = loss_r

                e_hn_optimizer_colors.zero_grad()
                hyper_network_colors.zero_grad()

                loss_all.backward()
                e_hn_optimizer_colors.step()

                total_loss_r += loss_r.item()
                total_loss_all += loss_all.item()

            else:
                loss_kld = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar).sum()
                loss_all = loss_r + loss_kld

                e_hn_optimizer_points.zero_grad()
                encoder.zero_grad()
                hyper_network_points.zero_grad()

                loss_all.backward()
                e_hn_optimizer_points.step()

                total_loss_r += loss_r.item()
                total_loss_kld += loss_kld.item()
                total_loss_all += loss_all.item()

        log.info(
            f'[{epoch}/{config["max_epochs"]}] '
            f'Loss_ALL: {total_loss_all / i:.4f} '
            f'Loss_R: {total_loss_r / i:.4f} '
            f'Loss_E: {total_loss_kld / i:.4f} '
            f'Time: {datetime.now() - start_epoch_time}'
        )

        losses_e.append(total_loss_r)
        losses_kld.append(total_loss_kld)
        losses_eg.append(total_loss_all)

        #
        # Save intermediate results
        #
        X = X.cpu().numpy()
        X_rec = X_rec.detach().cpu().numpy()

        for k in range(min(1, X_rec.shape[0])):
            C = None
            if train_colors:
                C = X_rec[k][3:6].transpose()
            fig = plot_3d_point_cloud(X_rec[k][0], X_rec[k][1], X_rec[k][2], C = C, in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_reconstructed.png'))
            plt.close(fig)

            fig = plot_3d_point_cloud(X[k][0], X[k][1], X[k][2], C = X[k][3:6].transpose(), in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_real.png'))
            plt.close(fig)
            
        if config['clean_weights_dir']:
            log.debug('Cleaning weights path: %s' % weights_path)
            shutil.rmtree(weights_path, ignore_errors=True)
            os.makedirs(weights_path, exist_ok=True)

        if epoch % config['save_frequency'] == 0:
            log.debug('Saving data...')

            torch.save(hyper_network_points.state_dict(), join(weights_path, f'{epoch:05}_GP.pth'))
            torch.save(hyper_network_colors.state_dict(), join(weights_path, f'{epoch:05}_GC.pth'))
            torch.save(encoder.state_dict(), join(weights_path, f'{epoch:05}_E.pth'))
            torch.save(e_hn_optimizer_points.state_dict(), join(weights_path, f'{epoch:05}_EGoP.pth'))
            torch.save(e_hn_optimizer_colors.state_dict(), join(weights_path, f'{epoch:05}_EGoC.pth'))

            np.save(join(metrics_path, f'{epoch:05}_E'), np.array(losses_e))
            np.save(join(metrics_path, f'{epoch:05}_KLD'), np.array(losses_kld))
            np.save(join(metrics_path, f'{epoch:05}_EG'), np.array(losses_eg))


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    args = parser.parse_args()

    config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            config = json.load(f)
    assert config is not None

    main(config)
