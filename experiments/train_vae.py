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
import skimage.color as colors

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
    hyper_network_p = aae.PointsHyperNetwork(config, device).to(device)
    hyper_network_cp = aae.ColorsAndPointsHyperNetwork(config, device).to(device)

    encoder_p = aae.PointsEncoder(config).to(device)
    encoder_cp = aae.ColorsAndPointsEncoder(config).to(device)

    hyper_network_p.apply(weights_init)
    hyper_network_cp.apply(weights_init)

    encoder_p.apply(weights_init)
    encoder_cp.apply(weights_init)

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
        reconstruction_loss = CombinedLoss().to(device)
    else:
        raise ValueError(f'Invalid reconstruction loss. Accepted `chamfer` or '
                         f'`earth_mover`, got: {config["reconstruction_loss"]}')

    #
    # Optimizers
    #
    e_hn_optimizer_p = getattr(optim, config['optimizer']['E_HN']['type'])
    e_hn_optimizer_p = e_hn_optimizer_p(chain(encoder_p.parameters(), hyper_network_p.parameters()),
                                    **config['optimizer']['E_HN']['hyperparams'])

    e_hn_optimizer_cp = getattr(optim, config['optimizer']['E_HN']['type'])
    e_hn_optimizer_cp = e_hn_optimizer_cp(chain(encoder_cp.parameters(), hyper_network_cp.parameters()),
                                    **config['optimizer']['E_HN']['hyperparams'])

    log.info("Starting epoch: %s" % starting_epoch)
    if starting_epoch > 1:
        log.info("Loading weights...")
        hyper_network_p.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_G_P.pth')))
        hyper_network_cp.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_G_CP.pth')))

        encoder_p.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_E_P.pth')))
        encoder_cp.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_E_CP.pth')))

        e_hn_optimizer_p.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EGoP.pth')))
        e_hn_optimizer_cp.load_state_dict(torch.load(
            join(weights_path, f'{starting_epoch - 1:05}_EGoC.pth')))

        log.info("Loading losses...")
        losses_e_p = np.load(join(metrics_path, f'{starting_epoch - 1:05}_E_P.npy')).tolist()
        losses_kld_p = np.load(join(metrics_path, f'{starting_epoch - 1:05}_KLD_P.npy')).tolist()
        losses_eg_p = np.load(join(metrics_path, f'{starting_epoch - 1:05}_EG_P.npy')).tolist()
        
        losses_e_cp = np.load(join(metrics_path, f'{starting_epoch - 1:05}_E_CP.npy')).tolist()
        losses_kld_cp = np.load(join(metrics_path, f'{starting_epoch - 1:05}_KLD_CP.npy')).tolist()
        losses_eg_cp = np.load(join(metrics_path, f'{starting_epoch - 1:05}_EG_CP.npy')).tolist()

    else:
        log.info("First epoch")
        losses_e_p = []
        losses_kld_p = []
        losses_eg_p = []
        
        losses_e_cp = []
        losses_kld_cp = []
        losses_eg_cp = []

    if config['target_network_input']['normalization']['enable']:
        normalization_type = config['target_network_input']['normalization']['type']
        assert normalization_type == 'progressive', 'Invalid normalization type'

    target_network_input = None
    for epoch in range(starting_epoch, config['max_epochs'] + 1):
        start_epoch_time = datetime.now()
        log.debug("Epoch: %s" % epoch)

        train_colors = epoch > config['target_network_input']['colors']['enable_after']
        train_points = epoch < config['target_network_input']['points']['disable_after']

        if train_colors:
            hyper_network_p.train()
            encoder_p.train()

            hyper_network_cp.eval()
            encoder_cp.eval()

        if train_points:
            hyper_network_cp.train()
            encoder_cp.train()

            hyper_network_p.eval()
            encoder_p.eval()


        total_loss_all_p = 0.0
        total_loss_r_p = 0.0
        total_loss_kld_p = 0.0

        total_loss_all_cp = 0.0
        total_loss_r_cp = 0.0
        total_loss_kld_cp = 0.0
        for i, point_data in enumerate(points_dataloader, 1):

            if dataset_name == "custom":
                X = torch.cat((point_data['points'], point_data['colors']), dim=2)
                X = X.to(device, dtype=torch.float)

            else: 
                X, _ = point_data
                X = X.to(device, dtype=torch.float)

            # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
            if X.size(-1) == 3 or X.size(-1) == 6 or X.size(-1) == 7:
                X.transpose_(X.dim() - 2, X.dim() - 1)

            codes_p, mu_p, logvar_p = None, None, None
            codes_cp, mu_cp, logvar_cp = None, None, None
            codes_comb = None
            X_rec = torch.zeros(torch.cat([X, X[:,:3,:]], dim=1).shape).to(device) # [b, 9, 4096]

            if train_points and not train_colors:
                codes_p, mu_p, logvar_p = encoder_p(X[:,:3,:])
                target_networks_weights_p = hyper_network_p(codes_p)
                for j, target_network_weights_p in enumerate(target_networks_weights_p):
                    target_network_p = aae.TargetNetwork(config, target_network_weights_p).to(device)

                    if not config['target_network_input']['constant'] or target_network_input is None:     
                        target_network_input = generate_points(config=config, epoch=epoch, size=(X.shape[2], 3)).to(device)
                    X_rec[j][:3] = torch.transpose(target_network_p(target_network_input.to(device, dtype=torch.float)), 0, 1)

            if train_colors:
                codes_p, mu_p, logvar_p = encoder_p(X[:,:3,:])
                codes_cp, mu_cp, logvar_cp = encoder_cp(X[:,:6,:])

                target_networks_weights_p = hyper_network_p(codes_p)
                target_networks_weights_cp = hyper_network_cp(codes_cp)
                
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


            if train_points and not train_colors:
                loss_r = reconstruction_loss(X.permute(0, 2, 1),
                                            X_rec.permute(0, 2, 1),
                                            False)

                loss_kld = 0.5 * (torch.exp(logvar_p) + torch.pow(mu_p, 2) - 1 - logvar_p).sum()
                loss_all = loss_r + loss_kld

                e_hn_optimizer_p.zero_grad()
                encoder_p.zero_grad()
                hyper_network_p.zero_grad()

                loss_all.backward()
                e_hn_optimizer_p.step()

                total_loss_r_p += loss_r.item()
                total_loss_kld_p += loss_kld.item()
                total_loss_all_p += loss_all.item()
            
            if train_colors and not train_points:
                loss_r = reconstruction_loss(X.permute(0, 2, 1),
                                            X_rec.permute(0, 2, 1),
                                            True)

                loss_kld = 0.5 * (torch.exp(logvar_cp) + torch.pow(mu_cp, 2) - 1 - logvar_cp).sum()
                loss_all = loss_r + loss_kld

                e_hn_optimizer_cp.zero_grad()
                hyper_network_cp.zero_grad()
                encoder_cp.zero_grad()

                loss_all.backward()
                e_hn_optimizer_cp.step()

                total_loss_r_cp += loss_r.item()
                total_loss_kld_cp += loss_kld.item()
                total_loss_all_cp += loss_all.item()


            if train_colors and train_points:
                loss_r_cp = reconstruction_loss(X.permute(0, 2, 1),
                                            X_rec.permute(0, 2, 1),
                                            True)
                loss_r_p = reconstruction_loss(X.permute(0, 2, 1),
                                            X_rec.permute(0, 2, 1),
                                            False)

                loss_r = loss_r_cp + loss_r_p

                loss_kld_cp = 0.5 * (torch.exp(logvar_cp) + torch.pow(mu_cp, 2) - 1 - logvar_cp).sum()
                loss_kld_p = 0.5 * (torch.exp(logvar_p) + torch.pow(mu_p, 2) - 1 - logvar_p).sum()
                loss_all = loss_r + loss_kld_cp + loss_kld_p

                e_hn_optimizer_cp.zero_grad()
                hyper_network_cp.zero_grad()
                encoder_cp.zero_grad()

                e_hn_optimizer_p.zero_grad()
                encoder_p.zero_grad()
                hyper_network_p.zero_grad()

                loss_all.backward()

                e_hn_optimizer_cp.step()
                e_hn_optimizer_p.step()

                total_loss_r_cp += loss_r_cp.item()
                total_loss_kld_cp += loss_kld_cp.item()
                total_loss_all_cp += (loss_r_cp + loss_kld_cp).item()
                
                total_loss_r_p += loss_r_p.item()
                total_loss_kld_p += loss_kld_p.item()
                total_loss_all_p += (loss_r_p + loss_kld_p).item()


        if train_points:
            log.info(
                f'P: [{epoch}/{config["max_epochs"]}] '
                f'Loss_ALL: {total_loss_all_p / i:.4f} '
                f'Loss_R: {total_loss_r_p/ i:.4f} '
                f'Loss_E: {total_loss_kld_p / i:.4f} '
                f'Time: {datetime.now() - start_epoch_time}'
            )

            losses_e_p.append(total_loss_r_p)
            losses_kld_p.append(total_loss_kld_p)
            losses_eg_p.append(total_loss_all_p)

        if train_colors:
            log.info(
                f'C: [{epoch}/{config["max_epochs"]}] '
                f'Loss_ALL: {total_loss_all_cp / i:.4f} '
                f'Loss_R: {total_loss_r_cp/ i:.4f} '
                f'Loss_E: {total_loss_kld_cp / i:.4f} '
                f'Time: {datetime.now() - start_epoch_time}'
            )

            losses_e_cp.append(total_loss_r_cp)
            losses_kld_cp.append(total_loss_kld_cp)
            losses_eg_cp.append(total_loss_all_cp)

        #
        # Save intermediate results
        #
        X = X.cpu().numpy()
        X_rec = X_rec.detach().cpu().numpy()

        for k in range(min(1, X_rec.shape[0])):
            C = None
            C_rec = None
            if train_colors:
                C_rec = colors.lab2xyz(X_rec[k][3:6].transpose())
                C = X[k][3:6].transpose()

            fig = plot_3d_point_cloud(X_rec[k][0], X_rec[k][2], X_rec[k][1], C = C_rec, in_u_sphere=True, show=False,
                                      title=str(epoch))
            fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_reconstructed.png'))
            plt.close(fig)

            fig = plot_3d_point_cloud(X[k][0], X[k][2], X[k][1], C = C, in_u_sphere=True, show=False)
            fig.savefig(join(results_dir, 'samples', f'{epoch}_{k}_real.png'))
            plt.close(fig)
            
        if config['clean_weights_dir']:
            log.debug('Cleaning weights path: %s' % weights_path)
            shutil.rmtree(weights_path, ignore_errors=True)
            os.makedirs(weights_path, exist_ok=True)

        if epoch % config['save_frequency'] == 0:
            log.debug('Saving data...')

            torch.save(hyper_network_p.state_dict(), join(weights_path, f'{epoch:05}_G_P.pth'))
            torch.save(hyper_network_cp.state_dict(), join(weights_path, f'{epoch:05}_G_CP.pth'))
            torch.save(encoder_p.state_dict(), join(weights_path, f'{epoch:05}_E_P.pth'))
            torch.save(encoder_cp.state_dict(), join(weights_path, f'{epoch:05}_E_CP.pth'))
            torch.save(e_hn_optimizer_p.state_dict(), join(weights_path, f'{epoch:05}_EGoP.pth'))
            torch.save(e_hn_optimizer_cp.state_dict(), join(weights_path, f'{epoch:05}_EGoC.pth'))

            np.save(join(metrics_path, f'{epoch:05}_E_P'), np.array(losses_e_p))
            np.save(join(metrics_path, f'{epoch:05}_KLD_P'), np.array(losses_kld_p))
            np.save(join(metrics_path, f'{epoch:05}_EG_P'), np.array(losses_eg_p))

            np.save(join(metrics_path, f'{epoch:05}_E_CP'), np.array(losses_e_cp))
            np.save(join(metrics_path, f'{epoch:05}_KLD_CP'), np.array(losses_kld_cp))
            np.save(join(metrics_path, f'{epoch:05}_EG_CP'), np.array(losses_eg_cp))


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
