import argparse
import json
import re
from datetime import datetime
from os import listdir
from os.path import join
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models import aae
from sklearn.neighbors import KNeighborsClassifier
import skimage.color as colors

from utils.points import generate_points
from utils.metrics_3_3 import jsd_between_point_cloud_sets
from utils.util import set_seed, cuda_setup, get_weights_dir, find_latest_epoch


"""
    AIR P/C: 0.002229
    CAR P/C: 0.00056, (0.12731111111111112, 0.2229)
    CHAIR P/C: 0.006 , 0.1 0.299 / 0.06888
"""
p_std = 0.002229
cp_std = 0.06888
n_points=1024


def _get_epochs_by_regex(path, regex):
    reg = re.compile(regex)
    return {int(w[:5]) for w in listdir(path) if reg.match(w)}


def jsd(config, weights_path, device):
    print('Evaluating Jensen-Shannon divergences on validation set on all saved epochs.')

    # Find all epochs that have saved model weights
    e_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})_E_P\.pth')
    g_epochs = _get_epochs_by_regex(weights_path, r'(?P<epoch>\d{5})_G_P\.pth')
    epochs = sorted(e_epochs.intersection(g_epochs))

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], split='valid')
    elif dataset_name == "custom":
        # import pdb; pdb.set_trace()
        from datasets.customDataset import CustomDataset
        dataset = CustomDataset(root_dir=config['data_dir'],
                                classes=config['classes'],
                                split='test',
                                config=config)
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet`. Got: `{dataset_name}`')

    classes_selected = ('all' if not config['classes']
                        else ','.join(config['classes']))
    print(f'Valid dataset. Selected {classes_selected} classes. Loaded {len(dataset)} '
          f'samples.')

    distribution = config['metrics']['distribution']
    assert distribution in ['normal', 'beta'], 'Invalid distribution. Choose normal or beta'

    #
    # Models
    #
    # hyper_network = aae.HyperNetwork(config, device).to(device)
    hyper_network_p = aae.PointsHyperNetwork(config, device).to(device)
    hyper_network_cp = aae.ColorsAndPointsHyperNetwork(config, device).to(device)

    hyper_network_p.eval()
    hyper_network_cp.eval()

    num_samples = len(dataset.point_clouds_names_test)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    point_data = next(iter(data_loader))
                    
    if dataset_name == "custom":
        X = torch.cat((point_data['points'], point_data['colors']), dim=2)
        X = X.to(device, dtype=torch.float)

    else: 
        X, _ = point_data
        X = X.to(device, dtype=torch.float)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    # noise = torch.zeros(3 * X.shape[0], config['z_size']).to(device)
    points_noise = torch.zeros(3 * X.shape[0], config['z_size'])
    colors_noise = torch.zeros(3 * X.shape[0], config['z_size'])

    
    n_last_epochs = config['metrics'].get('jsd_how_many_last_epochs', -1)
    epochs = epochs[-n_last_epochs:] if n_last_epochs > 0 else epochs
    print(f'Testing epochs: {epochs}')
    choice = np.random.randint(0, X.shape[1], size=n_points)
    X = X[:, choice, :]

    #for p_std in np.linspace(0.1, 1e-4, 10):
    #for cp_std in np.linspace(0.08, 0.06, 10):
    results_p = {}
    results_cp = {}
    for epoch in reversed(epochs):
        try:
            # hyper_network.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

            hyper_network_p.load_state_dict(torch.load(
                join(weights_path, f'{epoch:05}_G_P.pth')))
            hyper_network_cp.load_state_dict(torch.load(
                join(weights_path, f'{epoch:05}_G_CP.pth')))

            start_clock = datetime.now()

            # We average JSD computation from 3 independent trials.
            js_results_p = []
            js_results_cp = []
            for _ in range(3):
                if distribution == 'normal':
                    points_noise.normal_(config['metrics']['normal_mu_p'], p_std)
                    colors_noise.normal_(config['metrics']['normal_mu_cp'], cp_std)
                    points_noise = points_noise.to(device)
                    colors_noise = colors_noise.to(device)
                elif distribution == 'beta':
                    noise_np = np.random.beta(config['metrics']['beta_a'], config['metrics']['beta_b'], noise.shape)
                    noise = torch.tensor(noise_np).float().round().to(device)
                
                with torch.no_grad():
                    target_networks_points_weights = hyper_network_p(points_noise)
                    target_networks_colors_weights = hyper_network_cp(colors_noise)
    
                    X_rec = torch.zeros(3 * X.shape[0], n_points, X.shape[2]).to(device)    

                    # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
                    if X_rec.size(-1) == 3 or X_rec.size(-1) == 6 or X_rec.size(-1) == 7:
                        X_rec.transpose_(X_rec.dim() - 2, X_rec.dim() - 1)

                    for j, (weights_points, weights_color) in enumerate(zip(target_networks_points_weights, target_networks_colors_weights)):
                        target_network_p = aae.TargetNetwork(config, weights_points).to(device)
                        target_network_cp = aae.ColorsAndPointsTargetNetwork(config, weights_color).to(device)
                        
                        target_network_input = generate_points(config=config, epoch=epoch, size=(n_points, 3)).to(device)

                        pred_points = target_network_p(target_network_input.to(device, dtype=torch.float)).transpose(0, 1) # [4096, 3]
                        pred_colors = target_network_cp(target_network_input.to(device, dtype=torch.float)) # [4096, 3]
                        
                        pred_colors = torch.from_numpy(colors.lab2xyz(pred_colors.cpu().numpy()).transpose()).to(device)

                        X_rec[j] = torch.cat([pred_points, pred_colors], dim=0) # [B,6,N]

                                        
                jsd_p = jsd_between_point_cloud_sets(X[:, :, :3].cpu().numpy(), X_rec.transpose(1, 2)[:, :, :3].cpu().numpy(), color=False)
                jsd_cp = jsd_between_point_cloud_sets(X[:, :, 3:6].cpu().numpy(), X_rec.transpose(1, 2)[:, :, 3:6].cpu().numpy(), color=True)
                js_results_p.append(jsd_p)
                js_results_cp.append(jsd_cp)

            js_result_p = np.mean(js_results_p)
            print(f'Epoch: {epoch} JSD: {js_result_p: .6f} '
                f'Time: {datetime.now() - start_clock}')

            js_result_cp = np.mean(js_results_cp)
            print(f'Epoch: {epoch} JSD: {js_result_cp: .6f} '
                f'Time: {datetime.now() - start_clock}')
            results_p[epoch] = js_result_p
            results_cp[epoch] = js_result_cp
        except KeyboardInterrupt:
            print(f'Interrupted during epoch: {epoch}')
            break
    print(f"========== {p_std}, {cp_std}")
    results_p = pd.DataFrame.from_dict(results_p, orient='index', columns=['jsd'])
    results_cp = pd.DataFrame.from_dict(results_cp, orient='index', columns=['jsd'])
    print(f"Minimum JSD P at epoch {results_p.idxmin()['jsd']}: "
        f"{results_p.min()['jsd']: .6f}")
    
    print(f"Minimum JSD CP at epoch {results_cp.idxmin()['jsd']}: "
        f"{results_cp.min()['jsd']: .6f}")


    return results_p.idxmin()['jsd'], results_p.min()['jsd'], results_cp.idxmin()['jsd'], results_cp.min()['jsd']


def minimum_matching_distance(config, weights_path, device):
    from utils.metrics import EMD_CD
    print('Minimum Matching Distance (MMD) Test split')
    epoch = find_latest_epoch(weights_path)
    print(f'Last Epoch: {epoch}')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], split='test')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not config['classes']
                        else ','.join(config['classes']))
    print(f'Test dataset. Selected {classes_selected} classes. Loaded {len(dataset)} '
          f'samples.')

    #
    # Models
    #
    hyper_network = aae.HyperNetwork(config, device).to(device)
    encoder = aae.Encoder(config).to(device)

    hyper_network.eval()
    encoder.eval()

    num_samples = len(dataset.point_clouds_names_test)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    encoder.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))
    hyper_network.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    result = {}

    for point_data in data_loader:

        X, _ = point_data
        X = X.to(device)

        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3:
            X.transpose_(X.dim() - 2, X.dim() - 1)

        with torch.no_grad():
            z_a, _, _ = encoder(X)
            target_networks_weights = hyper_network(z_a)

            X_rec = torch.zeros(X.shape).to(device)
            for j, target_network_weights in enumerate(target_networks_weights):
                target_network = aae.TargetNetwork(config, target_network_weights).to(device)

                target_network_input = generate_points(config=config, epoch=epoch, size=(X.shape[2], X.shape[1]))

                X_rec[j] = torch.transpose(target_network(target_network_input.to(device)), 0, 1)

            for k, v in EMD_CD(torch.transpose(X, 1, 2).contiguous(),
                               torch.transpose(X_rec, 1, 2).contiguous(), X.shape[0]).items():
                result[k] = result.get(k, 0.0) + v.item()

    print(result)



def all_metrics(config, weights_path, device, epoch, jsd_value_p, jsd_value_cp):
    from utils.metrics_3_3 import compute_all_metrics, compute_all_metrics_colors, sort_data_by_points
    print('All metrics')
    if epoch is None:
        print('Finding latest epoch...')
        epoch = find_latest_epoch(weights_path)
        print(f'Epoch: {epoch}')

    if jsd_value_p is not None:
        print(f'Best Epoch selected via mimnimal JSD: {epoch}')

    #
    # Dataset
    #
    dataset_name = config['dataset'].lower()
    if dataset_name == 'shapenet':
        from datasets.shapenet import ShapeNetDataset
        dataset = ShapeNetDataset(root_dir=config['data_dir'], classes=config['classes'], split='test')
    elif dataset_name == "custom":
        # import pdb; pdb.set_trace()
        from datasets.customDataset import CustomDataset
        dataset = CustomDataset(root_dir=config['data_dir'],
                                classes=config['classes'],
                                split='test',
                                config=config)
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not config['classes']
                        else ','.join(config['classes']))
    print(f'Test dataset. Selected {classes_selected} classes. Loaded {len(dataset)} '
          f'samples.')

    distribution = config['metrics']['distribution']
    assert distribution in ['normal', 'beta'], 'Invalid distribution. Choose normal or beta'

    #
    # Models
    #
    hyper_network_p = aae.PointsHyperNetwork(config, device).to(device)
    hyper_network_cp = aae.ColorsAndPointsHyperNetwork(config, device).to(device)

    hyper_network_p.eval()
    hyper_network_cp.eval()


    data_loader = DataLoader(dataset, batch_size=32, 
                                shuffle=False, num_workers=4,
                                drop_last=False, pin_memory=True)

    # hyper_network.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    hyper_network_p.load_state_dict(torch.load(
        join(weights_path, f'{epoch:05}_G_P.pth')))
    hyper_network_cp.load_state_dict(torch.load(
        join(weights_path, f'{epoch:05}_G_CP.pth')))


    start_clock = datetime.now()

    #for p_std in np.linspace(0.01, 1e-5, 10): 
    #for cp_std in np.linspace(1, 1e-1, 10):
    result_p = {}
    result_cp = {}
    size = 0
    for point_data in data_loader:
        
        
        if dataset_name == "custom":
            X = torch.cat((point_data['points'], point_data['colors']), dim=2)
            X = X.to(device, dtype=torch.float)

        else: 
            X, _ = point_data
            X = X.to(device, dtype=torch.float)

        
        # Change dim [BATCH, N_POINTS, N_DIM] -> [BATCH, N_DIM, N_POINTS]
        if X.size(-1) == 3 or X.size(-1) == 6 or X.size(-1) == 7:
            X.transpose_(X.dim() - 2, X.dim() - 1)

        choice = np.random.randint(0, X.shape[2], size=n_points)
        X = X[:, :, choice]  

        with torch.no_grad():
            points_noise = torch.zeros(X.shape[0], config['z_size'])
            colors_noise = torch.zeros(X.shape[0], config['z_size'])

            if distribution == 'normal':
                points_noise.normal_(config['metrics']['normal_mu_p'], p_std)
                colors_noise.normal_(config['metrics']['normal_mu_cp'], cp_std)
                points_noise = points_noise.to(device)
                colors_noise = colors_noise.to(device)
            elif distribution == 'beta':
                noise_np = np.random.beta(config['metrics']['beta_a'], config['metrics']['beta_b'], noise.shape)
                noise = torch.tensor(noise_np).float().round().to(device)

            target_networks_points_weights = hyper_network_p(points_noise)
            target_networks_colors_weights = hyper_network_cp(colors_noise)

            X_rec = torch.zeros(X.shape[0], X.shape[1], n_points).to(device)
            for j, (weights_points, weights_color) in enumerate(zip(target_networks_points_weights, target_networks_colors_weights)):
                target_network_p = aae.TargetNetwork(config, weights_points).to(device)
                target_network_cp = aae.ColorsAndPointsTargetNetwork(config, weights_color).to(device)

                target_network_input = generate_points(config=config, epoch=epoch, size=(n_points, 3)).to(device)

                pred_points = target_network_p(target_network_input.to(device, dtype=torch.float)) # [4096, 3]
                pred_colors = target_network_cp(target_network_input.to(device, dtype=torch.float)) # [4096, 3]
                
                pred_colors = torch.from_numpy(colors.lab2xyz(pred_colors.cpu().numpy()).transpose()).transpose(0, 1).to(device)

                X_rec[j] = torch.cat([pred_points, pred_colors], dim=1).transpose(0, 1) # [B,6,N]

            X = X.transpose(1,2)
            X_rec = X_rec.transpose(1,2)
            
            X = torch.from_numpy(sort_data_by_points(X.cpu().numpy())).to(device)
            X_rec = torch.from_numpy(sort_data_by_points(X_rec.cpu().numpy())).to(device)
        

            for k, v in compute_all_metrics(X[:, :, :3], X_rec[:, :, :3], X.shape[0]).items():
                result_p[k] = (size * result_p.get(k, 0.0) + X.shape[0] * v.item()) / (size + X.shape[0])

            for k, v in compute_all_metrics_colors(X[:, :, 3:6], X_rec[:, :, 3:6], X.shape[0]).items():
                result_cp[k] = (size * result_cp.get(k, 0.0) + X.shape[0] * v.item()) / (size + X.shape[0])

        size += X.shape[0]
    print(f"======== {p_std} {cp_std}")
    result_p['jsd'] = jsd_value_p
    print(f'Time: {datetime.now() - start_clock}')
    print(f'Result p:')
    for k, v in result_p.items():
        print(f'{k}: {v}')

    result_cp['jsd'] = jsd_value_cp
    print(f'Result cp:')
    for k, v in result_cp.items():
        print(f'{k}: {v}')
    
    print("\n\n\n\n")


def main(config):
    set_seed(config['seed'])
    weights_path = get_weights_dir(config)

    device = cuda_setup(config['cuda'], config['gpu'])
    print(f'Device variable: {device}')
    if device.type == 'cuda':
        print(f'Current CUDA device: {torch.cuda.current_device()}')

    print('\n')
    all_metrics(config, weights_path, device, None, None, None)
    print('\n')

    set_seed(config['seed'])
    jsd_epoch_p, jsd_value_p, jsd_epoch_cp, jsd_value_cp = jsd(config, weights_path, device)
    print('\n')

    set_seed(config['seed'])
    all_metrics(config, weights_path, device, jsd_epoch_p, jsd_value_p, jsd_value_cp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str, help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)
