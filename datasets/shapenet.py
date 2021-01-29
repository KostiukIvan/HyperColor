import urllib
import shutil
from os import listdir, makedirs, remove
from os.path import exists, join
from zipfile import ZipFile

import pandas as pd
from torch.utils.data import Dataset

from utils.plyfile import load_ply

synth_id_to_category = {
    '02691156': 'airplane'
}

category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}


class ShapeNetDataset(Dataset):
    def __init__(self, root_dir='/home/datasets/shapenet', classes=[],
                 transform=None, split='train'):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        self._maybe_download_data()

        pc_df = self._get_names()
        if classes:
            if classes[0] not in synth_id_to_category.keys():
                classes = [category_to_synth_id[c] for c in classes]
            pc_df = pc_df[pc_df.category.isin(classes)].reset_index(drop=True)
        else:
            classes = synth_id_to_category.keys()

        self.point_clouds_names_train = pd.concat([pc_df[pc_df['category'] == c][:int(0.85*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])
        self.point_clouds_names_valid = pd.concat([pc_df[pc_df['category'] == c][int(0.85*len(pc_df[pc_df['category'] == c])):int(0.9*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])
        self.point_clouds_names_test = pd.concat([pc_df[pc_df['category'] == c][int(0.9*len(pc_df[pc_df['category'] == c])):].reset_index(drop=True) for c in classes])

    def __len__(self):
        if self.split == 'train':
            pc_names = self.point_clouds_names_train
        elif self.split == 'valid':
            pc_names = self.point_clouds_names_valid
        elif self.split == 'test':
            pc_names = self.point_clouds_names_test
        else:
            raise ValueError('Invalid split. Should be train, valid or test.')
        return len(pc_names)

    def __getitem__(self, idx):
        if self.split == 'train':
            pc_names = self.point_clouds_names_train
        elif self.split == 'valid':
            pc_names = self.point_clouds_names_valid
        elif self.split == 'test':
            pc_names = self.point_clouds_names_test
        else:
            raise ValueError('Invalid split. Should be train, valid or test.')

        pc_category, pc_filename = pc_names.iloc[idx].values

        pc_filepath = join(self.root_dir, pc_category, pc_filename)
        sample = load_ply(pc_filepath)

        if self.transform:
            sample = self.transform(sample)

        return sample, synth_id_to_number[pc_category]

    def _get_names(self) -> pd.DataFrame:
        filenames = []
        for category_id in synth_id_to_category.keys():
            for f in listdir(join(self.root_dir, category_id)):
                if f not in ['.DS_Store']:
                    filenames.append((category_id, f))
        return pd.DataFrame(filenames, columns=['category', 'filename'])

    def _maybe_download_data(self):
        if exists(self.root_dir):
            return

        print(f'ShapeNet doesn\'t exist in root directory {self.root_dir}. '
              f'Downloading...')
        makedirs(self.root_dir)

        url = 'https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip?dl=1'

        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2][:-5]
        file_path = join(self.root_dir, filename)
        with open(file_path, mode='wb') as f:
            d = data.read()
            f.write(d)

        print('Extracting...')
        with ZipFile(file_path, mode='r') as zip_f:
            zip_f.extractall(self.root_dir)

        remove(file_path)

        extracted_dir = join(self.root_dir,
                             'shape_net_core_uniform_samples_2048')
        for d in listdir(extracted_dir):
            shutil.move(src=join(extracted_dir, d),
                        dst=self.root_dir)

        shutil.rmtree(extracted_dir)

