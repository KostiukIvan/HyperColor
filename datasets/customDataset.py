import urllib
import shutil
from os import listdir, makedirs, remove
from os.path import exists, join
from zipfile import ZipFile

import pandas as pd
import numpy as np
from torch.utils.data import Dataset


synth_id_to_category = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'watercraft',
    '04554684': 'washer', '02992529': 'cellphone'
}

category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}

class CustomDataset(Dataset):
    def __init__(self, root_dir='/home/datasets/custom', classes=[],
                 transform=None, split='train', config=None):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        if not config:
            raise ValueError("PhotogrammetryDataset JSON config is not set")

        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.test_files_path = None
        try:
            self.test_files_path = config['csv_files_dir']
        except KeyError:
            print("Parameter csv_files_dir is required in case of experimet part!")
        
        self.config = config

        pc_df = self._get_names()
        if classes:
            if classes[0] not in synth_id_to_category.keys():
                classes = [category_to_synth_id[c] for c in classes]
            pc_df = pc_df[pc_df.category.isin(classes)].reset_index(drop=True)
        else:
            classes = synth_id_to_category.keys()

        self.point_clouds_names_train = pd.concat([pc_df[pc_df['category'] == c][:int(0.9*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])
        self.point_clouds_names_valid = pd.concat([pc_df[pc_df['category'] == c][int(0.85*len(pc_df[pc_df['category'] == c])):int(0.90*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])

        # if self.test_files_path == None:
        #     self.point_clouds_names_test = pd.concat([pc_df[pc_df['category'] == c][int(0.9*len(pc_df[pc_df['category'] == c])):].reset_index(drop=True) for c in classes])
        # else:
        #     print("Predefined dataset")
        #     self.point_clouds_names_test = pd.read_csv(join(self.test_files_path, self.config["classes"][0] + ".csv"),
        #                                                  usecols = ['category', 'file_prefix'], dtype=str)

        # self.point_clouds_names_test = pd.concat([pc_df[pc_df['category'] == c][int(0.9*len(pc_df[pc_df['category'] == c])):].reset_index(drop=True) for c in classes])
        # print(f"Test data len : {len(self.point_clouds_names_test)}")
        # self.point_clouds_names_test.to_csv(join(self.test_files_path,  "full_shapenet.csv"),
        #                                                   columns = ['category', 'file_prefix'])

    def __len__(self):
        if self.split == 'train':
            pc_names = self.point_clouds_names_train
        elif self.split == 'test':
            pc_names = self.point_clouds_names_test
        elif self.split == 'valid':
            pc_names = self.point_clouds_names_valid
        else:
            raise ValueError('Invalid split. Should be train, valid or test.')
        return len(pc_names)

    def __getitem__(self, idx):
        if self.split == 'train':
            pc_names = self.point_clouds_names_train
        elif self.split == 'test':
            pc_names = self.point_clouds_names_test
        elif self.split == 'valid':
            pc_names = self.point_clouds_names_valid
        else:
            raise ValueError('Invalid split. Should be train, valid or test.')

        pc_category, pc_file_prefix = pc_names.iloc[idx].values

        pc_filedir = join(self.root_dir, pc_category)
        sample = self.load_object(directory=pc_filedir, prefix=pc_file_prefix)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_object(self, directory: str, prefix: str) -> dict: # dict<str, np.ndarray>
        suffixes = ['_mesh_data.txt', '_color_data.txt']#, '_normals_data.txt']
        parts = ['points', 'colors']#, 'normals']
        result = dict()
        drop_indices = None

        for suffix, part in zip(suffixes, parts):
            filename = prefix + suffix
            path = join(directory, filename)
            df = pd.read_csv(path,sep=' ', header=None, engine='c', )
            if part == 'colors':
                df = df.iloc[:, :-1] # drop the last column

            if part == 'points':
                df = df.reindex(columns=[0,2,1])
            result[part] = df.to_numpy()
        return result

    def _get_names(self) -> pd.DataFrame:
        file_prefixes_by_category = []
        for category_id in synth_id_to_category.keys():
            file_prefixes = { f.split('_')[0] for f in listdir(join(self.root_dir, category_id)) }

            for f_prefix in file_prefixes:
                if f_prefix not in ['.DS_Store']:
                    file_prefixes_by_category.append((category_id, f_prefix))

        return pd.DataFrame(file_prefixes_by_category, columns=['category', 'file_prefix'])

    