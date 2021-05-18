import urllib
import shutil
from os import listdir, makedirs, remove
from os.path import exists, join
from zipfile import ZipFile

import pandas as pd
import numpy as np
from torch.utils.data import Dataset


synth_id_to_category = {
    '02691156': 'airplane', '03001627': 'chair', '02958343': 'car'
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

        self._maybe_download_data()

        pc_df = self._get_names()
        if classes:
            if classes[0] not in synth_id_to_category.keys():
                classes = [category_to_synth_id[c] for c in classes]
            pc_df = pc_df[pc_df.category.isin(classes)].reset_index(drop=True)
        else:
            classes = synth_id_to_category.keys()

        self.point_clouds_names_train = pd.concat([pc_df[pc_df['category'] == c][:int(0.9*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])
        if self.test_files_path == None:
            self.point_clouds_names_test = pd.concat([pc_df[pc_df['category'] == c][int(0.9*len(pc_df[pc_df['category'] == c])):].reset_index(drop=True) for c in classes])
        else:
            print("Predefined dataset")
            self.point_clouds_names_test = pd.read_csv(join(self.test_files_path, self.config["classes"][0] + ".csv"),
                                                         usecols = ['category', 'file_prefix'], dtype=str)

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

            #if len(df.index) > self.config['n_points']:
            #    remove_n = len(df.index) - self.config['n_points']
            #    if drop_indices is None:
            #        drop_indices = np.random.choice(df.index, remove_n, replace=False)
            #    df = df.drop(drop_indices)

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

    def _maybe_download_data(self):
        if exists(self.root_dir):
            return

        print(f'Custom Dataset doesn\'t exist in root directory {self.root_dir}. '
              f'Downloading...')
        makedirs(self.root_dir)

        file_id = '1iAq823TB1KOBLcBI2ZUkc961bBkSA1an'

        filename = 'data.zip'
        file_path = join(self.root_dir, filename)
        
        self.save_google_file(file_id=file_id, destination=file_path)

        print('Extracting...')
        with ZipFile(file_path, mode='r') as zip_f:
            zip_f.extractall(self.root_dir)

        remove(file_path)


def save_google_file(self, file_id : str, destination: str):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
   
    #get download confirmation token
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    #save data to destination file
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk) 
