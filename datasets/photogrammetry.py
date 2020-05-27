import pandas as pd
from os import listdir
from os.path import join
from torch.utils.data import Dataset

from utils.plyfile import load_photogrammetry

all_classes = ['person']

synth_id_to_number = {k : i for i, k in enumerate(all_classes)}

class PhotogrammetryDataset(Dataset):
    def __init__(self, root_dir='/home/datasets/rgb', classes=[],
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
	
        if not classes:
            classes = all_classes

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
        sample = load_photogrammetry(pc_filepath)

        if self.transform:
            sample = self.transform(sample)

        return sample, synth_id_to_number[pc_category]

    def _get_names(self) -> pd.DataFrame:
        filenames = []
        for category_id in all_classes:
            for f in listdir(join(self.root_dir, category_id)):
                filenames.append((category_id, f))
        return pd.DataFrame(filenames, columns=['category', 'filename'])

    
    def _maybe_download_data(self):
        pass
