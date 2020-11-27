import pandas as pd
import numpy as np
import random
import json
import argparse
from random import randint
from numpy import median
from os import listdir
from os.path import join
from torch.utils.data import Dataset
from utils.plyfile import load_photogrammetry

all_classes = ['person', 'boat','lamp']

synth_id_to_number = {k : i for i, k in enumerate(all_classes)}

class PhotogrammetryDataset(Dataset):
    def __init__(self, root_dir='/home/datasets/rgb', classes=[],
                 transform=None, split='train', config=None):
        """
        Args:
            root_dir (string): Directory with all the point clouds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        if not config:
            raise ValueError("PhotogrammetryDataset JSON config is not set")
            
        self.config = config

        self._maybe_download_data()

        pc_df = self._get_names()
	
        if not classes:
            classes = all_classes

        self.point_clouds_names_train = pd.concat([pc_df[pc_df['category'] == c][:int(0.85*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])
        self.point_clouds_names_valid = pd.concat([pc_df[pc_df['category'] == c][int(0.85*len(pc_df[pc_df['category'] == c])):int(0.9*len(pc_df[pc_df['category'] == c]))].reset_index(drop=True) for c in classes])
        self.point_clouds_names_test = pd.concat([pc_df[pc_df['category'] == c][int(0.9*len(pc_df[pc_df['category'] == c])):].reset_index(drop=True) for c in classes])

        self.random_normalization()
	
   
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

    def get_list_of_samples_len(self):
        sample_len_list = []
        
        if self.split == 'train':
            pc_names = self.point_clouds_names_train
        elif self.split == 'valid':
            pc_names = self.point_clouds_names_valid
        elif self.split == 'test':
            pc_names = self.point_clouds_names_test
        else:
            raise ValueError('Invalid split. Should be train, valid or test.')
    

        for idx in range(0,len(self)):
            pc_category, pc_filename = pc_names.iloc[idx].values
            pc_filepath = join(self.root_dir, pc_category, pc_filename)

            sample = load_photogrammetry(pc_filepath)
            sample_len_list.append(len(sample))
        return sample_len_list
    
    def add_random_points_to_object(self, sample, size ):
        missing_len = size - len(sample)
        end = len(sample) - 1
        random_indexes = [ randint(0, end ) for x in range(0, missing_len.astype(int)) ]
        #duplicate random indexes values
        for idx in random_indexes:
            tmp = sample[idx]
            sample = np.append(sample, [tmp], axis = 0)
        
        print("ADD sample lengh: " + str(len(sample)))
        return sample

    def remove_random_pointe_from_object(self, sample, size):
        oversize_len = len(sample) - size
        #random_indexes = [ randint(0, len(sample) - 1) for x in range(0, oversize_len) ]
        random_indexes = random.sample(range(len(sample) - 1), oversize_len)
        random_indexes.sort(reverse=True)
        for idx in random_indexes:
            sample = np.delete(sample, idx, 0)
        return sample

    def save_sample_at(self, file_path, sample):
        with open(file_path, "r") as f:
            file_data = f.read()
	
        np.savetxt(file_path, sample, delimiter=' ', fmt='%.4f')


    def random_normalization(self):
        sample_len = self.get_list_of_samples_len()
        #median_size = median(sample_len)
        #median_size = median_size.astype(int)
        median_size = self.config['n_points']
        #print("Median : " + median_size.astype(str))	
        print("Median : " + str(median_size))  
        
        if self.split == 'train':
            pc_names = self.point_clouds_names_train
        elif self.split == 'valid':
            pc_names = self.point_clouds_names_valid
        elif self.split == 'test':
            pc_names = self.point_clouds_names_test
        else:
            raise ValueError('Invalid split. Should be train, valid or test.')


        for idx in range(0,len(self)):
            pc_category, pc_filename = pc_names.iloc[idx].values
            pc_filepath = join(self.root_dir, pc_category, pc_filename)
            print("Processing file : " + pc_filepath)
            		
            sample = load_photogrammetry(pc_filepath)
            
            print("File length before processing  : " + str(len(sample)) + str(sample.shape))

            if len(sample) < median_size:
                sample = self.add_random_points_to_object(sample, median_size)
            elif len(sample) == median_size:
                print("Do nothing")
            else:
                sample = self.remove_random_pointe_from_object(sample, median_size)
            print("File length after processing  : " + str(len(sample)) + str(sample.shape))
            self.save_sample_at(pc_filepath, sample)
            print("File was saved")


    def _maybe_download_data(self):
        pass
