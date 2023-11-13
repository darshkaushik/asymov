import os, sys, pickle
import os.path as osp
from os.path import join as ospj
from pathlib import Path

import numpy as np

# import packages.Complextext2animation.src.data as d

class KITDataset:
    """A dataset class for loading KIT MoLan dataset"""
    
    def __init__(self, data_dir, preProcess_flag=False):
        self.data_dir = data_dir
        assert os.path.exists(self.data_dir), f'Data does not exist at {self.data_dir}!'
        filter_file = os.path.join(self.data_dir, 'ignore_list.txt')
        with open(filter_file, "r") as f:
            self.filter_file = [_[:-1] for _ in f.readlines()]

        # self.kitmocap = d.KITMocap(path2data=self.data_dir, preProcess_flag=preProcess_flag)
        with open(ospj(data_dir,'xyz_data.pkl'), 'rb') as handle:
            self.xyz_data = pickle.load(handle)
        self.all_seq = self.xyz_data.keys()
        # for tup in os.walk(self.data_dir):
        #     for filename in (tup[2]):
        #         if Path(filename).suffix == '.xml':
        #             self.all_seq.append(filename.split('_')[0])

    def _get_all_seq(self):
        return self.all_seq
        
    def load_keypoint3d(self, seq_name):
        """Load a 3D keypoint sequence represented using KITML format."""

        # file_path = Path(ospj(self.data_dir, '{0}_mmm.xml'.format(str(seq_name).zfill(5))))
        # assert os.path.exists(file_path), f'File {file_path} does not exist!'
        
        # xyz_data, skel_obj, joints, root_pos, root_rot = self.kitmocap.mmm2quat(file_path)
        
        # #TODO : check normalization
        # # Normalize by root position
        # if normalize:
        #     xyz_data = xyz_data - np.transpose(root_pos.numpy(), axes=(1,0,2))
         
        return self.xyz_data[str(seq_name).zfill(5)]  # (N, 21, 3)