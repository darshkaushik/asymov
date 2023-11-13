import os, sys
import os.path as osp
from os.path import join as ospj

import random

from kit_loader import KITDataset

class KITSkeletonDataset():

    def __del__(self, ):
        if hasattr(self, "official_loader"):
            del self.official_loader

    def __init__(self, data_dir, preProcess_flag=False, split=(0.6, 0.2), seed=0):
        self.data_dir = data_dir
        assert os.path.isdir(self.data_dir)
        self.split = split
        
        self.official_loader = KITDataset(self.data_dir, preProcess_flag)
        all_seq = self.official_loader._get_all_seq()
        # print(f'\n\n\nall_seq = {len(all_seq)}\n\n\n')
        bad_vids = self.official_loader.filter_file 
        # print(f'\n\n\nbad_vids = {len(bad_vids)}\n\n\n')
        all_seq = [_ for _ in all_seq if _ not in bad_vids]
        # print(f'\n\n\nall_seq without bad_vids = {len(all_seq)}\n\n\n')
        
        length = len(all_seq)
        end_train = int(length*self.split[0])
        start_dev = end_train
        end_dev = int(start_dev + length*self.split[1])
        start_test = end_dev

        if seed:
            random.seed(seed)
        random.shuffle(all_seq)
        self.train_split = all_seq[:end_train]
        # print(f'\n\n\ntrain_split = {len(self.train_split)}\n\n\n')
        self.validation_split = all_seq[start_dev:end_dev]
        # print(f'\n\n\nvalidation_split = {len(self.validation_split)}\n\n\n')
        self.test_split = all_seq[end_dev:]
        # print(f'\n\n\test_split = {len(self.test_split)}\n\n\n')

        print(f"Data loaded with {len(self.train_split)} training, {len(self.validation_split)} validation and {len(self.test_split)} testing videos")

    
