import os, sys
import os.path as osp
from os.path import join as ospj
import random
import pdb
import json

from src.data.dataset.loader import KITDataset

class KITSkeletonDataset():

    def __del__(self, ):
        if hasattr(self, "official_loader"):
            del self.official_loader

    def __init__(self, data_dir, data_name, split=(0.6, 0.2), preProcess_flag=False, seed=0):
        self.data_dir = data_dir
        # pdb.set_trace()
        assert os.path.isdir(self.data_dir)

        # self.split = tuple(map( float, split.split(',')))
        self.data_name = data_name
        
        self.official_loader = KITDataset(self.data_dir, self.data_name, preProcess_flag)
        # all_seq = self.official_loader._get_all_seq()
        # # print(f'\n\n\nall_seq = {len(all_seq)}\n\n\n')
        # bad_vids = self.official_loader.filter_file 
        # # print(f'\n\n\nbad_vids = {len(bad_vids)}\n\n\n')
        # all_seq = [_ for _ in all_seq if _ not in bad_vids]
        # # print(f'\n\n\nall_seq without bad_vids = {len(all_seq)}\n\n\n')
        
        # length = len(all_seq)
        # # pdb.set_trace()
        # end_train = int(length*self.split[0])
        # start_dev = end_train
        # end_dev = int(start_dev + length*self.split[1])
        # start_test = end_dev

        # if seed:
        #     random.seed(seed)
        # random.shuffle(all_seq)
        # self.train_split = all_seq[:end_train]
        # # print(f'\n\n\ntrain_split = {len(self.train_split)}\n\n\n')
        # self.validation_split = all_seq[start_dev:end_dev]
        # # print(f'\n\n\nvalidation_split = {len(self.validation_split)}\n\n\n')
        # self.test_split = all_seq[end_dev:]
        # print(f'\n\n\test_split = {len(self.test_split)}\n\n\n')

        with open(ospj(self.data_dir, self.data_name + '_data_split.json'), 'r') as handle:
            data_split = json.load(handle)

        self.train_split, self.validation_split, self.test_split = data_split['train'], data_split['val'], data_split['test']

        print(f"Data loaded with {len(self.train_split)} training, {len(self.validation_split)} validation and {len(self.test_split)} testing videos")
