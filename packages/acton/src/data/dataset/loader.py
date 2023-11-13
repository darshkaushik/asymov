import os, sys, pickle
import os.path as osp
from os.path import join as ospj
from pathlib import Path

import numpy as np

# import packages.Complextext2animation.src.data as d

class KITDataset:
    """A dataset class for loading KIT MoLan dataset"""
    
    def __init__(self, data_dir, data_name, preProcess_flag=False):
        self.data_dir = data_dir
        assert os.path.exists(self.data_dir), f'Data does not exist at {self.data_dir}!'
        filter_file = os.path.join(self.data_dir, 'ignore_list.txt')
        with open(filter_file, "r") as f:
            self.filter_file = [_[:-1] for _ in f.readlines()]

        # self.kitmocap = d.KITMocap(path2data=self.data_dir, preProcess_flag=preProcess_flag)
        with open(ospj(data_dir, data_name + '_data.pkl'), 'rb') as handle:
            self.xyz_data = pickle.load(handle)
        self.all_seq = list(self.xyz_data.keys())
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


#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """AIST++ Dataset Loader."""
import json

import aniposelib

# 17 joints of COCO:
# 0 - nose,  1 - left_eye,  2 - right_eye, 3 - left_ear, 4 - right_ear
# 5 - left_shoulder, 6 - right_shoulder, 7 - left_elbow, 8 - right_elbow, 9 - left_wrist, 10 - right_wrist
# 11 - left_hip, 12 - right_hip, 13 - left_knee, 14 - right_knee. 15 - left_ankle, 16 - right_ankle

class AISTDataset:
    """A dataset class for loading, processing and plotting AIST++."""
    # use this link to check naming method: https://aistdancedb.ongaaccel.jp/data_formats/

    VIEWS = ['c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09']

    def __init__(self, anno_dir):
        assert os.path.exists(anno_dir), f'Data does not exist at {anno_dir}!'

        # Init paths
        self.camera_dir = os.path.join(anno_dir, 'cameras/')
        self.motion_dir = os.path.join(anno_dir, 'motions/')
        self.keypoint3d_dir = os.path.join(anno_dir, 'keypoints3d/')
        self.keypoint2d_dir = os.path.join(anno_dir, 'keypoints2d/')
        self.splits_dir = os.path.join(anno_dir, 'splits/')
        filter_file = os.path.join(anno_dir, 'ignore_list.txt')
        with open(filter_file, "r") as f:
            self.filter_file = [_[:-1] for _ in f.readlines()]

        # Load environment setting mapping
        self.mapping_seq2env = {}  # sequence name -> env name
        self.mapping_env2seq = {}  # env name -> a list of sequence names
        env_mapping_file = os.path.join(self.camera_dir, 'mapping.txt')
        env_mapping = np.loadtxt(env_mapping_file, dtype=str)
        for seq_name, env_name in env_mapping:
            self.mapping_seq2env[seq_name] = env_name
            if env_name not in self.mapping_env2seq:
                self.mapping_env2seq[env_name] = []
            self.mapping_env2seq[env_name].append(seq_name)

    @classmethod
    def get_video_name(cls, seq_name, view):
        """Get AIST video name from AIST++ sequence name."""
        return seq_name.replace('cAll', view)

    @classmethod
    def get_seq_name(cls, video_name):
        """Get AIST++ sequence name from AIST video name."""
        tags = video_name.split('_')
        if len(tags) == 3:
            view = tags[1]
            tags[1] = 'cAll'
        else:
            view = tags[2]
            tags[2] = 'cAll'
        return '_'.join(tags), view

    @classmethod
    def load_camera_group(cls, camera_dir, env_name):
        """Load a set of cameras in the environment."""
        file_path = os.path.join(camera_dir, f'{env_name}.json')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        with open(file_path, 'r') as f:
            params = json.load(f)
        cameras = []
        for param_dict in params:
            camera = aniposelib.cameras.Camera(name=param_dict['name'],
                                               size=param_dict['size'],
                                               matrix=param_dict['matrix'],
                                               rvec=param_dict['rotation'],
                                               tvec=param_dict['translation'],
                                               dist=param_dict['distortions'])
            cameras.append(camera)
        camera_group = aniposelib.cameras.CameraGroup(cameras)
        return camera_group

    @classmethod
    def load_motion(cls, motion_dir, seq_name):
        """Load a motion sequence represented using SMPL format."""
        file_path = os.path.join(motion_dir, f'{seq_name}.pkl')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        smpl_poses = data['smpl_poses']  # (N, 24, 3)
        smpl_scaling = data['smpl_scaling']  # (1,)
        smpl_trans = data['smpl_trans']  # (N, 3)
        return smpl_poses, smpl_scaling, smpl_trans

    #   @classmethod
    def load_keypoint3d(self, seq_name, use_optim=True):
        """Load a 3D keypoint sequence represented using COCO format."""
        file_path = os.path.join(self.keypoint3d_dir, f'{seq_name}.pkl')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if use_optim:
            return data['keypoints3d_optim']  # (N, 17, 3)
        else:
            return data['keypoints3d']  # (N, 17, 3)

    @classmethod
    def load_keypoint2d(cls, keypoint_dir, seq_name):
        """Load a 2D keypoint sequence represented using COCO format."""
        file_path = os.path.join(keypoint_dir, f'{seq_name}.pkl')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        keypoints2d = data['keypoints2d']  # (nviews, N, 17, 3)
        det_scores = data['det_scores']  # (nviews, N)
        timestamps = data['timestamps']  # (N,)
        return keypoints2d, det_scores, timestamps
