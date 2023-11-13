#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import os, sys
import os.path as osp
from os.path import join as ospj
from os.path import basename as ospb
import pdb
from tqdm import tqdm
import pickle
from multiprocessing import cpu_count, Pool, Process
from functools import partial

import random
import numpy as np
import pandas as pd
import math
from torch.nn.functional import interpolate as intrp

import subprocess
import shutil
# import wandb
import uuid
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy.ndimage import uniform_filter1d, spline_filter1d

from pathlib import Path
import utils
from viz_utils import *
from itertools import groupby, product
import hydra  # https://hydra.cc/docs/intro/
from omegaconf import DictConfig, OmegaConf  # https://github.com/omry/omegaconf
# from benedict import benedict as bd  # https://github.com/fabiocaccamo/python-benedict

# sys.path.append('packages/TEMOS')
# from  packages.TEMOS import sample_asymov_for_viz

"""
Visualize input and output motion sequences and labels
"""



#cluster2vid--------------------------------------------------------------------
def cluster2vid(save_name, clusters_idx, sk_type, proxy_center_info_path, data_path, frames_dir, gt_fps:float=100.0, cons_fps:float=25.0, duration=0.5, force=False):
    '''
    Args:
        clusters_idx : cluster indices to visualize
        sk_type (str): {'kitml', 'coco17'}
        proxy_center_info_path : path to pickled Dataframe containing sequence name and frame of proxy centers or the DataFrame itself
        data_path : path to the pickled dictionary containing the per frame ground truth 3d keypoints of skeleton joints of the specified video sequence name
                    or the GT dictionary itself
        frames_dir : Path to root folder that will contain frames folder
        gt_fps : Ground truth framerate. Default = 100.0 (KIT)
        cons_fps (float): Construction output framerate. Default = 25.0
        duration : the duration (in secs) for which the cluster visualization should last
        force : If True, visualize all clusters overwriting existing ones. Defaults to False, visualizing only those whose .mp4 videos do not already exist.

    Return:
        None. Path of mp4 video: frames_dir/{cluster_idx}/video.mp4
    '''
    # from packages.acton.src.data.dataset.loader import KITDataset
    # pdb.set_trace()

    #get GT keypoints to visualize
    if type(data_path) == dict:
        gt_data = data_path
    else:
        with open(data_path, 'rb') as handle:
            gt_data = pickle.load(handle)
    # support frames on each side of center frame
    support_frames_count = int((gt_fps*duration-1)/2) #-1 for center frame

    #get proxy center info
    if type(proxy_center_info_path) == pd.DataFrame:
        proxy_center_info = proxy_center_info_path
    else:
        proxy_center_info = pd.read_pickle(proxy_center_info_path)
    center_frames_idx, seq_names = proxy_center_info.loc[clusters_idx, 'frame_index'], proxy_center_info.loc[clusters_idx, 'seq_name']

    seqs = []
    for cluster_idx, center_frame_idx, seq_name in tqdm(zip(clusters_idx, center_frames_idx, seq_names), desc='clusters', total = len(seq_names)):
        seq_complete = gt_data[seq_name]
        seq = seq_complete[max(0, center_frame_idx-support_frames_count):min(seq_complete.shape[0], center_frame_idx+support_frames_count+1)]
        seqs.append(change_fps(seq, gt_fps, cons_fps))
    #visualize the required fragment of complete sequence
    viz_l_seqs([str(i) for i in clusters_idx], seqs, ospj(frames_dir, str(cluster_idx)), sk_type, cons_fps, force)
#-------------------------------------------------------------------------------

#cluster_seq2vid----------------------------------------------------------------
def cluster_seq2vid(save_name, cluster_seq, cluster2keypoint_mapping_path, frames_dir, sk_type, cons_fps:float=25.0):
    '''
    Maps sequence of clusters to proxy center keypoints and visualizes into an mp4 video.

    Args:
        cluster_seq : Array of cluster indices per frame
        cluster2keypoint_mapping_path : Path to pickled dataframe containing the mapping of cluster to proxy center keypoints
        frames_dir : Path to root folder that will contain frames folder
        sk_type : {'smpl', 'nturgbd', 'kitml', 'coco17'}
        cons_fps (float): Desired frame rate. Default = 25.0

    Return:
        None. Path of mp4 video: frames_dir/video.mp4
    '''

    cluster2keypoint = pd.read_pickle(cluster2keypoint_mapping_path)
    skeleton_keypoints = np.array([cluster2keypoint.loc[i,'keypoints3d'] for i in cluster_seq])
    
    viz_l_seqs([save_name], [skeleton_keypoints], frames_dir, sk_type, cons_fps).join()
#-------------------------------------------------------------------------------

#pred reconstruction-----------------------------------------------------------------
def reconstruction(recons_type, filters, seq_names, data_path, sk_type, pred_fps:float=12.5, gt_fps:float=100.0, recons_fps:float=25.0, frames_dir=None, viz_names=None, force=False, **kwargs):
    '''
    Args:
        recons_type (str) : reconstruction technique to be used
        filters (List[str]) : smoothing filters to apply on reconstructions. Use string 'none' for no filter.
        seq_names (List[str]): name of video sequences to reconstruct
        data_path : path to the pickled dictionary containing the per frame ground truth 3d keypoints of skeleton joints of the specified video sequence name
                    or the GT dictionary itself
        sk_type : {'smpl', 'nturgbd', 'kitml', 'coco17'}
        pred_fps : Model prediction framerate. Default = 12.5
        gt_fps : Ground truth framerate. Default = 100.0 (KIT)
        recons_fps (float): Reconstruction output frame rate. Default = 25.0
        frames_dir : Path to root folder that will contain frames folder for visualization. If None, won't create visualization.
        viz_names (List[str]): name of video sequences to visualize. Defaults to 'seq_names' argument. Pass [] to not visualize any.
        force : If True, visualize all viz_names overwriting existing ones. Defaults to False, visualizing only those whose .mp4 videos do not already exist.
        **kwargs: Must contain
            if recons_type == 'naive_no_rep' or 'naive':
                contiguous_frame2cluster_mapping_path : Path to pickled dataframe containing the mapping of contiguous frames in a video to a cluster
                cluster2frame_mapping_path : Path to pickled dataframe containing the mapping of cluster to the proxy center frame (and the video sequence containing it)
            if recons_type == 'very_naive':
                cluster2keypoint_mapping_path : Path to pickled dataframe containing the mapping of cluster to proxy center keypoints
                frame2cluster_mapping_path : Path to pickled dataframe containing the mapping of each frame in a video to a cluster.
                frame2cluster_mapping_dir: Path of directory containing .npy files for TEMOS-asymov variant.
    '''

    # if fps is None:
    #     if sk_type == 'kitml':
    #         fps = 25.0
    #     elif sk_type == 'coco17':
    #         fps = 60.0
    #     else:
    #         fps = 30.0

    print('----------------------------------------------------')
    # print(recons_type+'_reconstruction')
    if type(data_path) == dict:
        gt_data = data_path
    else:
        with open(data_path, 'rb') as handle:
            gt_data = pickle.load(handle)

    if recons_type == 'naive_no_rep' or recons_type  == 'naive':  
        contiguous_frame2cluster_mapping_path = kwargs['contiguous_frame2cluster_mapping_path']
        cluster2frame_mapping_path = kwargs['cluster2frame_mapping_path']

        contiguous_frame2cluster = contiguous_frame2cluster_mapping_path
        if type(contiguous_frame2cluster_mapping_path) != pd.DataFrame:
            contiguous_frame2cluster = pd.read_pickle(contiguous_frame2cluster_mapping_path)
        contiguous_cluster_seqs = [contiguous_frame2cluster[contiguous_frame2cluster['name']==name][['cluster', 'length']].reset_index() for name in seq_names]

        output = eval(recons_type+'_reconstruction')(seq_names, contiguous_cluster_seqs,  gt_data, cluster2frame_mapping_path, pred_fps/gt_fps)
        if recons_type == 'naive_no_rep':
            recons, faulty = output
        else:
            recons = output

    elif recons_type == 'very_naive':
        cluster2keypoint_mapping_path = kwargs['cluster2keypoint_mapping_path']
        if 'frame2cluster_mapping_path' in kwargs.keys():
            frame2cluster_mapping_path = kwargs['frame2cluster_mapping_path']
        else:
            frame2cluster_mapping_path = None
        if 'frame2cluster_mapping_dir' in kwargs.keys():
            frame2cluster_mapping_dir = kwargs['frame2cluster_mapping_dir']
        else:
            frame2cluster_mapping_dir = None

        if frame2cluster_mapping_path is not None:
            frame2cluster = pd.read_pickle(frame2cluster_mapping_path)
            cluster_seqs = [frame2cluster[frame2cluster['seq_name']==name]['cluster'] for name in seq_names]
        elif frame2cluster_mapping_dir is not None:
            cluster_seqs = [np.load(os.path.join(frame2cluster_mapping_dir, f"{name}.npy")) for name in seq_names]
        else:
            ValueError('frame2cluster not given')

        recons = eval(recons_type+'_reconstruction')(seq_names, cluster_seqs, cluster2keypoint_mapping_path)

    # recons and gt in desired fps
    gt = [gt_data[name][:5000, :, :] for name in seq_names]
    gt_in_recons_fps = [change_fps(keypoint, gt_fps, recons_fps) for keypoint in gt]
    recons = [change_fps(keypoint, pred_fps, recons_fps) for keypoint in recons]

    mpjpe={}
    print('----------------------------------------------------')
    # print("MPJPE")
    for filter in filters:
        if filter == 'none':
            pass
        elif filter == 'spline':
            recons = [spline_filter1d(keypoint, axis=0) for keypoint in recons]
        elif filter == 'uniform':
            recons = [uniform_filter1d(keypoint, size=int(recons_fps/4), axis=0) for keypoint in recons]
        else :
            raise NameError(f'No such filter {filter}')
        print(f"Using {filter} filter")

        mpjpe_per_sequence=mpjpe3d(seq_names, recons, gt_in_recons_fps)
        mpjpe_mean = np.mean(mpjpe_per_sequence)
        mpjpe['filter'] = mpjpe_mean
        print(f'{recons_type}_{filter} mpjpe: ', mpjpe_mean)

        if frames_dir is not None:
            if viz_names is None:
                viz_names = seq_names

            if filter == 'none':
                frames_dir_temp = frames_dir / f"{recons_type}"
            else:
                frames_dir_temp = frames_dir / f"{recons_type}_{filter}"

            viz_l_seqs(viz_names, recons, frames_dir_temp, sk_type, recons_fps, force)
        print('----------------------------------------------------')
    print('----------------------------------------------------')
    # if per_seq_score:
    #     return np.mean(mpjpe_per_sequence), mpjpe_per_sequence
    # else:
    #     return np.mean(mpjpe_per_sequence)
    return mpjpe
#-------------------------------------------------------------------------------

#gt construction-----------------------------------------------------------------
def ground_truth_construction(seq_names, data_path, sk_type='kitml', gt_fps:float=100.0, cons_fps:float=25.0, frames_dir=None, force=False):
    '''
    Constructs original video from ground truth sequences, which are used as reference for mpjpe calculation.

    Args:
        seq_names : name of video sequences to construct and visualize ground truth
        data_path : path to the pickled dictionary containing the per frame ground truth 3d keypoints of skeleton joints of the specified video sequence name
                    or the GT dictionary itself
        sk_type : {'smpl', 'nturgbd', 'kitml', 'coco17'}
        gt_fps : Ground truth framerate. Default = 100.0 (KIT)
        cons_fps (float): Construction output framerate. Default = 25.0
        frames_dir : Path to root folder that will contain frames folder for visualization.
        force (Bool): If True visualizes all sequences even if already exists in frames_dir.
                    Defaults to False, only visualizes incomplete or un-visualized sequences.
    Returns:
        None.
        The reconstructed videos are saved in {frames_dir}/{seq_name} as {seq_name}.mp4
    '''
    assert frames_dir is not None, "path to store gt visualizations absent"

    if type(data_path) == dict:
        gt_data = data_path
    else:
        with open(data_path, 'rb') as handle:
            gt_data = pickle.load(handle)

    print('----------------------------------------------------')
    #TODO: remove 5000 limit
    gt = [change_fps(gt_data[name][:5000, :, :], gt_fps, cons_fps)
          for name in tqdm(seq_names, 'Ground Truth construction')]
    print('----------------------------------------------------')

    viz_l_seqs(seq_names, gt, frames_dir, sk_type, cons_fps, force)
    print('----------------------------------------------------')
#-------------------------------------------------------------------------------

#aggregate reconstruction methods-----------------------------------------------------------------
"""
Class that aggregates all visualization functions and data required by them.
"""

class Viz:

    def __init__(self, cfg: DictConfig):
        '''

        Example:
            >>> cfg_p = '~/asymov/...'
            >>> viz_obj = Viz(cfg_p)
        '''
        # Load config from path defined in global var CFG_PATH. # TODO: Arg.
        self.cfg = cfg
        # print(self.cfg)

        # Load data-structures required for viz.
        self.data = {}
        for f in self.cfg.data_fnames:
            fp = str(Path(self.cfg.datapath, self.cfg.data_fnames[f]))
            self.data[f] = utils.read_pickle(fp)

        # Init. member vars
        self.l_samples = []
        self.n_samples = -1
        self.og_split_file_p = Path(self.cfg.splitpath, self.cfg.split)
        self.og_l_samples = utils.read_textf(self.og_split_file_p, ret_type='list')
        self.og_n_samples = len(self.og_l_samples)


    def _get_l_samples(self, l_samples, n_samples):
        '''
        Args:
            l_samples <list>: of {<int>, <str>} of sample IDs to be viz'd.
        If empty, random `n_samples` are viz'd.
            n_samples <int>: Number of samples to be viz'd. If `l_samples` is
        empty, then random samples are chosen. If not, first `n_samples`
        elements of l_samples is chosen.
        '''
        # l_samples not given. Viz. random seqs.
        if len(l_samples) == 0:
            if n_samples < 0:
                raise TypeError('Either provide list of samples or # samples.')
            else:
                # Random sample `n_samples` seqs. from total samples
                raise NotImplementedError

                n_samples = len(self.val_sids) \
                    if n_samples > len(self.val_sids) else n_samples
                l_samples = random.sample(self.val_sids, n_samples)

        # Viz. from l_samples
        else:
            if n_samples > 0:
                if len(l_samples) != n_samples:
                    print('Warning: Both `n_samples` and `l_samples` are given\
. Visualizing only first `n_samples` from `l_samples`.')
                    l_samples = l_samples[:n_samples]

        # Format all seq IDs
        l_samples = [utils.int2padstr(sid, 5) for sid in l_samples]
        return l_samples


    def _get_cl_assignment(self, sample):
        '''
        '''
        clid_matrix = np.stack(self.data['clid2kp']['keypoints3d'].to_numpy())
        K = clid_matrix.shape[0]
        diff = np.sum((clid_matrix - sample)**2, axis=(1, 2))
        clid = np.argmin(diff)
        return clid


    def _get_clids_for_seq(self, seq):
        '''Given original seq., return equivalent cluster indices.
        Args:
            seq <np.array> (T, 21, 3): xyz keypoints of 21 joints.
        Return:
            cl_ids <list> (T) of <int>. Vector containing cluster ids.
        '''
        l_clids = [self._get_cl_assignment(frame) for frame in seq]
        return l_clids


    def _compress_l_clids(self, sid, idx, l_clids):
        '''Convert list of clids -> (clid, length). Store in the format
        described in _create_seq2clid_df_gt_cl(.).
        '''
        df_dict = {'name': [], 'idx': [], 'cluster': [], 'length': []}
        for k, g in groupby(l_clids):
            df_dict['name'].append(sid)
            df_dict['idx'].append(idx)
            idx += 1
            df_dict['cluster'].append(k)
            df_dict['length'].append(len(list(g)))
        df = pd.DataFrame(data=df_dict)
        return df, idx


    def _create_seq2clid_df_gt_cl(self):
        '''
        DataFrame format:
        name    idx    cluster    length
1       00009    1      953         13
...       ...    ...    ...         ...
178     00034    16     890         8
        '''
        # Append each entry (idx) to this df
        seq2clid_df = pd.DataFrame()
        idx = 1

        # Loop over each seq.
        for sid in self.l_samples:

            # Downsample 100 fps --> 12.5 fps, restrict to 50 sec.
            seq = change_fps(self.data['gt'][sid][:5000], self.cfg.fps.gt_fps, self.cfg.fps.pred_fps)

            # Get list of cluster ids for each frame in seq.
            l_clids = self._get_clids_for_seq(seq)

            # Compress list of cl. IDs --> (cl. IDs, counts). See above.
            df, idx = self._compress_l_clids(sid, idx, l_clids)

            # Append to common dataFrome
            seq2clid_df = pd.concat([seq2clid_df, df])

        return seq2clid_df


    def _create_seq2clid_df_preds(self, l_seq_clids):
        '''
        Args:
            l_seq_clids <list>: of <np.array> of cluster ID <int> for each seq.

        Return:
            seq2clid_df <pd.DataFrame>. Format:
                name    idx    cluster    length
        1       00009    1      953         13
        ...       ...    ...    ...         ...
        178     00034    16     890         8
        '''
        # Append each entry (idx) to this df
        seq2clid_df = pd.DataFrame()
        idx = 1

        # Loop over each seq.
        for sid, seq_clids in zip(self.l_samples, l_seq_clids):

            # Compress list of cl. IDs --> (cl. IDs, counts). See above.
            df, idx = self._compress_l_clids(sid, idx, seq_clids)

            # Append to common dataFrome
            seq2clid_df = pd.concat([seq2clid_df, df])

        return seq2clid_df


    def viz_diff_rec_types(self, seq2clid_df, dir_n):
        '''
        Args:
            seq2clid_df <pd.DataFrame>: Described in _create_seq2clid_df(.).
        Frame-rate of cluster ids @ 12.5 fps.
        '''
        frames_dir = Path(self.cfg.viz_dir, dir_n)

        for rec_type in self.cfg.rec_type:
            reconstruction(rec_type, self.cfg.filters, self.l_samples,
                self.data['gt'], 'kitml', self.cfg.fps.pred_fps, self.cfg.fps.gt_fps, self.cfg.fps.out_fps,
                frames_dir, None,
                False, contiguous_frame2cluster_mapping_path=seq2clid_df,
                cluster2frame_mapping_path=self.data['clid2frame'])


    def recons_viz(self):
        '''
        '''
        samples_dir = Path(self.cfg.approaches.recons_viz)
        
        #TODO: use pickle file everywhere
        # Get predicted cluster IDs for all seqs. @ 12.5 fps
        l_seq_clids = []
        for sid in self.l_samples:
            kp = np.array(np.load(f'{samples_dir}/{sid}.npy'), dtype=np.int64)
            l_seq_clids.append(kp)

        # Collate preds into specific compressed dataFrame
        seq2clid_df = self._create_seq2clid_df_preds(l_seq_clids)

        self.viz_diff_rec_types(seq2clid_df, 'asymov_mt')


    def sample_mt_asymov(self):
        '''
        Eg., path for model predictions (npy files):
        packages/TEMOS/outputs/kit-xyz-motion-word/asymov_full_run_1/uoon5wnl/samples/neutral_0ade04bd-954f-49bd-b25f-68f3d1ab8f1a
        '''
        ckpt_p = Path(self.cfg.approaches.asymov_mt)

        cmd = ['python', 'sample_asymov_mt.py']

        # Overwrite cfg at configs/sample_asymov_mt.yaml
        cmd.append(f'folder={ckpt_p.parent.parent}')
        cmd.append(f'split={self.split_file_p.name}')
        cmd.append(f'ckpt_path={ckpt_p}')
        print(f'Run: ', ' '.join(cmd))

        # Forward pass, store predictions
        subprocess.call(cmd, cwd=str(self.cfg.path))

        # Covert clids --> frames
        clid2kp = np.array(self.data['clid2kp']['keypoints3d'])

        # Destination npy files
        npy_folder = ckpt_p.parent.parent / 'samples' / f'neutral_{self.split_file_p.name}'

        # Get predicted cluster IDs for all seqs. @ 12.5 fps
        l_seq_clids = []
        for sid in self.l_samples:
            kp = np.array(np.load(f'{npy_folder}/{sid}.npy'), dtype=np.int64)
            l_seq_clids.append(kp)

        # Collate preds into specific compressed dataFrame
        seq2clid_df = self._create_seq2clid_df_preds(l_seq_clids)

        self.viz_diff_rec_types(seq2clid_df, 'asymov_mt')


    def sample_temos_asymov(self):
        '''
        Eg., path for model predictions (npy files):
        packages/TEMOS/outputs/kit-xyz-motion-word/asymov_full_run_1/uoon5wnl/samples/neutral_0ade04bd-954f-49bd-b25f-68f3d1ab8f1a
        '''
        ckpt_p = Path(self.cfg.approaches.asymov_temos)

        cmd = ['python', 'sample_asymov.py']

        # Overwrite cfg at configs/sample_asymov.yaml
        cmd.append(f'folder={ckpt_p.parent.parent}')
        cmd.append(f'split={self.split_file_p.name}')
        cmd.append(f'ckpt_path={ckpt_p}')
        print(f'Run: ', ' '.join(cmd))

        # Forward pass, store predictions
        subprocess.call(cmd, cwd=str(self.cfg.path))

        # Covert clids --> frames
        clid2kp = np.array(self.data['clid2kp']['keypoints3d'])

        # Destination npy files
        npy_folder = ckpt_p.parent.parent / 'samples' / f'neutral_{self.split_file_p.name}'

        # Get predicted cluster IDs for all seqs. @ 12.5 fps
        l_seq_clids = []
        for sid in self.l_samples:
            kp = np.array(np.load(f'{npy_folder}/{sid}.npy'), dtype=np.int64)
            l_seq_clids.append(kp)

        # Collate preds into specific compressed dataFrame
        seq2clid_df = self._create_seq2clid_df_preds(l_seq_clids)

        self.viz_diff_rec_types(seq2clid_df, 'asymov_temos')


    def sample_temos_bl(self):
        '''Sample and visualize seqs. specified in `self.l_samples`. Use TEMOS's sampling
        and viz. code out of the box.

        For testing purposes:
            <file_p>: packages/TEMOS/datasets/kit-splits/0d6f926b-52e9-4786-a1ae-1bd7dcf8d592
        '''

        # Load "latest ckpt" from folder spec. in viz.yaml. Overwrite TEMOS's sample cfg.
        folder = Path(self.cfg.approaches.temos_bl)

        # TEMOS's sampling script saves pred xyz motions as npy's in ${folder}/samples/${split}
        sample_args = f'folder={folder} split={self.split_file_p.name}'
        os.chdir('packages/TEMOS')
        cmd = f'HYDRA_FULL_ERROR=1 python sample.py {sample_args}'
        print(f'Run: ', cmd)
        os.system(cmd)

        # Destination npy files: 
        npy_folder = folder / 'samples' / f'neutral_{self.split_file_p.name}'
        keypoints = []
        for sid in self.l_samples:
            kp = np.load(f'{npy_folder}/{sid}.npy')
            # Downsample kp. TODO: Verify assumption of 100fps output
            kp = change_fps(kp, 0.3)
            keypoints.append(kp)

        # Visualize
        viz_l_seqs(self.l_samples, keypoints, Path(self.cfg.viz_dir, 'temos_bl'), 'kitml_temos', 25.0)


    def viz_seqs(self, **kwargs):
        '''Viz. a list of seqs.

        Example:
            >>> cfg_p = '~/asymov/...'
            >>> viz_obj = Viz(cfg_p)
            >>> viz_obj.viz_seqs(n_samples=10)
            >>> viz_obj.viz_seqs(l_samples=['00002', '45', 2343, 9999])
            >>> viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=1)
            >>> viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=1)
        '''

        #TODO: redundant
        # Get a list of samples to viz.
        l_samples = kwargs['l_samples'] if 'l_samples' in kwargs.keys() else self.og_l_samples
        n_samples = kwargs['n_samples'] if 'n_samples' in kwargs.keys() else self.og_n_samples
        self.l_samples = self._get_l_samples(l_samples, n_samples)
        self.n_samples = len(self.l_samples)

        print(f'Viz. the following {self.n_samples}: {self.l_samples}.')

        # Viz. GT seqs.
        if self.cfg.approaches.gt:
            frames_dir = str(Path(self.cfg.viz_dir, 'gt'))
            ground_truth_construction(self.l_samples, self.data['gt'], 'kitml', self.cfg.fps.gt_fps, self.cfg.fps.out_fps,
                        frames_dir, force=False)

        # Reconstruct with GT Cluster ids
        if self.cfg.approaches.gt_clid:
            seq2clid_df = self._create_seq2clid_df_gt_cl()  # GT cl ids for seqs.
            self.viz_diff_rec_types(seq2clid_df, 'gt_cluster_recon')

        # Create temp file in kit-splits that sample.py can load.
        if self.l_samples == self.og_l_samples:
            self.split_file_p = self.og_split_file_p
            print(f'Using given split: {self.cfg.split}')
        else:
            self.split_file_p = Path(self.cfg.splitpath, str(uuid.uuid4()))
            utils.write_textf('\n'.join(self.l_samples), self.split_file_p)
            print('Created input seq. list file: ', self.split_file_p)

        # Reconstruct and visualize
        if self.cfg.approaches.recons_viz:
            self.recons_viz()
        
        # Inference MT-ASyMov model and reconstruct
        if self.cfg.approaches.asymov_mt:
            self.sample_mt_asymov()

        # Inference TEMOS-ASyMov model and reconstruct
        if self.cfg.approaches.asymov_temos:
            self.sample_temos_asymov()  # Get pred cl ids foj

        if self.cfg.approaches.temos_bl:
            self.sample_temos_bl()

        return 1
#-------------------------------------------------------------------------------

