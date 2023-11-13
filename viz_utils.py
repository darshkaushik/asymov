
import os, sys
import os.path as osp
from os.path import join as ospj
from os.path import basename as ospb
import pdb
from tqdm import tqdm
import pickle
from multiprocessing import cpu_count, Pool, Process
from concurrent.futures import ThreadPoolExecutor
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
import cv2
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from mpl_toolkits.mplot3d import Axes3D

from pathlib import Path
import utils

from typing import List, Union
from torch import Tensor

#TODO: change the import path to inside acton package
# sys.path.append('/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/')
# from packages.acton.src.data.dataset import loader,utils

# sys.path.append('/content/drive/Shareddrives/vid tokenization/asymov/packages/Complextext2animation/src/')

# sys.path.append('/content/drive/Shareddrives/vid tokenization/asymov/kit-molan/code/')
# import kitml_utils
# sys.path.append('/content/drive/Shareddrives/vid tokenization/asymov/packages/Complextext2animation/src/common/')


# sys.path.append('packages/TEMOS')
# from  packages.TEMOS import sample_asymov_for_viz

#traj inclusion-------------------------------------------------------------------
def add_traj(in_place_keypoints: List[np.ndarray], root_traj: List[Union[Tensor,np.ndarray]], residual: bool = True) -> List[Union[Tensor,np.ndarray]]:
    '''
    Args:
        in_place_keypoints: [Frames, 21, 3]
        root_traj: [Frames, 3]
        residual: True if root_traj is residual and requires cumulative summation, False otherwise. Defaults to True.
    '''
    if residual:
        root_traj = [residual_traj.cumsum(0) for residual_traj in root_traj]

    # pdb.set_trace()
    assert type(in_place_keypoints) is not Tensor
    if type(root_traj[0]) is Tensor:
        root_traj = [root.cpu().numpy()  for root in root_traj]
    return [keypoint + np.expand_dims(root, 1) for keypoint, root in zip(in_place_keypoints, root_traj)]

#reconstruction methods-------------------------------------------------------------------
#TODO: naive_no_rep_reconstruction implementation
def naive_no_rep_reconstruction(seq_names, contiguous_cluster_seqs, ground_truth_data, cluster2frame_mapping_path, pred_gt_fps_ratio:float=1.0, verbose=True):
    '''
    Args:
        seq_names : name of video sequences to reconstruct
        contiguous_cluster_seqs : mapping of contiguous frames with identically predicted cluster
        ground_truth_data : ground truth 3d keypoints of skeleton joints
        cluster2frame_mapping_path : Path to pickled dataframe containing the mapping of cluster to the proxy center frame (and the video sequence containing it)
        pred_gt_fps_ratio: Ratio of model-prediction fps to ground-truth fps
    Returns:
        The reconstructed keypoints
    '''
    assert len(seq_names) == len(contiguous_cluster_seqs)

    reconstructed_keypoints = []
    faulty = []

    cluster2frame = cluster2frame_mapping_path
    if type(cluster2frame_mapping_path) != pd.DataFrame:
        cluster2frame = pd.read_pickle(cluster2frame_mapping_path)

    with tqdm(zip(seq_names, contiguous_cluster_seqs), desc='naive_no_rep reconstruction', total=len(seq_names), disable=(not verbose)) as pbar:
        for name, contiguous_cluster_seq in pbar:
            pbar.set_description(f'naive_no_rep reconstruction - {name}')
            reconstructed_keypoint = []

            # try:
            for i in range(contiguous_cluster_seq.shape[0]):
                #get contiguous cluster info
                contiguous_cluster = contiguous_cluster_seq.iloc[i]
                cluster, length = contiguous_cluster[['cluster', 'length']]
                #get center frame info
                center_frame = cluster2frame.iloc[cluster]
                center_frame_idx, center_frame_keypoint, center_frame_seq_name = center_frame[['frame_index','keypoints3d','seq_name']]
                center_frame_complete_seq = ground_truth_data[center_frame_seq_name]
                assert np.array_equal(center_frame_keypoint,center_frame_complete_seq[center_frame_idx]), "Incorrect center frame sequence"
                if pred_gt_fps_ratio != 1.0:
                    center_frame_idx  = int(center_frame_idx*pred_gt_fps_ratio)
                    center_frame_complete_seq = change_fps(center_frame_complete_seq, pred_gt_fps_ratio)

                center_frame_complete_seq_len = center_frame_complete_seq.shape[0]
                if length >= center_frame_complete_seq_len:
                    # raise Exception(f'seq name : {name}\ncontiguous_cluster_seq : {i}\n')
                    lb = 0
                else:
                    lb = max(0,center_frame_idx - (length-1)//2) #check left boundary
                    lb = min(center_frame_complete_seq_len-length, lb) #check right boundary
                assert lb>=0, f"{name} - Negative left boundary"
                assert lb<center_frame_complete_seq_len, f"{name} - Exceeding right boundary"

                #reconstruct
                reconstructed_keypoint.append(center_frame_complete_seq[lb:lb+min(length, center_frame_complete_seq_len)])
                assert reconstructed_keypoint[-1].shape[0]==length, f"{name} - {reconstructed_keypoint[-1].shape[0]-length} extra frames i reconstruction"

            reconstructed_keypoints.append(np.concatenate(reconstructed_keypoint, axis=0))
            # assert reconstructed_keypoints[-1].shape[0]==ground_truth_keypoints[-1].shape[0]
            # except :
            #     ground_truth_keypoints.pop()
            #     faulty.append(name)
            #     # print(f'{name} cannot be reconstructed naively (no rep)')

    return reconstructed_keypoints, faulty

def naive_reconstruction(seq_names, contiguous_cluster_seqs, ground_truth_data, cluster2frame_mapping_path, pred_gt_fps_ratio:float=1.0, verbose=True):
    '''
    Args:
        seq_names : name of video sequences to reconstruct
        contiguous_cluster_seqs : mapping of contiguous frames with identically predicted cluster
        ground_truth_data : ground truth 3d keypoints of skeleton joints
        cluster2frame_mapping_path : Path to pickled dataframe containing the mapping of cluster to the proxy center frame (and the video sequence containing it)
        pred_gt_fps_ratio: Ratio of model-prediction fps to ground-truth fps

    Returns:
        The reconstructed keypoints
    '''
    assert len(seq_names) == len(contiguous_cluster_seqs)
    reconstructed_keypoints = []

    cluster2frame = cluster2frame_mapping_path
    if type(cluster2frame_mapping_path) != pd.DataFrame:
        cluster2frame = pd.read_pickle(cluster2frame_mapping_path)

    with tqdm(zip(seq_names, contiguous_cluster_seqs), desc='naive reconstruction', total=len(seq_names), disable=(not verbose)) as pbar:
        for name, contiguous_cluster_seq in pbar:
            pbar.set_description(f'naive reconstruction - {name}')
            reconstructed_keypoint = []

            for i in range(contiguous_cluster_seq.shape[0]):
                #get contiguous cluster info
                contiguous_cluster = contiguous_cluster_seq.iloc[i]
                cluster, length = contiguous_cluster[['cluster', 'length']]
                #get center frame info
                center_frame = cluster2frame.iloc[cluster]
                center_frame_idx, center_frame_keypoint, center_frame_seq_name = center_frame[['frame_index','keypoints3d','seq_name']]
                center_frame_complete_seq = ground_truth_data[center_frame_seq_name]
                assert np.array_equal(center_frame_keypoint,center_frame_complete_seq[center_frame_idx])
                if pred_gt_fps_ratio != 1.0:
                    center_frame_idx  = int(center_frame_idx*pred_gt_fps_ratio)
                    center_frame_complete_seq = change_fps(center_frame_complete_seq, pred_gt_fps_ratio)

                #calculate left right and center frames
                side_length = (length-1)//2
                l_frames = min(side_length, center_frame_idx)
                r_frames = min(side_length, center_frame_complete_seq.shape[0]-1 - center_frame_idx)
                center_reps = length - l_frames - r_frames

                #reconstruct
                reconstructed_keypoint.append(np.concatenate((
                    center_frame_complete_seq[center_frame_idx-l_frames:center_frame_idx], #left supporting frames
                    np.repeat(np.expand_dims(center_frame_keypoint, axis=0), center_reps, axis=0), #center frames
                    center_frame_complete_seq[center_frame_idx+1:center_frame_idx+r_frames+1]), # right supporting frames
                    axis=0
                ))

            reconstructed_keypoints.append(np.concatenate(reconstructed_keypoint, axis=0))

    return reconstructed_keypoints

def very_naive_reconstruction(seq_names, cluster_seqs, cluster2keypoint_mapping_path, verbose=True):
    '''
    Args:
        seq_names : name of video sequences to reconstruct
        cluster_seqs : the mapping of each frame to a cluster.
        cluster2keypoint_mapping_path : Path to pickled dataframe containing the mapping of cluster to proxy center keypoints

    Returns:
        The reconstructed keypoints
    '''
    # pdb.set_trace()
    assert len(seq_names) == len(cluster_seqs)

    cluster2keypoint = pd.read_pickle(cluster2keypoint_mapping_path)
    reconstructed_keypoints = []
    with tqdm(zip(seq_names, cluster_seqs), desc='very_naive reconstruction', total=len(seq_names), disable=(not verbose)) as pbar:
        for name, cluster_seq in pbar:
            pbar.set_description(f'very_naive reconstruction - {name}')
            reconstructed_keypoints.append(np.array([cluster2keypoint.loc[i,'keypoints3d'] for i in cluster_seq]))

    return reconstructed_keypoints
#-------------------------------------------------------------------------------

#error_metric-------------------------------------------------------------------
def mpjpe3d(seq_names, pred_keypoints, target_keypoints):
    '''
    seq_names List(str): names of sequences to calculate mpjpe for
    pred_keypoints List([T, num_joints, 3]) : 3d keypoints of predicted skeleton joints for each sequence
    target_keypoints List([T, num_joints, 3]) : 3d keypoints of ground-truth skeleton joints for each sequence
    '''
    # pdb.set_trace()
    mpjpe_per_sequence=[]
    with tqdm(zip(seq_names, pred_keypoints, target_keypoints), desc='calculating mpjpe', total=len(seq_names), position=0, leave=False) as pbar:
        for name, pred_keypoint, target_keypoint in pbar:
            pbar.set_description(desc=f'calculating mpjpe for {name}')

            if pred_keypoint.shape[0]!=target_keypoint.shape[0]:
                # if abs(pred_keypoint.shape[0]-target_keypoint.shape[0]) > 100:
                #     print(f'Reconstruction of {name} had {pred_keypoint.shape[0]-target_keypoint.shape[0]} more frames')
                assert pred_keypoint.shape[0]!=0

            mn = min(pred_keypoint.shape[0], target_keypoint.shape[0])
            mpjpe_per_sequence.append(np.mean(np.sqrt(np.sum((target_keypoint[:mn] - pred_keypoint[:mn]) ** 2, axis=2))))

    return mpjpe_per_sequence
#-------------------------------------------------------------------------------

#codec---------------------------------------------------------------------
#TODO: add option to control dest vid names in vid dir
#vid names will remain the same saved in a different dir
def vid_dir_to_codec_h264(src_vid_dir, dst_vid_dir=None, split_path=None, overwrite:bool=False, keep_src_vid:bool=True, faulty_path=None):
    '''
    Will operate on all files with the extension '.mp4'
    Vid names will remain the same saved in a different dir
    
    No overwrite check for faulty path
    '''
    assert osp.isdir(src_vid_dir), "src dir not found"
    assert src_vid_dir!=dst_vid_dir, "src and dst dir can't have same path"
    if osp.isdir(dst_vid_dir): 
        assert not overwrite, "set overwrite to True or provide different dst_vid_dir"
    
    #TODO: change the dir structure to contain .mp4 files directly
    if split_path is None:
        # vids = [i for i in os.listdir(src_vid_dir) if i.split('.')[-1]=='mp4']
        vids = os.listdir(src_vid_dir)
    else:
        assert osp.isfile(split_path), "split path not found"
        with open(split_path, 'r') as f:
            vids = list(map(str.strip, f.readlines()))
    pdb.set_trace()
    
    faulty = []
    with tqdm(vids, desc="changing codec to h.264") as pbar:
        for vid in pbar:
            pbar.set_description(f"changing codec of {vid} to h.264")
            try:
                # vid_to_codec_h264(ospj(src_vid_dir, vid), ospj(dst_vid_dir, vid), overwrite, keep_src_vid)
                vid_to_codec_h264(ospj(src_vid_dir, vid, f'{vid}.mp4'), ospj(dst_vid_dir, vid, f'{vid}.mp4'), overwrite, keep_src_vid)
            except:
                faulty.append(vid)
    
    if faulty_path is not None:
        open(faulty_path, 'w').writelines(faulty)
    print(faulty)
    return faulty

def vid_to_codec_h264(src_vid_path, dst_vid_path, overwrite:bool=False, keep_src_vid:bool=True):
    '''
    Cannot have same src and dst path
    '''
    assert osp.isfile(src_vid_path), "src vid not found"
    assert src_vid_path!=dst_vid_path, "src and dst vid can't have same path"
    assert osp.isfile(dst_vid_path) and not overwrite, "set overwrite to True or provide different dst_vid_path"

    pdb.set_trace()
    cmd = ['ffmpeg', '-i', src_vid_path, '-y' if overwrite else '-n', '-c:v', 'libx264', dst_vid_path]
        
    FNULL = open(os.devnull, 'w')
    retcode = subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    assert retcode != 0, 'Error {0} executing command: {1}'.format(retcode, ' '.join(cmd))

    if not keep_src_vid:
        shutil.rmtree(src_vid_path)
#-------------------------------------------------------------------------------

#fps---------------------------------------------------------------------
def change_fps(motion, ratio_or_in_fps: float, out_fps: float=None):
    if out_fps is None: # ratio given directly
        ratio = ratio_or_in_fps
    else:
        ratio = out_fps/ratio_or_in_fps

    if ratio>1.0:
        return upsample(motion, ratio)
    elif ratio<1.0:
        return downsample(motion, ratio)
    else:
        return motion 

#TODO: change nomenclature, add upsampling without interpolation
def upsample(motion, ratio):
    assert ratio >= 1.0
    motion = np.array(motion)

    if ratio == 1.0:
        return motion
    step = int(ratio)
    assert step >= 1

    # Alpha blending => interpolation
    alpha = np.linspace(0, 1, step+1)
    last = np.einsum("l,...->l...", 1-alpha, motion[:-1])
    new = np.einsum("l,...->l...", alpha, motion[1:])

    chuncks = (last + new)[:-1]
    output = np.concatenate(chuncks.swapaxes(1, 0))
    # Don't forget the last one
    output = np.concatenate((output, motion[[-1]]))
    return output

def downsample(motion, ratio):
    assert ratio <= 1.0
    motion = np.array(motion)
    num_frames = len(motion)
    if ratio == 1.0:
        return motion

    step = int(1 / ratio)
    assert step >= 1
    subsampled_frames = np.arange(0, num_frames, step, dtype='int32')
    # Don't forget the last one
    output = motion[subsampled_frames]
    return output
#-------------------------------------------------------------------------------


#joint_names--------------------------------------------------------------------

def get_smpl_joint_names():
    # Joint names from SMPL Wiki
    joint_names_map = {
        0: 'Pelvis',

        1: 'L_Hip',
        4: 'L_Knee',
        7: 'L_Ankle',
        10: 'L_Foot',

        2: 'R_Hip',
        5: 'R_Knee',
        8: 'R_Ankle',
        11: 'R_Foot',

        3: 'Spine1',
        6: 'Spine2',
        9: 'Spine3',
        12: 'Neck',
        15: 'Head',

        13: 'L_Collar',
        16: 'L_Shoulder',
        18: 'L_Elbow',
        20: 'L_Wrist',
        22: 'L_Hand',
        14: 'R_Collar',
        17: 'R_Shoulder',
        19: 'R_Elbow',
        21: 'R_Wrist',
        23: 'R_Hand'}

    # Return all joints except indices 22 (L_Hand), 23 (R_Hand)
    return [joint_names_map[idx] for idx in range(len(joint_names_map)-2)]

def get_kitml_joint_names():
    return [
            'root',     # 0
            'BP',       # 1
            'BT',       # 2
            'BLN',      # 3
            'BUN',      # 4
            'LS',       # 5
            'LE',       # 6
            'LW',       # 7
            'RS',       # 8
            'RE',       # 9
            'RW',       # 10
            'LH',       # 11
            'LK',       # 12
            'LA',       # 13
            'LMrot',    # 14
            'LF',       # 15
            'RH',       # 16
            'RK',       # 17
            'RA',       # 18
            'RMrot',    # 19
            'RF']       # 20

def get_nturgbd_joint_names():
    '''From paper:
    1-base of the spine 2-middle of the spine 3-neck 4-head 5-left shoulder 6-left elbow 7-left wrist 8- left hand 9-right shoulder 10-right elbow 11-right wrist 12- right hand 13-left hip 14-left knee 15-left ankle 16-left foot 17- right hip 18-right knee 19-right ankle 20-right foot 21-spine 22- tip of the left hand 23-left thumb 24-tip of the right hand 25- right thumb
    '''
    # Joint names by AC, based on SMPL names
    joint_names_map = {
        0: 'Pelvis',

        12: 'L_Hip',
        13: 'L_Knee',
        14: 'L_Ankle',
        15: 'L_Foot',

        16: 'R_Hip',
        17: 'R_Knee',
        18: 'R_Ankle',
        19: 'R_Foot',

        1: 'Spine1',
        # 'Spine2',
        20: 'Spine3',
        2: 'Neck',
        3: 'Head',

        # 'L_Collar',
        4: 'L_Shoulder',
        5: 'L_Elbow',
        6: 'L_Wrist',
        7: 'L_Hand',
        21: 'L_HandTip',  # Not in SMPL
        22: 'L_Thumb',  # Not in SMPL

        # 'R_Collar',
        8: 'R_Shoulder',
        9: 'R_Elbow',
        10: 'R_Wrist',
        11: 'R_Hand',
        23: 'R_HandTip',  # Not in SMPL
        24: 'R_Thumb',  # Not in SMPL
    }

    return [joint_names_map[idx] for idx in range(len(joint_names_map))]

def get_coco17_joint_names():
    '''
    From loader.py in acton package
    '''
    return [
        'nose', #0
        'left_eye', 'right_eye', #1,2
        'left_ear', 'right_ear', #3,4
        'left_shoulder', 'right_shoulder', #5,6
        'left_elbow', 'right_elbow', #7,8
        'left_wrist', 'right_wrist', #9,10
        'left_hip', 'right_hip', #11,12
        'left_knee', 'right_knee', #13,14
        'left_ankle', 'right_ankle' #15,16
    ]

#-------------------------------------------------------------------------------

#joint_connectivity-------------------------------------------------------------

def get_smpl_skeleton():
    ''' AC -- change the skeleton ordering so that you traverse
    the joints in the following order: Left lower, Left upper,
    Spine, Neck, Head, Right lower, Right upper.
    '''
    return np.array(
        [
            # Left lower
            [ 0, 1 ],
            [ 1, 4 ],
            [ 4, 7 ],
            [ 7, 10],

            # Left upper
            [ 9, 13],
            [13, 16],
            [16, 18],
            [18, 20],
            # [20, 22],

            # Spinal column
            [ 0, 3 ],
            [ 3, 6 ],
            [ 6, 9 ],
            [ 9, 12],
            [12, 15],

            # Right lower
            [ 0, 2 ],
            [ 2, 5 ],
            [ 5, 8 ],
            [ 8, 11],

            # Right upper
            [ 9, 14],
            [14, 17],
            [17, 19],
            [19, 21],
            # [21, 23],
        ])

def get_kitml_skeleton():
    return np.array([
        # Spine
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],

        # Left upper
        [3, 5],
        [5, 6],
        [6, 7],

        # Left lower
        [0, 11],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],

        # Right upper
        [3, 8],
        [8, 9],
        [9, 10],

        # Right lower
        [0, 16],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20]
    ])

def get_nturgbd_skeleton():
    ''' AC -- change the skeleton ordering so that you traverse
    the joints in the following order: Left lower, Left upper,
    Spine, Neck, Head, Right lower, Right upper.
    '''
    return np.array(
        [
            # Left lower
            [0, 12],
            [12, 13],
            [13, 14],
            [14, 15],

            # Left upper
            [4, 20],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 21],
            [7, 22],  # --> L Thumb

            # Spinal column
            [0, 1],
            [1, 20],
            [20, 2],
            [2, 3],

            # Right lower
            [0, 16],
            [16, 17],
            [17, 18],
            [18, 19],

            # Right upper
            [20, 8],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 24],
            # [24, 11] --> R Thumb

            [21, 22],

            [23, 24],

        ]
    )

def get_coco17_skeleton():
    '''
    From visualizer.py in acton package
    '''
    return np.array([
        #Left lower
        [11, 13], [13, 15],

        #Left upper
        [5, 7], [7, 9],

        #Head
        [0, 1], [0, 2],
        [1, 3], [2, 4],

        #Spine
        [5, 11], [6, 12],
        [5, 6], [11, 12], #bridging left and right

        #Right lower
        [12, 14], [14, 16],

        #Right upper
        [6, 8], [8, 10]
    ])

#-------------------------------------------------------------------------------


def get_joint_colors(joint_names):
    '''Return joints based on a color spectrum. Also, joints on
    L and R should have distinctly different colors.
    '''
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(joint_names))]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    return colors


def calc_angle_from_x(sk):
    '''Given skeleton, calc. angle from x-axis'''
    # Hip bone
    id_l_hip = get_smpl_joint_names().index('L_Hip')
    id_r_hip = get_smpl_joint_names().index('R_Hip')
    pl, pr = sk[id_l_hip], sk[id_r_hip]
    bone = np.array(pr-pl)
    unit_v =  bone / np.linalg.norm(bone)
    # Angle with x-axis
    pdb.set_trace()
    x_ax = np.array([1, 0, 0])
    x_angle = math.degrees(np.arccos(np.dot(x_ax, unit_v)))

    '''
    l_hip_z = seq[0, joint_names.index('L_Hip'), 2]
    r_hip_z = seq[0, joint_names.index('R_Hip'), 2]
    az = 0 if (l_hip_z > zroot and zroot > r_hip_z) else 180
    '''
    if bone[1] > 0:
        x_angle = - x_angle

    return x_angle


def calc_angle_from_y(sk):
    '''Given skeleton, calc. angle from x-axis'''
    # Hip bone
    id_l_hip = get_smpl_joint_names().index('L_Hip')
    id_r_hip = get_smpl_joint_names().index('R_Hip')
    pl, pr = sk[id_l_hip], sk[id_r_hip]
    bone = np.array(pl-pr)
    unit_v =  bone / np.linalg.norm(bone)
    print(unit_v)
    # Angle with x-axis
    pdb.set_trace()
    y_ax = np.array([0, 1, 0])
    y_angle = math.degrees(np.arccos(np.dot(y_ax, unit_v)))

    '''
    l_hip_z = seq[0, joint_names.index('L_Hip'), 2]
    r_hip_z = seq[0, joint_names.index('R_Hip'), 2]
    az = 0 if (l_hip_z > zroot and zroot > r_hip_z) else 180
    '''
    # if bone[1] > 0:
    #    y_angle = - y_angle
    seq_y_proj = bone * np.cos(np.deg2rad(y_angle))
    print('Bone projected onto y-axis: ', seq_y_proj)

    return y_angle


def viz_skeleton(seq, folder_p, sk_type='smpl', radius=1, lcolor='#ff0000', rcolor='#0000ff', action='', inplace=False, debug=False):
    ''' Visualize skeletons for given sequence and store as images.

    Args:
        seq (np.array): Array (frames) of joint positions.
        Size depends on sk_type (see below).
            if sk_type is 'smpl' then assume:
                1. first 3 dims = translation.
                2. Size = (# frames, 69)
            elif sk_type is 'nturgbd', then assume:
                1. no translation.
                2. Size = (# frames, 25, 3)
            elif sk_type == 'kitml_temos'
        folder_p (str): Path to root folder containing visualized frames.
            Frames are dumped to the path: folder_p/frames/*.jpg
        radius (float): Space around the subject?
        inplace (bool): whether to remove translation to visualize in-place 
            motion where root is always at origin
    Returns:
        Stores skeleton sequence as jpg frames.
    '''
    joint_names = None
    az = 90
    if sk_type=='nturgbd':
        joint_names = get_nturgbd_joint_names()
        kin_chain = get_nturgbd_skeleton()
    elif sk_type=='smpl':
        joint_names = get_smpl_joint_names()
        kin_chain = get_smpl_skeleton()  # NOTE that hands are skipped.
        # Reshape flat pose features into (frames, joints, (x,y,z)) (skip trans)
        seq = seq[:, 3:].reshape(-1, len(joint_names), 3).cpu().detach().numpy()
    elif 'kit' in sk_type:
        joint_names = get_kitml_joint_names()
        kin_chain = get_kitml_skeleton()
        # seq[..., 1] = -seq[..., 1]
        seq = seq[..., [2,0,1]]
        az = 60
        # radius = 1.2
    elif sk_type=='coco17':
        joint_names = get_coco17_joint_names()
        kin_chain = get_coco17_skeleton()
    else:
        assert NotImplementedError

    n_j = len(joint_names)

    # Get color-spectrum for skeleton
    colors = get_joint_colors(joint_names)
    labels = [(joint_names[jidx[0]], joint_names[jidx[1]]) for jidx in kin_chain]

    xroot, yroot, zroot = 0.0, 0.0, 0.0
    if sk_type=='coco17':
        xroot, yroot, zroot = 0.5*(seq[0,11] + seq[0,12])
        seq=seq-np.array([[[xroot, yroot, zroot]]])
        seq=seq/np.max(np.abs(seq))
    elif 'kit' in sk_type:
        # !inital translation. Subtract all frames by frame 0's pelvis.
        seq -= seq[0, 0]
        # !global translation. Subtract all frames by corresponding pelvis.
        inplace_seq = seq - seq[:,0:1,:]

        # x, y, z --> [-1, 1]
        # seq /= np.max(np.abs(seq), axis=(0,1))  # Normalize each dim. separately.
        seq /= np.max(np.abs(inplace_seq))  # Normalize all dim. uniformly. Not considering trajectory while normalizing

        if inplace:
            seq = inplace_seq
    else:
        xroot, yroot, zroot = seq[0, 0, 0], seq[0, 0, 1], seq[0, 0, 2]

    # seq = seq - seq[0, :, :]

    # Change viewing angle so that first frame is in frontal pose
    # az = calc_angle_from_x(seq[0]-np.array([xroot, yroot, zroot]))
    # az = calc_angle_from_y(seq[0]-np.array([xroot, yroot, zroot]))

    # Create folder for frames
    if not os.path.exists(osp.join(folder_p, 'frames')):
        os.makedirs(osp.join(folder_p, 'frames'))

    # Viz. skeleton for each frame
    # for t in tqdm(range(seq.shape[0]), desc='frames', position=1, leave=False):
    for t in range(seq.shape[0]):
        # Fig. settings
        fig = plt.figure(figsize=(7, 6)) if debug else \
              plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

        # No moving camera, no root update required as `inplace` argument takes care of it if needed
        # if 'kit' in sk_type:
        #     xroot, yroot, zroot = seq[t,0]
        # else:
        #     pass  # Do not update root.  # TODO: Verify this.
        # seq[t] = seq[t] - [xroot, yroot, zroot]

        # More figure settings
        ax.set_title(action)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # pdb.set_trace()
        ax.set_xlim3d(-radius + xroot, radius + xroot)
        ax.set_ylim3d(-radius + yroot, radius + yroot)
        ax.set_zlim3d(-radius + zroot, radius + zroot)

        if True==debug:
            ax.axis('on')
            ax.grid(b=True)
        else:
            ax.axis('off')
            ax.grid(b=None)
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_zticklabels([])

        # ax.view_init(75, az)
        # ax.view_init(elev=20, azim=90+az)

        if sk_type == 'coco17':
            ax.view_init(elev=-90, azim=90)
        else:
            ax.view_init(elev=-20, azim=az)

        if True==debug:
            ax.legend(bbox_to_anchor=(1.1, 1), loc='upper right')
            pass

        for i, (j1, j2) in enumerate(kin_chain):
            # Store bones
            x = np.array([seq[t, j1, 0], seq[t, j2, 0]])#-xroot
            y = np.array([seq[t, j1, 1], seq[t, j2, 1]])#-yroot
            z = np.array([seq[t, j1, 2], seq[t, j2, 2]])#-zroot
            # Plot bones in skeleton
            ax.plot(x, y, z, c=colors[i], marker='o', linewidth=2, label=labels[i])
            pass

        # pdb.set_trace()
        cv2.waitKey(0)

        # fig.savefig(osp.join(folder_p, 'frames', '{1}_{0}.jpg'.format(t, ii)))
        fig.savefig(osp.join(folder_p, 'frames', '{:05d}.jpg'.format(t)))
        plt.close(fig)

def viz_skeleton_mp(seq_folder_p, sk_type, radius): # for multi-processing, can pass only 1 arg
    return viz_skeleton(*seq_folder_p, sk_type, radius)

def write_vid_from_imgs(folder_p, fps):
    '''Collate frames into a video sequence.

    Args:
        folder_p (str): Frame images are in the path: folder_p/frames/<int>.jpg
        fps (float): Output frame rate.

    Returns:
        Output video is stored in the path: folder_p/{seq_id}.mp4
    '''
    sid = ospb(folder_p)
    vid_p = osp.join(folder_p, f'{sid}.mp4')
    cmd = ['ffmpeg', '-r', str(int(fps)), '-i',
                    osp.join(folder_p, 'frames', '%05d.jpg'), '-y', '-c:v', 'libx264', vid_p]
    FNULL = open(os.devnull, 'w')
    retcode = subprocess.call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)
    if not 0 == retcode:
        print('*******ValueError(Error {0} executing command: {1}*********'.format(retcode, ' '.join(cmd)))
    shutil.rmtree(osp.join(folder_p, 'frames'))

def joint2vid(name_keypoint, sk_type, frames_dir, fps, radius=1.2): #combine viz_skeleton
    name, keypoint = name_keypoint
    folder_p = ospj(frames_dir, name)
    viz_skeleton(keypoint, folder_p, sk_type, radius)
    write_vid_from_imgs(folder_p, fps)
    return None


def viz_l_seqs(seq_names, keypoints, frames_dir, sk_type, fps, force=False):
    '''1. Dumps sequence of skeleton images for the given sequence of joints.
    2. Collates the sequence of images into an mp4 video.

    Args:
        seq_names (List(str)): name of sequences to visualize
        # TODO: Change this
        keypoints (np.array([T, num_joints, 3])): Joint positions for each frame
        frames_dir (str): Path to dir where visualizations will be dumped.
        sk_type (str): {'smpl', 'nturgbd','kitml','coco17'}
        fps (float): Output frame rate.
        force (Bool): If True visualizes all sequences even if already exists in frames_dir.
                    Defaults to False, only visualizes incomplete or un-visualized sequences.

    Return:
        None. Path of mp4 video: folder_p/{seq_name}.mp4
    '''
    # Delete folder if exists
    if force:
        _ = [shutil.rmtree(ospj(frames_dir, name)) for name in seq_names if osp.exists(ospj(frames_dir, name))]
        viz_names = seq_names
    else:
        viz_names=[]
        for name in seq_names:
            folder_p = ospj(frames_dir, name)
            if osp.exists(folder_p):
                # Rename folder
                if osp.exists(ospj(folder_p, f'{name}.mp4')):
                    # print(f'Video for {folder_p[-5:]} already exists')
                    continue
                else:
                    # print('Deleting existing folder ', folder_p)
                    shutil.rmtree(folder_p)
            viz_names.append(name)
    print(f'Sequences to visualize: {len(viz_names)}')

    # pdb.set_trace()
    # joint2img_procs = []
    # img2vid_procs = []
    # with tqdm(zip(viz_names, keypoints), desc='joint2img', total=len(viz_names), position=0) as pbar:
    #     for name, keypoint in pbar:
    #         pbar.set_description(desc=f'joint2img {name}')
    #         folder_p = ospj(frames_dir, name)
    #         viz_skeleton(keypoint, folder_p, sk_type, 1.2)
    #       # joint2img_proc = Process(target=viz_skeleton, args=(keypoint, folder_p, sk_type, 1.2))
    #       # joint2img_proc.start()
    #       # joint2img_procs.append(joint2img_proc)

    # with tqdm(enumerate(viz_names), desc='img2vid', total=len(viz_names), position=0) as pbar:
    #     for i, name in pbar:
    #         pbar.set_description(desc=f'img2vid {name}')
    #         folder_p = ospj(frames_dir, name)
    #         write_vid_from_imgs(folder_p, fps)
    #         # joint2img_procs[i].join()
    #         # img2vid_proc = Process(target=write_vid_from_imgs, args=(folder_p, fps))
    #         # img2vid_proc.start()
    #         # img2vid_procs.append(img2vid_proc)
    # _ = [p.join() for p in tqdm(img2vid_procs, 'generating vids')]

    # #option 1: new process for each seq_name (with Pool, concurrent operations)
    n = len(viz_names)

    with Pool(cpu_count()*2) as p:
        chunk_size = 2
        iter = zip(viz_names, keypoints)
        with tqdm(desc='joint2vid', total=n) as pbar:
            _ = [pbar.update() for _ in p.imap_unordered(partial(joint2vid, sk_type=sk_type, frames_dir=frames_dir, fps=fps), iter, chunk_size)]

    # option 2: new process for each seq_name for each operation (with Pool, sequential operations)
    # n = len(viz_names)
    # folders_p = [ospj(frames_dir, name) for name in viz_names]
    # with Pool(cpu_count()*2) as p:
    #     chunk_size = 2
    #     iter = zip(keypoints, folders_p)
    #     with tqdm(desc='joint2img', total=n) as pbar:
    #         _ = [pbar.update() for _ in p.imap_unordered(partial(viz_skeleton_mp, sk_type=sk_type, radius=1.2), iter, chunk_size)]

    #     with tqdm(desc='img2vid', total=n) as pbar:
    #         _ = [pbar.update() for _ in p.imap_unordered(partial(write_vid_from_imgs, fps=fps), folders_p, chunk_size)]

    return None


# def fill_xml_jpos(all_j_names, mmm_seq):
#     '''Given joint positions with missing values for some axes, return a filled
#     numpy array of joint positions.'''
#     # T, Root + 20 joints => 21, xyz
#     seq = np.zeros((len(mmm_seq), 21, 3))

#     # Get joints in the expected order
#     plt_jorder = get_kitml_joint_names()
#     c = 0

#     # Loop over each joint name for plotting
#     for j_i, plt_j in enumerate(plt_jorder):
#         # Find the relevant joint axis name and store its position
#         for ax_i, ax in enumerate(['x', 'y', 'z']):
#             if plt_j=='RM' or plt_j=='LM':
#                 jax_name = plt_j + ax + '_rot'
#             else:
#                 jax_name = plt_j + ax + '_joint'
#             if jax_name in all_j_names:
#                 i = all_j_names.index(jax_name)
#                 seq[:, j_i, ax_i] = np.array(mmm_seq)[:, i]
#                 c += 1
#     print(f'Found {c} indices.')
#     return seq

# def viz_rand_seq(X, Y, dtype, epoch, wb, urls=None,
#                  k=3, pred_labels=None):
#     '''
#     Args:
#         X (np.array): Array (frames) of SMPL joint positions.
#         Y (np.array): Multiple labels for each frame in x \in X.
#         dtype (str): {'input', 'pred'}
#         k (int): # samples to viz.
#         urls (tuple): Tuple of URLs of the rendered videos from original mocap.
#         wb (dict): Wandb log dict.
#     Returns:
#         viz_ds (dict): Data structure containing all viz. info so far.
#     '''
#     import packages.Complextext2animation.src.dataUtils as dutils

#     # `idx2al`: idx --> action label string
#     al2idx = dutils.read_json('data/action_label_to_idx.json')
#     idx2al = {al2idx[k]: k for k in al2idx}

#     # Sample k random seqs. to viz.
#     for s_idx in random.sample(list(range(X.shape[0])), k):
#         # Visualize a single seq. in path `folder_p`
#         folder_p = osp.join('viz', str(uuid.uuid4()))
#         viz_seq(seq=X[s_idx], folder_p=folder_p)
#         title='{0} seq. {1}: '.format(dtype, s_idx)
#         pdb.set_trace()
#         acts_str = ', '.join([idx2al[l] for l in torch.unique(Y[s_idx])])
#         wb[title+urls[s_idx]] = wandb.Video(osp.join(folder_p, 'video.mp4'),
#                                            caption='Actions: '+acts_str)

#         if 'pred' == dtype or 'preds'==dtype:
#             raise NotImplementedError☺

#     print('Done viz. {0} seqs.'.format(k))
#     return wb


# def debug_viz_kitml_seq():
#     '''
#     '''
#     import packages.Complextext2animation.src.data as d
#     import packages.Complextext2animation.src.common.mmm as mmm


#     kitml_fol_p = '/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/dataset/kit-mocap/'

#     # List of seq ids to viz.
#     l_sids = list(range(1, 10))
#     for sid in l_sids:
#         fp = Path(ospj(kitml_fol_p, '{0}_mmm.xml'.format(str(sid).zfill(5))))
#         # seq = [ <list> joint names, <list> joint positions] of length (44, 44*T)
#         # all_j_names, mmm_seq = kitml_utils.parse_motions(fp)[0]
#         # all_j_names, mmm_dict = mmm.parse_motions(fp)[0]
#         # seq = fill_xml_jpos(all_j_names, mmm_seq)
#         # jnames, root_pos, root_rot, values, joint_dict = mmm.mmm2csv(fp)

#         # Get joint position data: (T, J=21, 3)
#         K = d.KITMocap(path2data=kitml_fol_p)
#         xyz_data, skel_obj, joints, root_pos, root_rot = K.mmm2quat(fp)

#         # Normalize by root position in case it's not already:
#         seq = xyz_data - np.transpose(root_pos.numpy(), axes=(1,0,2)) # seq = (T, J, 3)

#         # (J, T, 3) --> (T, J, 3)
#         # seq = values.transpose(1, 0, 2)

#         # TODO: Assign root position value. For now, copy over Pelvis values.
#         # jnames = ['root'] + jnames
#         # seq = np.concatenate((seq[:, 0:1, :], seq), axis=1)
#         # Assume root position = (0, 0, 0)
#         # T, J, _ = seq.shape
#         # seq = np.concatenate((np.zeros((T, 1, 3)), seq), axis=1)

#         # Make sure that joints are in the expected order
#         # plt_jorder = get_kitml_joint_names()
#         # new_idxs = []
#         # for j in plt_jorder:
#         #   new_i = jnames.index(j)
#         #   new_idxs.append(new_i)
#         # seq = seq[:, np.array(new_idxs), :]

#         # Viz. "values" (joint positions?)
#         viz_seq(seq, './test_viz/{}_orig_format'.format(sid), 'kitml', debug=True)

#specific_skeleton_viz----------------------------------------------------------

# def viz_kitml_seq():
#     '''
#     '''
#     import packages.Complextext2animation.src.data as d
#     kitml_fol_p = '/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/dataset/kit-mocap/'

#     # List of seq ids to viz.
#     l_sids = list(range(1, 10))
#     for sid in l_sids:
#         fp = Path(ospj(kitml_fol_p, '{0}_mmm.xml'.format(str(sid).zfill(5))))

#         # Get joint position data: (T, J=21, 3)
#         K = d.KITMocap(path2data=kitml_fol_p)
#         xyz_data, skel_obj, joints, root_pos, root_rot = K.mmm2quat(fp)

#         # Normalize by root position in case it's not already.
#         # seq = (T, J, 3)
#         seq = xyz_data - np.transpose(root_pos.numpy(), axes=(1,0,2))

#         # Viz. "values" (joint positions?)
#         viz_seq(seq, './test_viz/{}'.format(sid), 'kitml', debug=True)

# def viz_aistpp_seq():
#     '''
#     '''
#     d = loader.AISTDataset('/content/drive/Shareddrives/vid tokenization/aistpp_subset/aistplusplus/annotations')
#     seq=d.load_keypoint3d('gWA_sFM_cAll_d27_mWA2_ch21')

#     frames_dir='/content/drive/Shareddrives/vid tokenization/frames2'
#     viz_seq(seq, frames_dir, 'coco17', debug=False)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    src = 'F:/space over - dumped trash/vid tokenization/asymov_mt/naive_no_rep_uniform'
    split = 'packages/TEMOS/datasets/kit-splits-tiny/recons'
    vid_dir_to_codec_h264(src, src+'_', split, True)
    
    # # from packages.acton.src.data.dataset.loader import KITDataset

    # # viz_kitml_seq()
    # # viz_aistpp_seq()

    # # d = pd.read_pickle('/content/drive/Shareddrives/vid tokenization/acton/logs/TAN/advanced_tr_res_150.pkl')
    # # cluster_seq = d[d['seq_name']=='gWA_sFM_cAll_d25_mWA2_ch03']['cluster']
    # # cluster_seq2vid(cluster_seq[:500], '/content/drive/Shareddrives/vid tokenization/acton/logs/TAN/proxy_centers_tr_150.pkl', '/content/drive/Shareddrives/vid tokenization/seq2vid', 'coco17')

    # # cluster2vid(clusters_idx=[i for i in range(150)], sk_type='kitml',
    # #     proxy_center_info_path='/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/proxy_centers_tr_complete_150.pkl',
    # #     data_dir='/content/drive/Shareddrives/vid tokenization/asymov/kit-molan/', data_name='xyz',
    # #     frames_dir='/content/drive/Shareddrives/vid tokenization/cluster2vid')

    # data_dir = '/content/drive/Shareddrives/vid tokenization/asymov/kit-molan/'
    # data_name = 'xyz'
    # from packages.acton.src.data.dataset.loader import KITDataset
    # d = KITDataset(data_dir, data_name)

    # # seq_names=['02654']
    # # seq = d.load_keypoint3d('02654')
    # # seq_names = ['01699', #'02855',
    # #    '00826', '02031', '01920', '02664', '01834',
    # #    '02859', '00398', '03886', '01302', '02053', '00898', '03592',
    # #    '03580', '00771', '01498', '00462', '01292', '02441', '03393',
    # #    '00376', '02149', '03200', '03052', '01788', '00514', '01744',
    # #    '02977', '00243', '02874', '00396', '03597', '02654', '03703',
    # #    '00456', '00812', '00979', '00724', '01443', '03401', '00548',
    # #    '00905', '00835', #'02612',
    # #    '02388', '03788', '03870', '03181',
    # #    '00199']
    # seq_names = ["00017",
    #     "00018",
    #     "00002",
    #     "00014",
    #     "00005",
    #     "00010"]

    # #TODO use cofig to get viz paths
    # frame2cluster_mapping_path = '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/20220409_173106/advanced_tr_res_150.pkl'
    # contiguous_frame2cluster_mapping_path = '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/20220409_173106/advanced_tr_150.pkl'
    # cluster2keypoint_mapping_path = '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/20220409_173106/proxy_centers_tr_150.pkl'
    # cluster2frame_mapping_path = '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/kit_logs/tan_kitml/20220409_173106/proxy_centers_tr_complete_150.pkl'
    # sk_type = 'kitml'
    # frames_dir = '/content/drive/Shareddrives/vid tokenization/kit_reconstruction/'

    # # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, frames_dir+'very_naive')
    # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type)
    # print('very naive mpjpe : ', very_naive_mpjpe_mean)

    # # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, frames_dir+'naive')
    # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type)
    # print('naive mpjpe : ', naive_mpjpe_mean)

    # # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, frames_dir+'naive_no_rep')
    # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type)
    # print('naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)


    # #uniform filter
    # # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, filter = 'uniform', frames_dir=frames_dir+'very_naive_ufilter')
    # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, filter='uniform')
    # print('uniform filtered very naive mpjpe : ', very_naive_mpjpe_mean)

    # # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='uniform', frames_dir=frames_dir+'naive_ufilter')
    # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='uniform')
    # print('uniform filtered naive mpjpe : ', naive_mpjpe_mean)

    # # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='uniform', frames_dir=frames_dir+'naive_no_rep_ufilter')
    # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='uniform')
    # print('uniform filtered naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)

    # #spline filter
    # # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, filter = 'spline', frames_dir=frames_dir+'very_naive_sfilter')
    # very_naive_mpjpe_mean, _ = very_naive_reconstruction(seq_names, d, frame2cluster_mapping_path, cluster2keypoint_mapping_path, sk_type, filter='spline')
    # print('spline filtered very naive mpjpe : ', very_naive_mpjpe_mean)

    # # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='spline', frames_dir=frames_dir+'naive_sfilter')
    # naive_mpjpe_mean, _ = naive_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='spline')
    # print('spline filtered naive mpjpe : ', naive_mpjpe_mean)

    # # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='spline', frames_dir=frames_dir+'naive_no_rep_sfilter')
    # naive_no_rep_mpjpe_mean, _ = naive_no_rep_reconstruction(seq_names, d, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, sk_type, filter='spline')
    # print('spline filtered naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)