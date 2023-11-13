from typing import List
from pathlib import Path
import pickle
import os
import pandas as pd
import numpy as np
import pdb

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import Metric, MeanMetric
from hydra.utils import instantiate

from temos.transforms.joints2jfeats import Rifke
from temos.tools.geometry import matrix_of_angles
from temos.model.utils.tools import remove_padding
import sys
# pdb.set_trace()
sys.path.append(str(Path(__file__).resolve().parents[5]))
from viz_utils import add_traj, mpjpe3d, change_fps, very_naive_reconstruction, naive_reconstruction, naive_no_rep_reconstruction
from scipy.ndimage import uniform_filter1d, spline_filter1d

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)

def get_contiguous_cluster_seqs(seq_names: List[str], cluster_seqs: List[Tensor]):
    # pdb.set_trace()
    contiguous_frame2cluster_mapping = {"name":[], "idx":[], "cluster":[], "length":[]}
    for name, cluster_seq in zip(seq_names, cluster_seqs):
        prev=-1
        running_idx=0
        current_len = 0
        cluster_seq = np.append(cluster_seq, [-1])
        for cc in cluster_seq:
            if cc == prev:
                current_len += 1
            else:
                contiguous_frame2cluster_mapping["name"].append(name)
                contiguous_frame2cluster_mapping["idx"].append(int(running_idx))
                contiguous_frame2cluster_mapping["cluster"].append(prev)
                contiguous_frame2cluster_mapping["length"].append(current_len)
                running_idx += 1
                current_len = 1
            prev = cc
    contiguous_frame2cluster_mapping = pd.DataFrame.from_dict(contiguous_frame2cluster_mapping)
    contiguous_frame2cluster_mapping = contiguous_frame2cluster_mapping[contiguous_frame2cluster_mapping["idx"]>0]
    contiguous_cluster_seqs = [contiguous_frame2cluster_mapping[contiguous_frame2cluster_mapping['name']==name][['cluster', 'length']].reset_index(drop=True) for name in seq_names]
    return contiguous_cluster_seqs

class Perplexity(MeanMetric):
    '''
    Calculates perplexity from logits and target.
    Wrapper around SumMetric.
    '''
    def __init__(self, ignore_index: int, **kwargs):
        super().__init__(**kwargs)
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def update(self, logits: Tensor, target: Tensor):
        # Compute $$\sum PP}L(X)$$ where X = single sequence.
        # Since target = 1-hot, we use CE(.) to compute -log p_{gt}
        # pdb.set_trace()
        ce_tensor = self.CE(logits, target) #[b_sz, T]
        ppl = torch.exp(ce_tensor.mean(dim=-1)) #[b_sz]
        return super().update(ppl)

class ReconsMetrics(Metric):
    def __init__(self, traj: bool, recons_types: List[str], filters: List[str], gt_path: str,
                 recons_fps: float, pred_fps: float, gt_fps: float, num_mw_clusters: int,
                 decoding_scheme: str, beam_width: int,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if jointstype != "mmm":
            raise NotImplementedError("This jointstype is not implemented.")
        
        super().__init__()
        self.jointstype = jointstype
        self.rifke = Rifke(jointstype=jointstype,
                           normalization=False)
        
        self.force_in_meter = force_in_meter
        self.traj=traj
        self.recons_types = recons_types
        self.filters = filters
        self.recons_fps = recons_fps
        self.gt_fps = gt_fps
        self.pred_fps = pred_fps
        self.num_clusters = num_mw_clusters
        self.decoding_scheme = decoding_scheme
        if decoding_scheme == 'greedy':
            beam_width = 1
        self.beam_width = beam_width
        self.kwargs = kwargs

        gt_path = Path(gt_path)
        print("Retrieving GT data for recons loss from", gt_path)
        with open(gt_path, 'rb') as handle:
            self.ground_truth_data = pickle.load(handle)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_good", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_good_seq", default=torch.tensor(0), dist_reduce_fx="sum")

        self.MPJPE_metrics=[]
        self.APE_metrics=[]
        self.AVE_metrics=[]
        for recons_type in self.recons_types:
            for filter in self.filters:
                # APE (beam avg)
                self.add_state(f"APE_root_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"APE_traj_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"APE_pose_{recons_type}_{filter}", default=torch.zeros(20), dist_reduce_fx="sum")
                self.add_state(f"APE_joints_{recons_type}_{filter}", default=torch.zeros(21), dist_reduce_fx="sum")
                self.APE_metrics.extend([f"APE_{i}_{recons_type}_{filter}" for i in ["root", "traj", 
                                                                                     "pose", 
                                                                                     "joints"]])
                # APE (beam min)
                self.add_state(f"min_APE_root_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"min_APE_traj_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"min_APE_pose_{recons_type}_{filter}", default=torch.zeros(20), dist_reduce_fx="sum")
                self.add_state(f"min_APE_joints_{recons_type}_{filter}", default=torch.zeros(21), dist_reduce_fx="sum")
                self.APE_metrics.extend([f"min_APE_{i}_{recons_type}_{filter}" for i in ["root", "traj", 
                                                                                     "pose", 
                                                                                     "joints"]])

                # AVE (beam avg)
                self.add_state(f"AVE_root_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"AVE_traj_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"AVE_pose_{recons_type}_{filter}", default=torch.zeros(20), dist_reduce_fx="sum")
                self.add_state(f"AVE_joints_{recons_type}_{filter}", default=torch.zeros(21), dist_reduce_fx="sum")
                self.AVE_metrics.extend([f"AVE_{i}_{recons_type}_{filter}" for i in ["root", "traj", 
                                                                                     "pose", 
                                                                                     "joints"]])
                # AVE (beam min)
                self.add_state(f"min_AVE_root_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"min_AVE_traj_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.add_state(f"min_AVE_pose_{recons_type}_{filter}", default=torch.zeros(20), dist_reduce_fx="sum")
                self.add_state(f"min_AVE_joints_{recons_type}_{filter}", default=torch.zeros(21), dist_reduce_fx="sum")
                self.AVE_metrics.extend([f"min_AVE_{i}_{recons_type}_{filter}" for i in ["root", "traj", 
                                                                                     "pose", 
                                                                                     "joints"]])
                
                # MPJPE (beam avg)
                self.add_state(f"MPJPE_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.MPJPE_metrics.append(f"MPJPE_{recons_type}_{filter}")
                # MPJPE (beam min)
                self.add_state(f"min_MPJPE_{recons_type}_{filter}", default=torch.tensor(0.), dist_reduce_fx="sum")
                self.MPJPE_metrics.append(f"min_MPJPE_{recons_type}_{filter}")

        # All metric
        self.metrics = self.APE_metrics + self.AVE_metrics + self.MPJPE_metrics

    def compute(self):
        count = self.count_good
        APE_metrics = {metric: getattr(self, metric) / count for metric in self.APE_metrics}
        
        count_seq = self.count_good_seq
        AVE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.AVE_metrics}

        for recons_type in self.recons_types:
            for filter in self.filters:
                # Compute average of APEs (beam avg)
                APE_metrics[f"APE_mean_pose_{recons_type}_{filter}"] = getattr(self, f"APE_pose_{recons_type}_{filter}").mean() / count
                APE_metrics[f"APE_mean_joints_{recons_type}_{filter}"] = getattr(self, f"APE_joints_{recons_type}_{filter}").mean() / count
                # Compute average of APEs (beam min)
                APE_metrics[f"min_APE_mean_pose_{recons_type}_{filter}"] = getattr(self, f"min_APE_pose_{recons_type}_{filter}").mean() / count
                APE_metrics[f"min_APE_mean_joints_{recons_type}_{filter}"] = getattr(self, f"min_APE_joints_{recons_type}_{filter}").mean() / count
                
                # Compute average of AVEs (beam avg)
                AVE_metrics[f"AVE_mean_pose_{recons_type}_{filter}"] = getattr(self, f"AVE_pose_{recons_type}_{filter}").mean() / count_seq
                AVE_metrics[f"AVE_mean_joints_{recons_type}_{filter}"] = getattr(self, f"AVE_joints_{recons_type}_{filter}").mean() / count_seq
                # Compute average of AVEs (beam min)
                AVE_metrics[f"min_AVE_mean_pose_{recons_type}_{filter}"] = getattr(self, f"min_AVE_pose_{recons_type}_{filter}").mean() / count_seq
                AVE_metrics[f"min_AVE_mean_joints_{recons_type}_{filter}"] = getattr(self, f"min_AVE_joints_{recons_type}_{filter}").mean() / count_seq

                # Remove arrays (beam avg)
                APE_metrics.pop(f"APE_pose_{recons_type}_{filter}")
                APE_metrics.pop(f"APE_joints_{recons_type}_{filter}")
                AVE_metrics.pop(f"AVE_pose_{recons_type}_{filter}")
                AVE_metrics.pop(f"AVE_joints_{recons_type}_{filter}")
                # Remove arrays (beam min)
                APE_metrics.pop(f"min_APE_pose_{recons_type}_{filter}")
                APE_metrics.pop(f"min_APE_joints_{recons_type}_{filter}")
                AVE_metrics.pop(f"min_AVE_pose_{recons_type}_{filter}")
                AVE_metrics.pop(f"min_AVE_joints_{recons_type}_{filter}")

        # Compute average of MPJPEs
        MPJPE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.MPJPE_metrics}

        return {**APE_metrics, **AVE_metrics, **MPJPE_metrics,
                'all_seq':self.count_seq, 'good_seq':self.count_good_seq, 'good_seq_%': self.count_good_seq/self.count_seq,
                'all_pred':self.count, 'good_pred':self.count_good, 'good_pred_%': self.count_good/self.count
                }

    def update(self, seq_names: List[str], cluster_seqs: List[Tensor], traj: List[Tensor] = None):
        '''
        Args:
            seq_names: name of sequences in batch.
            cluster_seqs: cluster sequences for each beam of each sequence in the batch.
                expected order - [batch1beam1, batch2beam2, ..., batch2beam1, batch2beam2, ...]
            traj: predicted trajectory for each beam of each sequence in the batch.
                expected order - same as cluster_seqs
        '''
        if self.traj:
            assert traj is not None

        assert len(seq_names)==(len(cluster_seqs)/self.beam_width)
        
        num_seq = len(seq_names)
        self.count_seq += num_seq
        self.count += sum([cluster_seq.shape[0] for cluster_seq in cluster_seqs])
        
        #beamed cluster and traj seqs : List[List[beams]]
        beamed_cluster_seqs = [cluster_seqs[i*self.beam_width:(i+1)*self.beam_width] for i in range(num_seq)]
        beamed_traj = [traj[i*self.beam_width:(i+1)*self.beam_width] for i in range(num_seq)]
        
        # get good sequences (no <bos>, <unk> or <pad>)
        good_beams_per_seq = [[j for j in range(self.beam_width) if beam_seqs[j].max()<self.num_clusters and beam_seqs[j].min()>=0] 
                              for beam_seqs in beamed_cluster_seqs]
        good_seq_idx = [i for i, good_beams in enumerate(good_beams_per_seq) if len(good_beams)>0]
        
        # update good stuff
        seq_names = [seq_names[i] for i in good_seq_idx]
        beam_count = [len(good_beams) for good_beams in good_beams_per_seq]
        assert len(seq_names)==len(beam_count)
        seq_names_with_beams = [f"{seq_name}_{i}" for seq_name, num_beams in zip(seq_names, beam_count) 
                                for i in range(num_beams)]
        beamed_cluster_seqs = [[beamed_cluster_seqs[i][j].cpu().numpy() for j in good_beams_per_seq[i]] 
                               for i in good_seq_idx]
        beamed_traj = [[beamed_traj[i][j].cpu().numpy() for j in good_beams_per_seq[i]] 
                       for i in good_seq_idx]
        cluster_seqs = sum(beamed_cluster_seqs, [])
        traj = sum(beamed_traj, [])

        self.count_good_seq += len(seq_names)
        self.count_good += sum([cluster_seq.shape[0] for cluster_seq in cluster_seqs])

        # pdb.set_trace()
        # get contiguous cluster sequences (grouping contiguous identical clusters)
        if ('naive_no_rep' in self.recons_types) or ('naive' in self.recons_types):
            contiguous_cluster_seqs = get_contiguous_cluster_seqs(seq_names_with_beams, cluster_seqs)
        assert len(contiguous_cluster_seqs)==len(cluster_seqs)
        
        # get GT
        gt = [self.ground_truth_data[name][:5000, :, :] for name in seq_names]
        gt = [change_fps(keypoint, self.gt_fps, self.recons_fps) for keypoint in gt]
        gt_with_beams = [keypoint for keypoint, num_beams in zip(gt, beam_count) for i in range(num_beams)]
        assert len(gt_with_beams)==len(cluster_seqs)

        # reconstruct from predicted clusters using different strategies
        for recons_type in self.recons_types:
            if recons_type == 'naive_no_rep' or recons_type  == 'naive':
                cluster2frame_mapping_path = Path(self.kwargs['cluster2frame_mapping_path'])
                output = eval(recons_type+'_reconstruction')(seq_names_with_beams, contiguous_cluster_seqs, self.ground_truth_data, cluster2frame_mapping_path, verbose=False)
                if recons_type == 'naive_no_rep':
                    recons, faulty = output
                else:
                    recons = output
            elif recons_type == 'very_naive':
                cluster2keypoint_mapping_path = Path(self.kwargs['cluster2keypoint_mapping_path'])
                recons = eval(recons_type+'_reconstruction')(seq_names_with_beams, cluster_seqs, cluster2keypoint_mapping_path, verbose=False)
            # traj inclusion
            if self.traj:
                recons = add_traj(recons, traj)
            recons = [change_fps(keypoint, self.pred_fps, self.recons_fps) for keypoint in recons]

            # apply different filters
            for filter in self.filters:
                if filter == 'none':
                    pass
                elif filter == 'spline':
                    recons = [spline_filter1d(keypoint, axis=0) for keypoint in recons]
                elif filter == 'uniform':
                    recons = [uniform_filter1d(keypoint, size=int(self.recons_fps/4), axis=0) for keypoint in recons]
                else :
                    raise NameError(f'No such filter {filter}')

                # MPJPE
                mpjpe_with_beams=mpjpe3d(seq_names_with_beams, recons, gt_with_beams)
                mpjpe_per_sequence=[np.mean(mpjpe_with_beams[i::self.count_good_seq]) for i in range(self.count_good_seq)]
                getattr(self, f"MPJPE_{recons_type}_{filter}").__iadd__(np.mean(mpjpe_per_sequence))

                # length = min(gt, predicted)
                lengths = [min(recons[i].shape[0], gt_with_beams[i].shape[0]) for i in range(len(recons))]
                
                jts_text = pad_sequence([*map(torch.from_numpy, recons)], batch_first=True)
                jts_text, poses_text, root_text, traj_text = self.transform(jts_text, lengths)
                # jts_text = [torch.from_numpy(keypoint)[:l] for keypoint, l in zip(recons, lengths)]
                # poses_text = jts_text
                # root_text = [jts[..., 0, :] for jts in jts_text]
                # traj_text = [jts[..., 0, [0, 2]] for jts in jts_text]

                jts_ref = pad_sequence([*map(torch.from_numpy, gt_with_beams)], batch_first=True)
                jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref, lengths)
                # jts_ref = [torch.from_numpy(keypoint)[:l] for keypoint, l in zip(gt_with_beams, lengths)]
                # poses_ref = jts_ref
                # root_ref = [jts[..., 0, :] for jts in jts_ref]
                # traj_ref = [jts[..., 0, [0, 2]] for jts in jts_ref]

                # AVE and APE
                for start_idx, end_idx in zip(np.cumsum([0]+beam_count[:-1]), np.cumsum(beam_count)): #aggregate over beams and update
                    # APE
                    APE_root_per_beam =  torch.stack([l2_norm(root_text[i], root_ref[i], dim=1).sum() for i in range(start_idx, end_idx)])
                    mean_APE_root = APE_root_per_beam.mean(0)
                    getattr(self, f"APE_root_{recons_type}_{filter}").__iadd__(mean_APE_root)
                    min_APE_root = APE_root_per_beam.min(0)[0]
                    getattr(self, f"min_APE_root_{recons_type}_{filter}").__iadd__(min_APE_root)
                    
                    APE_traj_per_beam = torch.stack([l2_norm(traj_text[i], traj_ref[i], dim=1).sum() for i in range(start_idx, end_idx)])
                    mean_APE_traj = APE_traj_per_beam.mean(0)
                    getattr(self, f"APE_traj_{recons_type}_{filter}").__iadd__(mean_APE_traj)
                    min_APE_traj = APE_traj_per_beam.min(0)[0]
                    getattr(self, f"min_APE_traj_{recons_type}_{filter}").__iadd__(min_APE_traj)
                    
                    APE_pose_per_beam = torch.stack([l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0) for i in range(start_idx, end_idx)])
                    mean_APE_pose = APE_pose_per_beam.mean(0)
                    getattr(self, f"APE_pose_{recons_type}_{filter}").__iadd__(mean_APE_pose)
                    min_APE_pose = APE_pose_per_beam.min(0)[0]
                    getattr(self, f"min_APE_pose_{recons_type}_{filter}").__iadd__(min_APE_pose)
                    
                    APE_joints_per_beam = torch.stack([l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0) for i in range(start_idx, end_idx)])
                    mean_APE_joints = APE_joints_per_beam.mean(0)
                    getattr(self, f"APE_joints_{recons_type}_{filter}").__iadd__(mean_APE_joints)
                    min_APE_joints = APE_joints_per_beam.min(0)[0]
                    getattr(self, f"min_APE_joints_{recons_type}_{filter}").__iadd__(min_APE_joints)

                    # AVE
                    root_sigma_text = [variance(root_text[i], lengths[i], dim=0) for i in range(start_idx, end_idx)]
                    root_sigma_ref = [variance(root_ref[i], lengths[i], dim=0) for i in range(start_idx, end_idx)]
                    AVE_root_per_beam = torch.stack([l2_norm(i, j, dim=0) for i,j in zip(root_sigma_text, root_sigma_ref)])
                    mean_AVE_root = AVE_root_per_beam.mean(0)
                    getattr(self, f"AVE_root_{recons_type}_{filter}").__iadd__(mean_AVE_root)
                    min_AVE_root = AVE_root_per_beam.min(0)[0]
                    getattr(self, f"min_AVE_root_{recons_type}_{filter}").__iadd__(min_AVE_root)

                    traj_sigma_text = [variance(traj_text[i], lengths[i], dim=0) for i in range(start_idx, end_idx)]
                    traj_sigma_ref = [variance(traj_ref[i], lengths[i], dim=0) for i in range(start_idx, end_idx)]
                    AVE_traj_per_beam = torch.stack([l2_norm(i, j, dim=0) for i,j in zip(traj_sigma_text, traj_sigma_ref)])
                    mean_AVE_traj = AVE_traj_per_beam.mean(0)
                    getattr(self, f"AVE_traj_{recons_type}_{filter}").__iadd__(mean_AVE_traj)
                    min_AVE_traj = AVE_traj_per_beam.min(0)[0]
                    getattr(self, f"min_AVE_traj_{recons_type}_{filter}").__iadd__(min_AVE_traj)

                    poses_sigma_text = [variance(poses_text[i], lengths[i], dim=0) for i in range(start_idx, end_idx)]
                    poses_sigma_ref = [variance(poses_ref[i], lengths[i], dim=0) for i in range(start_idx, end_idx)]
                    AVE_pose_per_beam = torch.stack([l2_norm(i, j, dim=1) for i,j in zip(poses_sigma_text, poses_sigma_ref)])
                    mean_AVE_pose = AVE_pose_per_beam.mean(0)
                    getattr(self, f"AVE_pose_{recons_type}_{filter}").__iadd__(mean_AVE_pose)
                    min_AVE_pose = AVE_pose_per_beam.min(0)[0]
                    getattr(self, f"min_AVE_pose_{recons_type}_{filter}").__iadd__(min_AVE_pose)

                    jts_sigma_text = [variance(jts_text[i], lengths[i], dim=0) for i in range(start_idx, end_idx)]
                    jts_sigma_ref = [variance(jts_ref[i], lengths[i], dim=0) for i in range(start_idx, end_idx)]
                    AVE_joints_per_beam = torch.stack([l2_norm(i, j, dim=1) for i,j in zip(jts_sigma_text, jts_sigma_ref)])
                    mean_AVE_joints = AVE_joints_per_beam.mean(0)
                    getattr(self, f"AVE_joints_{recons_type}_{filter}").__iadd__(mean_AVE_joints)
                    min_AVE_joints = AVE_joints_per_beam.min(0)[0]
                    getattr(self, f"min_AVE_joints_{recons_type}_{filter}").__iadd__(min_AVE_joints)
                    
    def transform(self, joints: Tensor, lengths):
        features = self.rifke(joints)

        ret = self.rifke.extract(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = ret

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the local poses
        poses_local = rearrange(poses_features, "... (joints xyz) -> ... joints xyz", xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]], rotations)
        poses = torch.stack((poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local, rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # get the root joint
        root = torch.cat((trajectory[..., :, [0]],
                          root_y[..., None],
                          trajectory[..., :, [1]]), dim=-1)

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        # put back the root joint y
        poses[..., 0, 1] = root_y

        # Add the trajectory globally
        poses[..., [0, 2]] += trajectory[..., None, :]

        if self.force_in_meter:
            # return results in meters
            return (remove_padding(poses / 1000, lengths),
                    remove_padding(poses_local / 1000, lengths),
                    remove_padding(root / 1000, lengths),
                    remove_padding(trajectory / 1000, lengths))
        else:
            return (remove_padding(poses, lengths),
                    remove_padding(poses_local, lengths),
                    remove_padding(root, lengths),
                    remove_padding(trajectory, lengths))
