import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import os
from pathlib import Path
from tqdm import tqdm
import pdb
# pdb.set_trace()
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from viz import Viz, reconstruction, ground_truth_construction
from temos.data.utils import get_split_keyids
import temos.launch.prepare #noqa

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="viz")
def _viz(cfg: DictConfig):
    viz_obj = Viz(cfg)
    viz_obj.viz_seqs()
    # viz_obj.viz_seqs(n_samples=10)
    # viz_obj.viz_seqs(n_samples=2)
    # viz_obj.viz_seqs(l_samples=['00002', '45', 2343, 9999])
    # viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=2)
    # viz_obj.viz_seqs(l_samples=['00002', '45', 2343], n_samples=2)

@hydra.main(config_path="configs", config_name="reconstruct")
def _reconstruct(cfg: DictConfig):
    return reconstruct(cfg)

def reconstruct(cfg: DictConfig) -> None:
    # Load last config
    pred_dir = Path(hydra.utils.to_absolute_path(cfg.folder))
    data_dir = Path(cfg.datapath)
    
    logger.info("Predicted data is taken from: ")
    logger.info(f"{pred_dir}")

    # seq_names = sorted([i[:-4] for i in os.listdir(pred_dir) if i.endswith('.npy')])
    #TODO: take all seq_names
    seq_names = sorted(get_split_keyids(path=cfg.splitpath, split=cfg.split)[:50])
    logger.info(f"Total sequences: {len(seq_names)}")
    #TODO: resolve the need of viz_names
    viz_names = seq_names
    # viz_names = get_split_keyids(path=cfg.splitpath, split='recons')
    logger.info(f"Visualizing sequences: {len(viz_names)}")
    
    gt_path = data_dir / cfg.gt
    frame2cluster_mapping_dir = pred_dir
    contiguous_frame2cluster_mapping_path = pred_dir / cfg.contiguous_frame2cluster_mapping
    cluster2keypoint_mapping_path = data_dir / cfg.cluster2keypoint_mapping
    cluster2frame_mapping_path = data_dir /  cfg.cluster2frame_mapping

    #TODO : give fps directly to reconstruction function instead of ratios
    logger.info(f"Output FPS: {cfg.fps}")
    
    if cfg.visualize:
        viz_dir = pred_dir/cfg.viz_dir
        viz_dir.mkdir(exist_ok=True, parents=True)
        logger.info("Reconstructed visualizations will be stored in:")
        logger.info(f"{viz_dir}")
        if cfg.force == True:
            logger.info("All existing videos will be forcefully overwritten")
    else:
        viz_dir = None
        logger.info("Reconstructions will not be visualized")
    
    if cfg.visualize_gt:
        viz_gt_dir = data_dir / cfg.viz_gt_dir
        logger.info("Ground truth visualizations will be stored in:")
        logger.info(f"{viz_gt_dir}")
    else:
        viz_gt_dir = None
        logger.info("Ground truth will not be visualized")
    
    reconstruction('very_naive', ['none', 'uniform', 'spline'], seq_names, gt_path, cfg.sk_type, cfg.pred_fps, cfg.gt_fps, cfg.recons_fps, viz_dir, viz_names, cfg.force, 
                    cluster2keypoint_mapping_path=cluster2keypoint_mapping_path, frame2cluster_mapping_dir=frame2cluster_mapping_dir)
    reconstruction('naive', ['none', 'uniform', 'spline'], seq_names, gt_path, cfg.sk_type, cfg.pred_fps, cfg.gt_fps, cfg.recons_fps, viz_dir, viz_names, cfg.force,
                    contiguous_frame2cluster_mapping_path=contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path=cluster2frame_mapping_path)
    reconstruction('naive_no_rep', ['none', 'uniform', 'spline'], seq_names, gt_path, cfg.sk_type, cfg.pred_fps, cfg.gt_fps, cfg.recons_fps, viz_dir, viz_names, cfg.force,
                    contiguous_frame2cluster_mapping_path=contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path=cluster2frame_mapping_path)    
    
    if cfg.visualize_gt:
        ground_truth_construction(viz_names, gt_path, cfg.sk_type, cfg.gt_fps, cfg.recons_fps, viz_gt_dir, cfg.force_gt)
    
    # print('very naive mpjpe : ', very_naive_mpjpe_mean)
    # print('naive mpjpe : ', naive_mpjpe_mean)
    # print('naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)
    # print(f'{len(faulty)} faulty seqs : {faulty}')
    # print('----------------------------------------------------')
    # print('uniform filtered very naive mpjpe : ', uni_very_naive_mpjpe_mean)
    # print('uniform filtered naive mpjpe : ', uni_naive_mpjpe_mean)
    # print('uniform filtered naive (no rep) mpjpe : ', uni_naive_no_rep_mpjpe_mean)
    # print(f'{len(faulty)} faulty seqs : {faulty}')
    # print('----------------------------------------------------')
    # print('spline filtered very naive mpjpe : ', spline_very_naive_mpjpe_mean)
    # print('spline filtered naive mpjpe : ', spline_naive_mpjpe_mean)
    # print('spline filtered naive (no rep) mpjpe : ', spline_naive_no_rep_mpjpe_mean)
    # print(f'{len(faulty)} faulty seqs : {faulty}')
    
    # mpjpe_table = { 'filter': ['none', 'uniform', 'spline'], 
    #                 'very_naive':[very_naive_mpjpe_mean, uni_very_naive_mpjpe_mean, spline_very_naive_mpjpe_mean],
    #                 'naive':[naive_mpjpe_mean, uni_naive_mpjpe_mean, spline_naive_mpjpe_mean],
    #                 'naive_no_rep':[naive_no_rep_mpjpe_mean, uni_naive_no_rep_mpjpe_mean, spline_naive_no_rep_mpjpe_mean]
    #                 }
    # pd.DataFrame.from_dict(mpjpe_table).to_csv(pred_dir / "mpjpe_scores.csv")

if __name__ == '__main__':
    # _reconstruct()
    _viz()
