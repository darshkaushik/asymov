import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pdb

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import temos.launch.prepare  # noqa
from temos.data.tools.collate import *
from temos.data.sampling import upsample
from temos.model.utils.beam_search import beam_search
from temos.model.utils.tools import create_mask

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="sample_asymov_mt")
def _sample(cfg: DictConfig):
    return sample(cfg)

def get_path(sample_path: Path, gender: str, split: str):
    extra_str = ""
    fact_str = ""
    path = sample_path / f"{fact_str}{gender}_{split}{extra_str}"
    return path


def load_checkpoint(model, ckpt_path, *, eval_mode):
    # Load the last checkpoint
    # model = model.load_from_checkpoint(ckpt_path)
    # this will overide values
    # for example relative to rots2joints
    # So only load state dict is preferable
    # pdb.set_trace()
    try:
        ckpt = torch.load(ckpt_path)
    except:
        #TODO handle multi-gpu
        print('Device mismatch when loading checkpoint !')
        ckpt = torch.load(ckpt_path, map_location='cpu')
    # finally:
    model.load_state_dict(ckpt["state_dict"])
    logger.info("Model weights restored.")

    if eval_mode:
        model.eval()
        logger.info("Model in eval mode.")


def sample(newcfg: DictConfig) -> None:

    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))

    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)

    logger.info("Sample script. The outputs will be stored in:")
    storage = Path(os.getcwd()) / "samples" #${hydra.run.dir}/samples
    path = get_path(storage, cfg.gender, cfg.split)
    path.mkdir(exist_ok=True, parents=True)
    logger.info(f"{path}")

    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")
    
    if OmegaConf.is_missing(cfg.model, 'text_vocab_size'):
        cfg.model.text_vocab_size = data_module.text_vocab_size
    if OmegaConf.is_missing(cfg.model, 'mw_vocab_size'):
        cfg.model.mw_vocab_size = data_module.mw_vocab_size
    if OmegaConf.is_missing(cfg.model, 'max_frames'):
        cfg.model.max_frames = data_module.max_frames
    max_frames = data_module.max_frames
    
    dataloader = DataLoader(getattr(data_module, f"{cfg.split}_dataset"), shuffle=False, **data_module.dataloader_options)
    logger.info(f"Retrieved {cfg.split}_dataloader")
    
    logger.info("Loading model")
    model = instantiate(cfg.model,
                        logger_name="wandb",
                        # nvids_to_save=None,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    ckpt_path = cfg.ckpt_path
    logger.info(f"Using checkpoint {ckpt_path}")
    load_checkpoint(model, ckpt_path, eval_mode=True)



    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)

    frame2cluster_mapping = {}
    contiguous_frame2cluster_mapping = {"name":[], "idx":[], "cluster":[], "length":[]}
    with torch.no_grad():
        pbar = tqdm(dataloader, "Sampling")
        for batch in pbar:
            src: Tensor = batch["text"] #[Frames, Batch size]
            tgt: Tensor = batch["motion_words"] #[Frames, Batch size]
            tgt_input = tgt[:-1, :] #[Frames-1, Batch size]
            src_mask, _, src_padding_mask, _ = create_mask(src, tgt_input, model.PAD_IDX)
            
            pred_mw_tokens = model.batch_translate(src, src_mask, src_padding_mask, max_frames)
            assert len(batch["keyid"]) == len(pred_mw_tokens)
            
            for keyid, clusters in zip(batch["keyid"], pred_mw_tokens):
                name = f"{keyid}"
                clusters = np.array(clusters)
                frame2cluster_mapping[name] = clusters
                
                npypath = path / f"{name}.npy"
                np.save(npypath, clusters)

                prev=-1
                running_idx=0
                current_len = 0
                clusters = np.append(clusters, [-1])
                for cc in clusters:
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
    
    frame2cluster_mapping = pd.DataFrame.from_dict(frame2cluster_mapping, orient='index')
    frame2cluster_mapping.to_pickle(path/"frame2cluster_mapping.pkl")
    
    contiguous_frame2cluster_mapping = pd.DataFrame.from_dict(contiguous_frame2cluster_mapping)
    contiguous_frame2cluster_mapping = contiguous_frame2cluster_mapping[contiguous_frame2cluster_mapping["idx"]>0]
    contiguous_frame2cluster_mapping.to_pickle(path/"contiguous_frame2cluster_mapping.pkl")

    logger.info(f"All the sampling are done. You can find them here:\n{path}")


if __name__ == '__main__':
    _sample()
