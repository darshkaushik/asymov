import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import os
from os.path import join as ospj
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pdb

import torch
import pytorch_lightning as pl

import sys

# import temos.launch.prepare  # noqa
# from temos.data.tools.collate import *
# from temos.model.utils.beam_search import beam_search
# from temos.data.sampling import upsample


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='configs', config_name='sample_asymov')
def _sample_for_viz(cfg: DictConfig):
    return sample_for_viz(cfg)


def cfg_mean_nsamples_resolution(cfg):
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1


def get_path(sample_path: Path, gender: str, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    path = sample_path / f"{fact_str}{gender}_{split}{extra_str}"
    return path


def load_checkpoint(model, last_ckpt_path, *, eval_mode):
    # Load the last checkpoint
    # model = model.load_from_checkpoint(last_ckpt_path)
    # this will overide values
    # for example relative to rots2joints
    # So only load state dict is preferable
    try:
        ckpt = torch.load(last_ckpt_path)
    except:
        #TODO handle multi-gpu
        print('Device mismatch when loading checkpoint !')
        ckpt = torch.load(last_ckpt_path, map_location='cpu')
    finally:
        model.load_state_dict(ckpt["state_dict"])
        logger.info("Model weights restored.")

    if eval_mode:
        model.eval()
        logger.info("Model in eval mode.")


def sample_for_viz(newcfg: DictConfig):

    # Load previous config
    runcfg_p = Path(Path(newcfg.ckpt_p).parent.parent, '.hydra/config.yaml')
    prevcfg = OmegaConf.load(Path('packages/TEMOS', runcfg_p))

    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)

    print('Loading checkpoint: ', cfg.ckpt_p)
    print('Samples: ', cfg.l_samples)
    print('Viz. dir: ', cfg.viz_dir)

    # onesample = cfg_mean_nsamples_resolution(cfg)

    # Path to save predictions
    pred_path = Path(cfg.viz_dir, 'asymov_preds')
    # path = get_path(storage, cfg.gender, cfg.split, onesample, cfg.mean, cfg.fact)
    pred_path.mkdir(exist_ok=True, parents=True)
    logger.info(f'Sample script. The outputs will be stored in: {pred_path}')

    pl.seed_everything(cfg.seed)


    pdb.set_trace()
    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    # Instantiate all modules specified in the configs
    logger.info("Loading model")
    model = instantiate(cfg.model,
                        # nfeats=data_module.nfeats,
                        logger_name="wandb",
                        nvids_to_save=None,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")

    logger.info(f'Loading checkpoint: ', cfg.ckpt_p)
    load_checkpoint(model, cfg.ckpt_p, eval_mode=True)

    # if "amass" in cfg.data.dataname and "xyz" not in cfg.data.dataname:
    #     model.transforms.rots2joints.jointstype = cfg.jointstype

    model.sample_mean = cfg.mean
    model.fact = cfg.fact

    pdb.set_trace()
    if not model.hparams.vae and cfg.number_of_samples > 1:
        raise TypeError("Cannot get more than 1 sample if it is not a VAE.")

    dataset = getattr(data_module, f"{cfg.split}_dataset")

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)

    # frame2cluster_mapping = pd.DataFrame(columns=["frame_index", "cluster", "seq_name"])
    # contiguous_frame2cluster_mapping = pd.DataFrame(columns=["idx", "cluster", "length", "name"])
    contiguous_frame2cluster_mapping = {"name":[], "idx":[], "cluster":[], "length":[]}

    with torch.no_grad():
        pbar = tqdm(dataset.keyids)
        for keyid in pbar:
            pbar.set_description(f"Processing {keyid}")
            for index in range(cfg.number_of_samples):
                one_data = dataset.load_keyid(keyid)
                duration = one_data['length']
                # batch_size = 1 for reproductability
                batch = collate_motion_words_and_text([one_data], cfg.data.vocab_size)
                # fix the seed
                pl.seed_everything(index)

                # if cfg.jointstype == "vertices":
                #     vertices = model(batch)[0]
                #     motion = vertices.numpy()
                #     # no upsampling here to keep memory
                #     # vertices = upsample(vertices, cfg.data.framerate, 100)
                # else:
                probs = model(batch)[0].permute(1, 0).numpy() #[Frames, Classes]
                # upscaling to compare with other methods
                # probs = upsample(probs, cfg.data.framerate, 100)
                assert probs.shape == (duration, cfg.data.vocab_size)

                clusters = np.argmax(probs, axis=1)
                assert np.max(clusters) < cfg.data.vocab_size

                if cfg.number_of_samples > 1:
                    name = f"{keyid}_{index}"
                    npypath = path / f"{name}.npy"
                else:
                    name = f"{keyid}"
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
    contiguous_frame2cluster_mapping = pd.DataFrame.from_dict(contiguous_frame2cluster_mapping)
    contiguous_frame2cluster_mapping = contiguous_frame2cluster_mapping[contiguous_frame2cluster_mapping["idx"]>0]
    contiguous_frame2cluster_mapping.to_pickle(path/"contiguous_frame2cluster_mapping.pkl")

    logger.info("All the sampling are done")
    logger.info(f"All the sampling are done. You can find them here:\n{path}")



if __name__ == '__main__':
    _sample_for_viz()
