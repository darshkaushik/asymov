#@title
import argparse
import os
import pprint, pickle
import sys
sys.path.insert(0, '/content/drive/Shareddrives/vid tokenization/asymov/packages/acton')

import json
import yaml
import shutil
import time
import logging, json
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from plb.models.self_supervised import TAN
from plb.models.self_supervised.tan import TANEvalDataTransform, TANTrainDataTransform
# from plb.datamodules import KITSeqDataModule
from pytorch_lightning.plugins import DDPPlugin

KEYPOINT_NAME = ['root','BP','BT','BLN','BUN','LS','LE','LW','RS','RE','RW',
                'LH','LK','LA','LMrot','LF','RH','RK','RA','RMrot','RF']

def parse_args():
    # parser = argparse.ArgumentParser(description='Train classification network')

    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # parser.add_argument('--data_dir',
    #                     help='path to aistplusplus data directory from repo root',
    #                     type=str)
    
    # parser.add_argument('--seed',
    #                     help='seed for this run',
    #                     default=1,
    #                     type=int)

    # args, _ = parser.parse_known_args()
    # pl.utilities.seed.seed_everything(args.seed)
    with open('/content/drive/Shareddrives/vid tokenization/asymov/packages/acton/configs/tan.yaml', 'r') as stream:
        ldd = yaml.safe_load(stream)

    # if args.data_dir:
    ldd["PRETRAIN"]["DATA"]["DATA_DIR"] = '/content/drive/Shareddrives/vid tokenization/asymov/kit-molan/'
    # pprint.pprint(ldd)
    return ldd


args = parse_args()
# debug = args["NAME"] == "debug"
# log_dir = os.path.join("./kit_logs", args["NAME"])

# dirpath = Path(log_dir)
# dirpath.mkdir(parents=True, exist_ok=True)

# timed = time.strftime("%Y%m%d_%H%M%S")
# with open(os.path.join(log_dir, f"config_used_{timed}.yaml"), "w") as stream:
#     yaml.dump(args, stream, default_flow_style=False)
# video_dir = os.path.join(log_dir, "saved_videos")
# Path(video_dir).mkdir(parents=True, exist_ok=True)

# # log
# tt_logger = TestTubeLogger(
#     save_dir=log_dir,
#     name="default",
#     debug=False,
#     create_git_tag=False
# )

# # trainer
# trainer = pl.Trainer(
#     gpus=None,#args["PRETRAIN"]["GPUS"],
#     check_val_every_n_epoch=args["PRETRAIN"]["TRAINER"]["VAL_STEP"],
#     logger=tt_logger,
#     accelerator=args["PRETRAIN"]["TRAINER"]["ACCELERATOR"],
#     max_epochs=args["PRETRAIN"]["EPOCH"],
#     gradient_clip_val=0.5,
#     num_sanity_val_steps=0,
#     plugins=DDPPlugin(find_unused_parameters=False),
# )

j = 21
# dm = KITSeqDataModule(**args["PRETRAIN"]["DATA"])
transform_args = {"min_length": args["PRETRAIN"]["DATA"]["MIN_LENGTH"],
                    "max_length": args["PRETRAIN"]["DATA"]["MAX_LENGTH"],
                    "aug_shift_prob": args["PRETRAIN"]["DATA"]["AUG_SHIFT_PROB"],
                    "aug_shift_range": args["PRETRAIN"]["DATA"]["AUG_SHIFT_RANGE"],
                    "aug_rot_prob": args["PRETRAIN"]["DATA"]["AUG_ROT_PROB"],
                    "aug_rot_range": args["PRETRAIN"]["DATA"]["AUG_ROT_RANGE"],
                    "aug_time_prob": args["PRETRAIN"]["DATA"]["AUG_TIME_PROB"],
                    "aug_time_rate": args["PRETRAIN"]["DATA"]["AUG_TIME_RATE"], }
train_transforms = eval(args["PRETRAIN"]["ALGO"] + "TrainDataTransform")(**transform_args)
val_transforms = eval(args["PRETRAIN"]["ALGO"] + "EvalDataTransform")(**transform_args)
# model = eval(args["PRETRAIN"]["ALGO"])(
#     gpus=args["PRETRAIN"]["GPUS"],
#     num_samples=dm.num_samples,
#     batch_size=dm.batch_size,
#     length=dm.min_length,
#     dataset=dm.name,
#     max_epochs=args["PRETRAIN"]["EPOCH"],
#     warmup_epochs=args["PRETRAIN"]["WARMUP"],
#     arch=args["PRETRAIN"]["ARCH"]["ARCH"],
#     val_configs=args["PRETRAIN"]["VALIDATION"],
#     learning_rate=float(args["PRETRAIN"]["TRAINER"]["LR"]),
#     log_dir=log_dir,
#     protection=args["PRETRAIN"]["PROTECTION"],
#     optim=args["PRETRAIN"]["TRAINER"]["OPTIM"],
#     lars_wrapper=args["PRETRAIN"]["TRAINER"]["LARS"],
#     tr_layer=args["PRETRAIN"]["ARCH"]["LAYER"],
#     tr_dim=args["PRETRAIN"]["ARCH"]["DIM"],
#     neg_dp=args["PRETRAIN"]["ARCH"]["DROPOUT"],
#     j=j*3, 
# )

with open('/content/drive/Shareddrives/vid tokenization/asymov/kit-molan/xyz_data.pkl', 'rb') as handle:
    xyz_data = pickle.load(handle)
xyz_data = xyz_data['00001']
# xyz_data = np.random.rand(378,17,3)
sample = torch.flatten(torch.from_numpy(xyz_data), start_dim=1).float()
sample.shape

xi, xj, veloi, veloj = train_transforms(sample)

xi = xi.numpy()
xj = xj.numpy()
xi=np.reshape(xi, (-1,21,3))
xj=np.reshape(xj, (-1,21,3))

# os.chdir('/content/drive/Shareddrives/vid tokenization/asymov')
import viz as viz
# viz.viz_seq(xyz_data['00001'], '/content/drive/Shareddrives/vid tokenization/kit_viz/00001', 'kitml')
viz.viz_seq(xi, '/content/drive/Shareddrives/vid tokenization/kit_viz/00001_transform3', 'kitml')
viz.viz_seq(xj, '/content/drive/Shareddrives/vid tokenization/kit_viz/00001_transform4', 'kitml')