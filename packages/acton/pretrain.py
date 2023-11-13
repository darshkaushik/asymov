import argparse
import os
import pprint
import sys

import json
import yaml
import shutil
import time
import logging, json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from plb.models.self_supervised import TAN
from plb.models.self_supervised.tan import TANEvalDataTransform, TANTrainDataTransform
from plb.datamodules import KITSeqDataModule
from pytorch_lightning.plugins import DDPPlugin

import wandb
os.environ['WANDB_API_KEY'] = ''
os.environ['WANDB_DISABLE_CODE'] = 'true'  # Workaround cluster error


KEYPOINT_NAME = ['root','BP','BT','BLN','BUN','LS','LE','LW','RS','RE','RW',
				'LH','LK','LA','LMrot','LF','RH','RK','RA','RMrot','RF']

def parse_args():
	parser = argparse.ArgumentParser(description='Train classification network')

	parser.add_argument('--cfg',
						help='experiment configure file name',
						required=True,
						type=str)

	parser.add_argument('--data_dir',
						help='path to data directory from repo root',
						type=str)
	parser.add_argument('--data_name',
						help='which version of the dataset, subset or not',
						default=1,
						type=str)

	parser.add_argument('--log_dir',
						help='path to directory to store logs (kit_logs) directory',
						type=str)

	parser.add_argument('--seed',
						help='seed for this run',
						default=1,
						type=int)


	args, _ = parser.parse_known_args()
	pl.utilities.seed.seed_everything(args.seed)
	with open(args.cfg, 'r') as stream:
		ldd = yaml.safe_load(stream)

	ldd["PRETRAIN"]["DATA"]["DATA_NAME"] = args.data_name
	if args.data_dir:
		ldd["PRETRAIN"]["DATA"]["DATA_DIR"] = args.data_dir
	if args.log_dir:
		ldd["PRETRAIN"]["TRAINER"]["LOG_DIR"] = args.log_dir
	pprint.pprint(ldd)
	return ldd


def main():
	args = parse_args()

	# Log, viz. dirs
	timed = time.strftime("%Y%m%d_%H%M%S")
	log_dir = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], args["NAME"], timed)
	dirpath = Path(log_dir)
	dirpath.mkdir(parents=True, exist_ok=True)
	with open(os.path.join(log_dir, "config.yaml"), "w") as stream:
		yaml.dump(args, stream, default_flow_style=False)
	# video_dir = os.path.join(log_dir, f"saved_videos")
	# Path(video_dir).mkdir(parents=True, exist_ok=True)

	# Model
	j = 21
	dm = KITSeqDataModule(**args["PRETRAIN"]["DATA"])
	transform_args = {"min_length": args["PRETRAIN"]["DATA"]["MIN_LENGTH"],
					  "max_length": args["PRETRAIN"]["DATA"]["MAX_LENGTH"],
					  "aug_shift_prob": args["PRETRAIN"]["DATA"]["AUG_SHIFT_PROB"],
					  "aug_shift_range": args["PRETRAIN"]["DATA"]["AUG_SHIFT_RANGE"],
					  "aug_rot_prob": args["PRETRAIN"]["DATA"]["AUG_ROT_PROB"],
					  "aug_rot_range": args["PRETRAIN"]["DATA"]["AUG_ROT_RANGE"],
					  "aug_time_prob": args["PRETRAIN"]["DATA"]["AUG_TIME_PROB"],
					  "aug_time_rate": args["PRETRAIN"]["DATA"]["AUG_TIME_RATE"], }
	dm.train_transforms = eval(args["PRETRAIN"]["ALGO"] + "TrainDataTransform")(**transform_args)
	dm.val_transforms = eval(args["PRETRAIN"]["ALGO"] + "EvalDataTransform")(**transform_args)
	model = eval(args["PRETRAIN"]["ALGO"])(
		gpus=args["PRETRAIN"]["GPUS"],
		num_samples=dm.num_samples,
		batch_size=dm.batch_size,
		length=dm.min_length,
		dataset=dm.name,
		max_epochs=args["PRETRAIN"]["EPOCH"],
		warmup_epochs=args["PRETRAIN"]["WARMUP"],
		arch=args["PRETRAIN"]["ARCH"]["ARCH"],
		val_configs=args["PRETRAIN"]["VALIDATION"],
		learning_rate=float(args["PRETRAIN"]["TRAINER"]["LR"]),
		log_dir=log_dir,
		protection=args["PRETRAIN"]["PROTECTION"],
		optim=args["PRETRAIN"]["TRAINER"]["OPTIM"],
		lars_wrapper=args["PRETRAIN"]["TRAINER"]["LARS"],
		tr_layer=args["PRETRAIN"]["ARCH"]["LAYER"],
		tr_dim=args["PRETRAIN"]["ARCH"]["DIM"],
        out_dim=args["PRETRAIN"]["ARCH"]["OUT_DIM"],
		neg_dp=args["PRETRAIN"]["ARCH"]["DROPOUT"],
		j=j*3,
	)

	# Logger
	wandb_logger = WandbLogger(project=args['NAME'])
	wandb_logger.watch(model, log='all', log_freq=100)

	# Log best model (one with min lowest 3 val. losses)
	checkpoint_callback = ModelCheckpoint(
								monitor='val_loss',
								dirpath=os.path.join(log_dir,  'checkpoints'),
								filename='best_ckpt_{epoch:03d}-{val_loss:.2f}',
								save_top_k=3,
								mode='min'
							)

	# trainer
	trainer = pl.Trainer(
		gpus=args["PRETRAIN"]["GPUS"],
		check_val_every_n_epoch=args["PRETRAIN"]["TRAINER"]["VAL_STEP"],
		logger=wandb_logger,
		log_every_n_steps = args["PRETRAIN"]["TRAINER"]["LOG_STEP"],
		accelerator=args["PRETRAIN"]["TRAINER"]["ACCELERATOR"],
		max_epochs=args["PRETRAIN"]["EPOCH"],
		gradient_clip_val=0.5,
		num_sanity_val_steps=0,
		plugins=DDPPlugin(find_unused_parameters=False),
		callbacks=[checkpoint_callback]
	)
	trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
	main()
