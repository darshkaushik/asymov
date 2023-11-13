
import argparse
import os
import pprint
# import shutil
# import time
# import sys
import yaml
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from pathlib import Path

import pdb
import json
import pickle

from src.data.dataset.loader import KITDataset
# from src import algo
# from src.data.dataset.cluster_misc import lexicon#, get_names, genre_list

# from plb.models.self_supervised import TAN
# from plb.models.self_supervised.tan import TANEvalDataTransform, TANTrainDataTransform
# from plb.datamodules import SeqDataModule
from plb.datamodules.data_transform import body_center, euler_rodrigues_rotation

KEYPOINT_NAME = ['root','BP','BT','BLN','BUN','LS','LE','LW','RS','RE','RW',
                'LH','LK','LA','LMrot','LF','RH','RK','RA','RMrot','RF']

import pytorch_lightning as pl
pl.utilities.seed.seed_everything(0)

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
                        default='xyz',
                        type=str)

    parser.add_argument('--seed',
                        help='seed for this run',
                        default=1,
                        type=int)

    parser.add_argument('--log_dir',
						help='path to directory to store logs (kit_logs) directory',
						type=str)
    parser.add_argument('--log_ver',
                        help='version in kitml_logs',
                        type=str)

    parser.add_argument('--use_raw',
                        required=True,
                        help='whether to use raw skeleton for clustering',
                        type=int)
    
    args, _ = parser.parse_known_args()
    print(f'SEED: {args.seed}')
    pl.utilities.seed.seed_everything(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.cfg, 'r') as stream:
        ldd = yaml.safe_load(stream)

    ldd["PRETRAIN"]["DATA"]["DATA_NAME"] = args.data_name
    if args.data_dir:
        ldd["PRETRAIN"]["DATA"]["DATA_DIR"] = args.data_dir
    if args.log_dir:
        ldd["PRETRAIN"]["TRAINER"]["LOG_DIR"] = args.log_dir

    ldd["CLUSTER"]["USE_RAW"] = args.use_raw
    # if args.use_raw :
    #     ldd["EMBEDDING"]["TYPE"] = 'raw'
    # else :
    #     ldd["EMBEDDING"]["TYPE"] = 'tan'
    if ldd["CLUSTER"]["CKPT"] == -1 :
        ldd["CLUSTER"]["CKPT"] = ldd["NAME"]
    if args.log_ver:
        ldd["CLUSTER"]["VERSION"] = str(args.log_ver)
    elif not args.use_raw:
        ldd["CLUSTER"]["VERSION"] = sorted([f.name for f in os.scandir(os.path.join(args.log_dir, ldd["CLUSTER"]["CKPT"])) if f.is_dir()], reverse=True)[0]
    pprint.pprint(ldd)
    return ldd


def get_model(args):
    '''Identify checkpoint to use, create log files, and  return model'''
    print('Using TAN model\'s features for clustering')
    load_name = args["CLUSTER"]["CKPT"]
    # with open(os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], f"val_cluster_zrsc_scores.txt"), "a") as f:
    #     f.write(f"EXP: {load_name}\n")
    cfg = None
    for fn in os.listdir(args['EMBED_DIR']):
        if fn.endswith(".yaml"):
            cfg = fn
    with open(os.path.join(args['EMBED_DIR'], cfg), 'r') as stream:
        old_args = yaml.safe_load(stream)
    cpt_name = os.listdir(os.path.join(args['EMBED_DIR'], "checkpoints"))[0]
    print(f"We are using checkpoint: {cpt_name}")
    model = eval(old_args["PRETRAIN"]["ALGO"]).load_from_checkpoint(os.path.join(args['EMBED_DIR'], "checkpoints", cpt_name))
    return model


def get_feats(args, ldd, model=None):
    ''''''
    feats = None
    if int(args["CLUSTER"]["USE_RAW"]) == 0:
        ldd1 = torch.Tensor(ldd).flatten(1, -1) #/ 100  # [T, 63]
        ttl = ldd1.shape[0]
        ct = body_center(ldd1[0])
        ldd1 -= ct.repeat(args['NUM_JOINTS']).unsqueeze(0)
        res1 = model(ldd1.unsqueeze(0).to(args['DEVICE']),
                     torch.tensor([ttl]).to(args['DEVICE']))
        forward_feat = res1[:, 0]  # [T1, f]
        forward_feat /= torch.linalg.norm(forward_feat, dim=-1, keepdim=True, ord=2)
        feats = forward_feat
    else:
        # to get results for using raw skeleton, swap with
        ldd1 = torch.Tensor(ldd).flatten(1, -1) #/ 100  # [T, 63]
        ttl = ldd1.shape[0]
        ct = body_center(ldd1[0])
        ldd1 -= ct.repeat(args['NUM_JOINTS']).unsqueeze(0)
        feats = ldd1
    return feats

def main():

    args = parse_args()

    # KIT Dataset configs
    args['NUM_JOINTS'] = 21
    if args["CLUSTER"]["USE_RAW"] :
        args['EMBED_DIR'] = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], 'raw')
    else :
        args['EMBED_DIR'] = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], args["NAME"], args["CLUSTER"]["VERSION"])

    # print(args['EMBED_DIR'])

    # Load KIT Dataset from stored pkl file (e.g., xyz_data.pkl)
    official_loader = KITDataset(args["PRETRAIN"]["DATA"]["DATA_DIR"], args["PRETRAIN"]["DATA"]["DATA_NAME"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['DEVICE'] = device

    # Load model only if we are using TAN Featuers ("USE_RAW" == 0)
    model = None
    if int(args["CLUSTER"]["USE_RAW"]) == 0:
        model = get_model(args)
        torch.set_grad_enabled(False)
        model.eval()
        model = model.to(args['DEVICE'])
    
    with open(os.path.join(args["PRETRAIN"]["DATA"]["DATA_DIR"], 'data_split.json'), 'r') as handle:
        data_split = json.load(handle)

    # Get data
    # tr_kpt_container = []
    tr_len_container = []
    tr_feat_container = []
    tr_name_container = []

    
    tr_df = data_split['train']
    print(f"Training samples = {len(tr_df)}")
    
    for reference_name in tqdm(tr_df, desc='Loading training set features'):
        try:
            ldd = official_loader.load_keypoint3d(reference_name)

            # TODO: Temp. debug hack -- truncate to T=5000
            ldd = ldd[:5000, :, :]

            # print(reference_name, ldd.shape[0])
            # tr_kpt_container.append(ldd)
            tr_len_container.append(ldd.shape[0])
            feats = get_feats(args, ldd, model)
            tr_feat_container.append(feats.detach().cpu().numpy())
            tr_name_container.append(reference_name)
        except:
            print(f'ERROR w/ seq. {reference_name}. In except: block')
    
    tr_where_to_cut = [0, ] + list(np.cumsum(np.array(tr_len_container)))
    tr_stacked = np.vstack(tr_feat_container)

    #Save data

    # with open(os.path.join(args['EMBED_DIR'], 'tr_kpt_container.pkl'), "wb") as fp: 
    #     pickle.dump(tr_kpt_container, fp)
    # print(f"tr_kpt_container.pkl dumped to {args['EMBED_DIR']}") 
    with open(os.path.join(args['EMBED_DIR'], 'tr_len_container.pkl'), "wb") as fp: 
        pickle.dump(tr_len_container, fp)
    print(f"tr_len_container.pkl dumped to {args['EMBED_DIR']}")
    # with open(os.path.join(args['EMBED_DIR'], 'tr_feat_container.pkl'), "wb") as fp: 
    #     pickle.dump(tr_feat_container, fp)
    # print(f"tr_feat_container.pkl dumped to {args['EMBED_DIR']}")
    with open(os.path.join(args['EMBED_DIR'], 'tr_name_container.pkl'), "wb") as fp: 
        pickle.dump(tr_name_container, fp)
    print(f"tr_name_container.pkl dumped to {args['EMBED_DIR']}")

    #TODO: check np.savez
    with open(os.path.join(args['EMBED_DIR'], 'tr_where_to_cut.pkl'), "wb") as fp: 
        pickle.dump(tr_where_to_cut, fp)
    print(f"tr_where_to_cut.pkl dumped to {args['EMBED_DIR']}")
    np.save(os.path.join(args['EMBED_DIR'], 'tr_stacked.npy'), tr_stacked)
    print(f"tr_stacked.npy dumped to {args['EMBED_DIR']}")

    del tr_stacked, tr_where_to_cut, tr_len_container, tr_name_container
#-------------------- TODO: handle more than one splits --------------------#

    # val_kpt_container = []
    val_len_container = []
    val_feat_container = []
    val_name_container = []
    val_df = data_split['val'] + data_split['test']
    print(f"Validation samples = {len(val_df)}")


    for reference_name in tqdm(val_df, desc='Loading validation set features'):
        try:
            ldd = official_loader.load_keypoint3d(reference_name)

            # FIXME: Temp. debug hack -- truncate to T=5000
            ldd = ldd[:5000, :, :]

            # val_kpt_container.append(ldd)
            val_len_container.append(ldd.shape[0])
            feats = get_feats(args, ldd, model)
            val_feat_container.append(feats.detach().cpu().numpy())
            val_name_container.append(reference_name)
        except:
            print(f'ERROR w/ seq. {reference_name}. In except: block')


    val_where_to_cut = [0, ] + list(np.cumsum(np.array(val_len_container)))
    val_stacked = np.vstack(val_feat_container)


    # with open(os.path.join(args['EMBED_DIR'], 'val_kpt_container.pkl'), "wb") as fp: 
    #     pickle.dump(val_kpt_container, fp)
    # print(f"val_kpt_container.pkl dumped to {args['EMBED_DIR']}")
    with open(os.path.join(args['EMBED_DIR'], 'val_len_container.pkl'), "wb") as fp: 
        pickle.dump(val_len_container, fp)
    print(f"val_len_container.pkl dumped to {args['EMBED_DIR']}")
    # with open(os.path.join(args['EMBED_DIR'], 'val_feat_container.pkl'), "wb") as fp: 
    #     pickle.dump(val_feat_container, fp)
    # print(f"val_feat_container.pkl dumped to {args['EMBED_DIR']}")
    with open(os.path.join(args['EMBED_DIR'], 'val_name_container.pkl'), "wb") as fp: 
        pickle.dump(val_name_container, fp)
    print(f"val_name_container.pkl dumped to {args['EMBED_DIR']}")
    

    with open(os.path.join(args['EMBED_DIR'], 'val_where_to_cut.pkl'), "wb") as fp: 
        pickle.dump(val_where_to_cut, fp)
    print(f"val_where_to_cut.pkl dumped to {args['EMBED_DIR']}")
    np.save(os.path.join(args['EMBED_DIR'], 'val_stacked.npy'), val_stacked)
    print(f"val_stacked.npy dumped to {args['EMBED_DIR']}")

#---------------------------------------------------------------------------#

if __name__ == '__main__':
    main()