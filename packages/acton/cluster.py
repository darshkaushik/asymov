import argparse
import os
import pprint
# import shutil
import time
# import sys
import yaml
from tqdm import tqdm
import numpy as np
# import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances_argmin_min

import pdb
# import json
import pickle

from src.data.dataset.loader import KITDataset
from src import algo
# from src.data.dataset.cluster_misc import lexicon#, get_names, genre_list

# from plb.models.self_supervised import TAN
# # from plb.models.self_supervised.tan import TANEvalDataTransform, TANTrainDataTransform
# # from plb.datamodules import SeqDataModule
# from plb.datamodules.data_transform import body_center, euler_rodrigues_rotation

# KEYPOINT_NAME = ['root','BP','BT','BLN','BUN','LS','LE','LW','RS','RE','RW',
#                 'LH','LK','LA','LMrot','LF','RH','RK','RA','RMrot','RF']

# import pytorch_lightning as pl
# pl.utilities.seed.seed_everything(0)

def plain_distance(a, b):
    return np.linalg.norm(a - b, ord=2)

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
    parser.add_argument('--batch_size',
                        default=-1,
                        help='batch size for clustering',
                        type=int)
    parser.add_argument('--force',
                        required=True,
                        help='forcefully fit kmeans or allow previously saved centers',
                        type=int)
    #TODO arg k for clusters
    args, _ = parser.parse_known_args()
    print(f'SEED: {args.seed}')
    # pl.utilities.seed.seed_everything(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    with open(args.cfg, 'r') as stream:
        ldd = yaml.safe_load(stream)

    ldd["PRETRAIN"]["DATA"]["DATA_NAME"] = args.data_name
    if args.data_dir:
        ldd["PRETRAIN"]["DATA"]["DATA_DIR"] = args.data_dir
    if args.log_dir:
        ldd["PRETRAIN"]["TRAINER"]["LOG_DIR"] = args.log_dir

    ldd["CLUSTER"]["USE_RAW"] = args.use_raw
    ldd["CLUSTER"]["FORCE"] = args.force
    ldd["CLUSTER"]["BATCH_SIZE"] = args.batch_size
    if args.batch_size > 0:
        ldd["CLUSTER"]["TYPE"] = "batch_kmeans_skl"
    if ldd["CLUSTER"]["CKPT"] == -1 :
        ldd["CLUSTER"]["CKPT"] = ldd["NAME"]
    if args.log_ver:
        ldd["CLUSTER"]["VERSION"] = str(args.log_ver)
    elif not args.use_raw:
        ldd["CLUSTER"]["VERSION"] = sorted([f.name for f in os.scandir(os.path.join(args.log_dir, ldd["CLUSTER"]["CKPT"])) if f.is_dir()], reverse=True)[0]
    pprint.pprint(ldd)
    return ldd

def create_log_viz_dirs(args):
    dirname = Path(args['CLUSTER_DIR'])
    dirname.mkdir(parents=True, exist_ok=True)
    timed = time.strftime("%Y%m%d_%H%M%S")
    # with open(os.path.join(args['CLUSTER_DIR'], f"config_used_{timed}.yaml"), "w") as stream:
    #     yaml.dump(args, stream, default_flow_style=False)
    # video_dir = os.path.join(args['CLUSTER_DIR'], "saved_videos")
    # Path(video_dir).mkdir(parents=True, exist_ok=True)

def main():

    args = parse_args()

    # KIT Dataset configs
    args['NUM_JOINTS'] = 21
    if args["CLUSTER"]["USE_RAW"] :
        args['CLUSTER_DIR'] = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], 'raw')
    else :
        args['CLUSTER_DIR'] = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], args["NAME"], args["CLUSTER"]["VERSION"])
    # print(args['CLUSTER_DIR'])
    # Create log, viz. dirs
    create_log_viz_dirs(args)

    # Load KIT Dataset from stored pkl file (e.g., xyz_data.pkl)
    official_loader = KITDataset(args["PRETRAIN"]["DATA"]["DATA_DIR"], args["PRETRAIN"]["DATA"]["DATA_NAME"])

    # Get data

    # with open(os.path.join(args['CLUSTER_DIR'], 'tr_kpt_container.pkl'), "rb") as fp:
    #     tr_kpt_container = pickle.load(fp)
    with open(os.path.join(args['CLUSTER_DIR'], 'tr_len_container.pkl'), "rb") as fp:
        tr_len_container = pickle.load(fp)
    # with open(os.path.join(args['CLUSTER_DIR'], 'tr_feat_container.pkl'), "rb") as fp:
    #     tr_feat_container = pickle.load(fp)
    with open(os.path.join(args['CLUSTER_DIR'], 'tr_name_container.pkl'), "rb") as fp:
        tr_name_container = pickle.load(fp)
    with open(os.path.join(args['CLUSTER_DIR'], 'tr_where_to_cut.pkl'), "rb") as fp:
        tr_where_to_cut = pickle.load(fp)
    tr_stacked = np.load(os.path.join(args['CLUSTER_DIR'], 'tr_stacked.npy'))

#-------------------- TODO: handle more than one splits --------------------#

    # with open(os.path.join(args['CLUSTER_DIR'], 'val_kpt_container.pkl'), "rb") as fp:
    #     val_kpt_container = pickle.load(fp)
    with open(os.path.join(args['CLUSTER_DIR'], 'val_len_container.pkl'), "rb") as fp:
        val_len_container = pickle.load(fp)
    # with open(os.path.join(args['CLUSTER_DIR'], 'val_feat_container.pkl'), "rb") as fp:
    #     val_feat_container = pickle.load(fp)
    with open(os.path.join(args['CLUSTER_DIR'], 'val_name_container.pkl'), "rb") as fp:
        val_name_container = pickle.load(fp)
    with open(os.path.join(args['CLUSTER_DIR'], 'val_where_to_cut.pkl'), "rb") as fp:
        val_where_to_cut = pickle.load(fp)
    val_stacked = np.load(os.path.join(args['CLUSTER_DIR'], 'val_stacked.npy'))

#---------------------------------------------------------------------------#


    for K in range(args["CLUSTER"]["K_MIN"], args["CLUSTER"]["K_MAX"], 10):
        # get cluster centers
        argument_dict = {"distance": plain_distance, "TYPE": "vanilla", "K": K, "TOL": 1e-4, "BATCH_SIZE": args["CLUSTER"]["BATCH_SIZE"]}
        if (args["CLUSTER"]["FORCE"]) or (not os.path.exists(os.path.join(args['CLUSTER_DIR'], f"advanced_centers_{K}.npy"))):
            print('Finding cluster centers')
            c, scores = getattr(algo, args["CLUSTER"]["TYPE"])(tr_stacked, times=args["CLUSTER"]["TIMES"], argument_dict=argument_dict)
            
            #handle empty clusters
            non_empty_clusters = np.unique(c.get_assignment(tr_stacked))
            # pdb.set_trace()
            assert np.max(non_empty_clusters)<K, "more no. of clusters than expected"
            assert len(non_empty_clusters)<=K, "more unique clusters than expected"
            
            print(f"Found {K-len(non_empty_clusters)} empty clusters")
            non_empty_cluster_centers = c.kmeans.cluster_centers_[non_empty_clusters]
            
            np.save(os.path.join(args['CLUSTER_DIR'], f"advanced_centers_{K}.npy"), non_empty_cluster_centers)
            np.save(os.path.join(args['CLUSTER_DIR'], f"advanced_centers_{K}_scores.npy"), scores)
        
        # else:
        print('Loading saved cluster centers')
        ctrs = np.load(os.path.join(args['CLUSTER_DIR'], f"advanced_centers_{K}.npy"))
        argument_dict['K'] = len(ctrs)
        print(f"Found {argument_dict['K']} clusters")
        
        c = getattr(algo, args["CLUSTER"]["TYPE"] + "_clusterer")(TIMES=args["CLUSTER"]["TIMES"], argument_dict=argument_dict)
        c.kmeans.fit(tr_stacked[:argument_dict['K']])
        c.kmeans.cluster_centers_ = ctrs

        # infer on training set and save
        y = np.concatenate([np.ones((l,)) * i for i, l in enumerate(tr_len_container)], axis=0)
        s = np.concatenate([np.arange(l) for i, l in enumerate(tr_len_container)], axis=0)
        tr_res_df = pd.DataFrame(y, columns=["y"])  # from which sequence
        cluster_l = c.get_assignment(tr_stacked)  # assigned to which cluster
        tr_res_df['cluster'] = cluster_l
        tr_res_df['frame_index'] = s  # the frame index in home sequence
        tr_res_df['seq_name'] = np.concatenate([[name] * tr_len_container[i] for i, name in enumerate(tr_name_container)], axis=0)
        tr_res_df['feat_vec'] = [[vec] for vec in tr_stacked]
        tr_res_df['dist'] = tr_res_df[['feat_vec', 'cluster']].apply(
            lambda x: plain_distance(x['feat_vec'][0],c.kmeans.cluster_centers_[x['cluster']]), #euclidean distance from cluster center
            axis=1)

        proxy_centers_tr = tr_res_df.loc[tr_res_df.groupby('cluster')['dist'].idxmin()].reset_index(drop=True)  #frames with feature vectors closest to cluster centers
        proxy_centers_tr['keypoints3d'] = proxy_centers_tr[['frame_index','seq_name']].apply(
            lambda x: official_loader.load_keypoint3d(x['seq_name'])[x['frame_index']], axis=1)   #3d skeleton keypoints of the closest frame

        # replace empty clusters
        # non_empty_clusters = proxy_centers_tr['cluster'].unique()
        # # pdb.set_trace()
        # if non_empty_clusters.shape[0] < K:
        #     empty_clusters = [i for i in range(K) if i not in non_empty_clusters]
        #     print(f"Found {len(empty_clusters)} empty clusters")
        #     closest_non_empty_clusters = non_empty_clusters[pairwise_distances_argmin_min(c.kmeans.cluster_centers_[empty_clusters], c.kmeans.cluster_centers_[non_empty_clusters])[0]]
        #     closest_non_empty_centers = proxy_centers_tr.loc[proxy_centers_tr['cluster'].isin(closest_non_empty_clusters)]
        #     closest_non_empty_centers.loc[:,('cluster', 'dist')]=np.array([ empty_clusters, [None]*len(empty_clusters)]).transpose()
        #     proxy_centers_tr = proxy_centers_tr.append(closest_non_empty_centers, ignore_index=True)

        # not needed
        # sorted_proxies_tr = tr_res_df.drop(['feat_vec'], axis=1).groupby('cluster').apply(lambda x: x.sort_values('dist')) #frames in sorted order of closeness to cluster center

        tr_word_df = pd.DataFrame(columns=["idx", "cluster", "length", "y", "name"])  # word index in home sequence
        for sequence_idx in tqdm(range(len(tr_len_container))):
            name = tr_name_container[sequence_idx]
            cluster_seq = list(cluster_l[tr_where_to_cut[sequence_idx]: tr_where_to_cut[sequence_idx + 1]]) + [-1, ]
            running_idx = 0
            prev = -1
            current_len = 0
            for cc in cluster_seq:
                if cc == prev:
                    current_len += 1
                else:
                    tr_word_df = tr_word_df.append(
                        {"idx": int(running_idx), "cluster": prev, "length": current_len, "y": sequence_idx,
                         "name": name}, ignore_index=True)
                    running_idx += 1
                    current_len = 1
                prev = cc
        tr_word_df = tr_word_df[tr_word_df["idx"] > 0]

        tr_word_df.to_pickle(Path(args['CLUSTER_DIR']) / f"advanced_tr_{argument_dict['K']}.pkl")
        print(f"advanced_tr_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}")  # saved tokenization of training set

        tr_res_df.drop(['feat_vec'], axis=1).to_pickle(Path(args['CLUSTER_DIR']) / f"advanced_tr_res_{argument_dict['K']}.pkl")
        print(f"advanced_tr_res_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}") # frame wise tokenization

        proxy_centers_tr.to_pickle(Path(args['CLUSTER_DIR']) / f"proxy_centers_tr_complete_{argument_dict['K']}.pkl")
        print(f"proxy_centers_tr_complete_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}") # saved complete proxy cluster center info
        proxy_centers_tr[['cluster', 'keypoints3d']].to_pickle(Path(args['CLUSTER_DIR']) / f"proxy_centers_tr_{argument_dict['K']}.pkl")
        print(f"proxy_centers_tr_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}") # saved proxy centers to feature vector mapping

        # not needed
        # sorted_proxies_tr.to_pickle(Path(args['CLUSTER_DIR']) / f"sorted_proxies_tr_{argument_dict['K']}.pkl")
        # print(f"sorted_proxies_tr_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}") # saved sorted proxies

#-------------------- TODO: handle more than one splits --------------------#

        #infer on validation set and save
        y = np.concatenate([np.ones((l,)) * i for i, l in enumerate(val_len_container)], axis=0)
        s = np.concatenate([np.arange(l) for i, l in enumerate(val_len_container)], axis=0)
        val_res_df = pd.DataFrame(y, columns=["y"])  # from which sequence
        cluster_l = c.get_assignment(val_stacked)  # assigned to which cluster
        val_res_df['cluster'] = cluster_l
        val_res_df['frame_index'] = s  # the frame index in home sequence
        val_res_df['seq_name'] = np.concatenate([[name] * val_len_container[i] for i, name in enumerate(val_name_container)], axis=0)
        val_res_df['feat_vec'] = [[vec] for vec in val_stacked]
        val_res_df['dist'] = val_res_df[['feat_vec', 'cluster']].apply(
            lambda x: plain_distance(x['feat_vec'][0],c.kmeans.cluster_centers_[x['cluster']]), #euclidean distance from cluster center
            axis=1)
        
        # proxy_centers_val = val_res_df.loc[val_res_df.groupby('cluster')['dist'].idxmin()].reset_index(drop=True)  #frames with feature vectors closest to cluster centers
        # proxy_centers_val['keypoints3d'] = proxy_centers_val[['frame_index','seq_name']].apply(
        #     lambda x: official_loader.load_keypoint3d(x['seq_name'])[x['frame_index']], axis=1)   #3d skeleton keypoints of the closest frame

        # non_empty_clusters = proxy_centers_val['cluster'].unique()
        # if non_empty_clusters < K:
            # empty_clusters = [i for i in range(K) if i not in non_empty_clusters]
            # closest_non_empty_clusters, _ = pairwise_distances_argmin_min(c.kmeans.cluster_centers_[empty_clusters], c.kmeans.cluster_centers_[non_empty_clusters])
            # closest_non_empty_centers = proxy_centers_val[proxy_centers_val['cluster'].isin(closest_non_empty_clusters)]
            # closest_non_empty_centers.loc[:,('cluster', 'dist')]=np.array([ empty_clusters, [None]*len(empty_clusters)]).transpose()
            # proxy_centers_val = proxy_centers_val.append(closest_non_empty_centers, ignore_index=True)

        # not needed
        # sorted_proxies_val = val_res_df.drop(['feat_vec'], axis=1).groupby('cluster').apply(lambda x: x.sort_values('dist')) #frames in sorted order of closeness to cluster center

        val_word_df = pd.DataFrame(columns=["idx", "cluster", "length", "y", "name"])  # word index in home sequence
        for sequence_idx in tqdm(range(len(val_len_container))):
            name = val_name_container[sequence_idx]
            cluster_seq = list(cluster_l[val_where_to_cut[sequence_idx]: val_where_to_cut[sequence_idx + 1]]) + [-1, ]
            running_idx = 0
            prev = -1
            current_len = 0
            for cc in cluster_seq:
                if cc == prev:
                    current_len += 1
                else:
                    val_word_df = val_word_df.append(
                        {"idx": int(running_idx), "cluster": prev, "length": current_len, "y": sequence_idx,
                         "name": name}, ignore_index=True)
                    running_idx += 1
                    current_len = 1
                prev = cc
        val_word_df = val_word_df[val_word_df["idx"] > 0]

        val_word_df.to_pickle(Path(args['CLUSTER_DIR']) / f"advanced_val_{argument_dict['K']}.pkl")
        print(f"advanced_val_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}")  # saved tokenization of validation set

        val_res_df.drop(['feat_vec'], axis=1).to_pickle(Path(args['CLUSTER_DIR']) / f"advanced_val_res_{argument_dict['K']}.pkl")
        print(f"advanced_val_res_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}") # frame wise tokenization

        # proxy_centers_val.to_pickle(Path(args['CLUSTER_DIR']) / f"proxy_centers_val_complete_{argument_dict['K']}.pkl")
        # print(f"proxy_centers_val_complete_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}") # saved proxy centers
        # proxy_centers_val[['cluster', 'keypoints3d']].to_pickle(Path(args['CLUSTER_DIR']) / f"proxy_centers_val_{argument_dict['K']}.pkl")
        # print(f"proxy_centers_val_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}")

        # not needed
        # sorted_proxies_val.to_pickle(Path(args['CLUSTER_DIR']) / f"sorted_proxies_val_{argument_dict['K']}.pkl")
        # print(f"sorted_proxies_val_{argument_dict['K']}.pkl dumped to {args['CLUSTER_DIR']}") # saved sorted proxies

#---------------------------------------------------------------------------#

if __name__ == '__main__':
    main()
