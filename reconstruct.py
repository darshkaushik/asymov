import argparse
import os
import yaml, pprint, json

from viz_utils import naive_no_rep_reconstruction, naive_reconstruction, very_naive_reconstruction, ground_truth_construction

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
    parser.add_argument('--data_splits',
                        help='which splits of the dataset to reconstruct',
                        nargs='*',
                        type=str)

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

    parser.add_argument('--K',
						help='k used for k means',
						type=int)
    parser.add_argument('--frames_dir',
						help='path to directory to store reconstructions',
						type=str)
    parser.add_argument('--ground',
						help='whether to construct ground truth sequences',
						default=0,
                        type=int)

    args, _ = parser.parse_known_args()
    with open(args.cfg, 'r') as stream:
        ldd = yaml.safe_load(stream)

    if args.data_dir:
        ldd["PRETRAIN"]["DATA"]["DATA_DIR"] = args.data_dir
    ldd["PRETRAIN"]["DATA"]["DATA_NAME"] = args.data_name
    ldd["PRETRAIN"]["DATA"]["DATA_SPLITS"] = args.data_splits
    
    ldd["CLUSTER"]["USE_RAW"] = args.use_raw

    if args.log_dir:
        ldd["PRETRAIN"]["TRAINER"]["LOG_DIR"] = args.log_dir
    if args.log_ver:
        ldd["CLUSTER"]["VERSION"] = str(args.log_ver)
    else:
        ldd["CLUSTER"]["VERSION"] = sorted([f.name for f in os.scandir(os.path.join(args.log_dir, ldd["CLUSTER"]["CKPT"])) if f.is_dir()], reverse=True)[0]
    
    ldd["FRAMES_DIR"] = args.frames_dir
    ldd["GROUND"] = args.ground
    ldd["K"] = args.K
    ldd["SK_TYPE"] = 'kitml'
    pprint.pprint(ldd)
    return ldd

def main():

    args = parse_args()

    # seq_names = ["00017","00018","00002","00014","00005","00010"]
    # seq_names = ['01699', #'02855', 
    #    '00826', '02031', '01920', '02664', '01834',
    #    '02859', '00398', '03886', '01302', '02053', '00898', '03592',
    #    '03580', '00771', '01498', '00462', '01292', '02441', '03393',
    #    '00376', '02149', '03200', '03052', '01788', '00514', '01744',
    #    '02977', '00243', '02874', '00396', '03597', '02654', '03703',
    #    '00456', '00812', '00979', '00724', '01443', '03401', '00548',
    #    '00905', '00835', #'02612', 
    #    '02388', '03788', '03870', '03181',
    #    '00199']
    seq_names = [
        "01459","00803","00656","03089","03098","00066","00943","01715","02349",
        "02521","01930","03740","03375","02356","00643","03959","00026","00587",
        "03017","02765","01409","01310","02320","02188","03898","03102","01156",
        "01784","00057","01175"]
    
    if args["PRETRAIN"]["DATA"]["DATA_SPLITS"] != None:
        with open(os.path.join(args["PRETRAIN"]["DATA"]["DATA_DIR"], args["PRETRAIN"]["DATA"]["DATA_NAME"] + '_data_split.json'), 'r') as handle:
            data_split = json.load(handle)
        seq_names = []
        for split in args["PRETRAIN"]["DATA"]["DATA_SPLITS"]:
            #TODO fraction of data to create reconstructions
            seq_names.extend(data_split[split])
    
    data_path = os.path.join(args["PRETRAIN"]["DATA"]["DATA_DIR"], args["PRETRAIN"]["DATA"]["DATA_NAME"] + '_data.pkl')
    if args["CLUSTER"]["USE_RAW"] :
        log_dir = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], 'raw')
    else :
        log_dir = os.path.join(args["PRETRAIN"]["TRAINER"]["LOG_DIR"], args["NAME"], args["CLUSTER"]["VERSION"])
    if(args["PRETRAIN"]["DATA"]["DATA_SPLITS"]==['train']):
        split='tr'
    else:
        split='val'
    frame2cluster_mapping_path = os.path.join(log_dir, f'advanced_{split}_res_{args["K"]}.pkl')
    contiguous_frame2cluster_mapping_path = os.path.join(log_dir, f'advanced_{split}_{args["K"]}.pkl')
    cluster2keypoint_mapping_path = os.path.join(log_dir, f'proxy_centers_tr_{args["K"]}.pkl')
    cluster2frame_mapping_path = os.path.join(log_dir, f'proxy_centers_tr_complete_{args["K"]}.pkl')

    
    if args["FRAMES_DIR"] == None:
        #No filter
        very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"])
        naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"])
        naive_no_rep_mpjpe_mean, faulty = naive_no_rep_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"])
        
        #uniform filter
        uni_very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], filter = 'uniform')
        uni_naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='uniform')
        uni_naive_no_rep_mpjpe_mean, faulty = naive_no_rep_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='uniform')
        
        #spline filter
        spline_very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], filter = 'spline')
        spline_naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='spline')
        spline_naive_no_rep_mpjpe_mean, faulty = naive_no_rep_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='spline')

    else:
        #No filter
        # very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], frames_dir=args["FRAMES_DIR"]+'very_naive')
        # naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], frames_dir=args["FRAMES_DIR"]+'naive')
        # naive_no_rep_mpjpe_mean, faulty = naive_no_rep_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], frames_dir=args["FRAMES_DIR"]+'naive_no_rep')
        
        #uniform filter
        uni_very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], filter = 'uniform', frames_dir=args["FRAMES_DIR"]+'very_naive_ufilter')
        uni_naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='uniform', frames_dir=args["FRAMES_DIR"]+'naive_ufilter')
        uni_naive_no_rep_mpjpe_mean, faulty = naive_no_rep_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='uniform', frames_dir=args["FRAMES_DIR"]+'naive_no_rep_ufilter')
        
        #spline filter
        spline_very_naive_mpjpe_mean = very_naive_reconstruction(seq_names, data_path, frame2cluster_mapping_path, cluster2keypoint_mapping_path, args["SK_TYPE"], filter = 'spline', frames_dir=args["FRAMES_DIR"]+'very_naive_sfilter')
        spline_naive_mpjpe_mean = naive_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='spline', frames_dir=args["FRAMES_DIR"]+'naive_sfilter')
        spline_naive_no_rep_mpjpe_mean, faulty = naive_no_rep_reconstruction(seq_names, data_path, contiguous_frame2cluster_mapping_path, cluster2frame_mapping_path, args["SK_TYPE"], filter='spline', frames_dir=args["FRAMES_DIR"]+'naive_no_rep_sfilter')
    
    if args["GROUND"]:
        #original video
        ground_truth_construction(seq_names, data_path, args["SK_TYPE"], frames_dir='./kit_reconstruction/ground/')

    print('very naive mpjpe : ', very_naive_mpjpe_mean)
    print('naive mpjpe : ', naive_mpjpe_mean)
    print('naive (no rep) mpjpe : ', naive_no_rep_mpjpe_mean)
    print(f'faulty seqs : {faulty}')
    print('----------------------------------------------------')
    print('uniform filtered very naive mpjpe : ', uni_very_naive_mpjpe_mean)
    print('uniform filtered naive mpjpe : ', uni_naive_mpjpe_mean)
    print('uniform filtered naive (no rep) mpjpe : ', uni_naive_no_rep_mpjpe_mean)
    print(f'faulty seqs : {faulty}')
    print('----------------------------------------------------')
    print('spline filtered very naive mpjpe : ', spline_very_naive_mpjpe_mean)
    print('spline filtered naive mpjpe : ', spline_naive_mpjpe_mean)
    print('spline filtered naive (no rep) mpjpe : ', spline_naive_no_rep_mpjpe_mean)
    print(f'faulty seqs : {faulty}')

if __name__ == '__main__':
    main()