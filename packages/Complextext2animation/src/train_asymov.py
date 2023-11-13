import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from dataUtils import *
from model.model_hierarchical_twostream import *
from data import *

# from pycasper.name import Name
from BookKeeper import *
import wandb
os.environ['WANDB_API_KEY'] = 'bac3a003428df951a8e0b9e3878002a3227bbf0c'
os.environ['WANDB_DISABLE_CODE'] = 'true'  # Workaround cluster error


from argsUtils import argparseNloop
from slurmpy import Slurm
import numpy as np
from tqdm import tqdm
import copy
import pdb
import pickle
from collections import defaultdict


def train(args, exp_num):
    if args.load and os.path.isfile(args.load):
        load_pretrained_model = True
    else:
        load_pretrained_model = False
    args_subset = ['exp', 'cpk', 'model', 'time']

    # Loss hyper-params
    lmb = {k: getattr(args, k) for k in vars(args) if 'lmb_' in k}

    book = BookKeeper(args, args_subset, args_dict_update={'chunks': args.chunks,
                                                           'batch_size': args.batch_size,
                                                           'model': args.model,
                                                           's2v': args.s2v,
                                                           'cuda': args.cuda,
                                                           'save_dir': args.save_dir,
                                                           'early_stopping': args.early_stopping,
                                                           'debug': args.debug,
                                                           'stop_thresh': args.stop_thresh,
                                                           'desc': args.desc,
                                                           'curriculum': args.curriculum,
                                                           'lr': args.lr,
                                                           'lmb': lmb
                                                           },
                      # tensorboard=args.tb,
                      load_pretrained_model=load_pretrained_model)
    # load_pretrained_model makes sure that the model is loaded, old save files are not updated and _new_exp is called to assign new filename
    args = book.args

    # # Start Log
    book._start_log()
    wandb.init(project='Language2Motion', config=args.__dict__)

    # Training parameters
    path2data = args.path2data
    dataset = args.dataset
    lmksSubset = args.lmksSubset
    desc = args.desc
    split = (args.train_frac, args.dev_frac)
    idx_dependent = args.idx_dependent
    b_sz = args.batch_size
    time = args.time
    global chunks
    chunks = args.chunks
    offset = args.offset
    mask = args.mask
    feats_kind = args.feats_kind
    s2v = args.s2v
    f_new = args.f_new
    curriculum = args.curriculum

    if args.debug:
        shuffle = False
    else:
        shuffle = True

    # Load data iterables
    data_obj = None
    if args.dobj_path is None:
        data_obj = Data(path2data, dataset, lmksSubset, desc,
                    split, batch_size=b_sz,
                    time=time,
                    chunks=chunks,
                    offset=offset,
                    shuffle=shuffle,
                    mask=mask,
                    feats_kind=feats_kind,
                    s2v=s2v,
                    f_new=f_new)
        with open('dataProcessing/data_obj.pkl', 'wb') as outfile:
            pickle.dump(data_obj, outfile)
        print('Data Loaded and saved to disk.')
    else:
        with open(args.dobj_path, 'rb') as infile:
            data_obj = pickle.load(infile)

    train = data_obj.train
    dev = data_obj.dev
    test = data_obj.test

    # Create a model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_shape = data_obj.input_shape
    kwargs_keys = ['pose_size', 'trajectory_size']
    modelKwargs = {key: input_shape[key] for key in kwargs_keys}
    modelKwargs.update(args.modelKwargs)

    input_size = 4096  # the size of BERT

    discriminator = SymDiscriminator(vocab_size=args.vocab_size,
                                     embed_size=args.embed_size
                                     ).to(device).double()
    generator = SymPoseGenerator(chunks,
                                 input_size=input_size,
                                 vocab_size=args.vocab_size,
                                 Seq2SeqKwargs=modelKwargs,
                                 load=args.load).to(device).double()

    BCE = nn.BCELoss()

    optim_gen = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optim_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

    print('Model Created')
    print("Model's state_dict:")
    for param_tensor in generator.state_dict():
        print(param_tensor, "\t", generator.state_dict()[param_tensor].size())

    # Transforms
    columns = get_columns(feats_kind, data_obj)
    pre = None
    if feats_kind != 'symbol':
        pre = Transforms(args.transforms, columns, args.seed,
                     mask, feats_kind, dataset, f_new)

    def loop_train(generator, discriminator, data, epoch=0):
        '''Local method -- one epoch of training. Forward and Backward.'''

        generator.train()
        discriminator.train()

        # Logging
        iter_loss = defaultdict(list)

        Tqdm = tqdm(data, desc='train {:.10f}'.format(0), leave=False, ncols=50)
        for count, batch in enumerate(Tqdm):

            # Prep. data
            # -------------
            X, Y, s2v = batch['input'], batch['output'], batch['desc']
            pose, trajectory, start_trajectory = X
            pose_gt, trajectory_gt, start_trajectory_gt = Y

            if isinstance(s2v, torch.Tensor):
                s2v = s2v.to(device)

            x, y = None, None
            if feats_kind != 'symbol':
                x = torch.cat((trajectory, pose), dim=-1).to(device)
                y = torch.cat((trajectory_gt, pose_gt), dim=-1).to(device)

                # Transform before the model
                x = pre.transform(x)
                y = pre.transform(y)
                x = x[..., :-4]
                y = y[..., :-4]
            else:
                # TODO: Handle the subsampling better
                x = (pose/12).type(torch.cuda.LongTensor)
                y = (pose_gt/12).type(torch.cuda.LongTensor)

            # Train Disc.
            # -------------------
            optim_disc.zero_grad()

            # Create label vector = [real labels, fake labels]
            real_ls = torch.ones((b_sz, 1)).to(device='cuda').double()
            fake_ls = torch.zeros((b_sz, 1)).to(device='cuda').double()
            all_ls = torch.cat((real_ls, fake_ls))

            # Create samples = [real data, fake data]
            p_hat_l, _ = generator(x, y, s2v, lmb, train=True)
            sym_hat_l = torch.argmax(p_hat_l, dim=-1)
            all_samples = torch.cat((y.squeeze(), sym_hat_l))

            # Disc. forward pass
            out_disc = discriminator(all_samples)

            # Disc. loss
            loss_disc = args.lmb_D * BCE(out_disc, all_ls)
            iter_loss['disc'].append(loss_disc.item())

            # Disc. backward pass
            loss_disc.backward()
            optim_disc.step()

            # Train Gen.
            # -------------------
            optim_gen.zero_grad()

            # Gen. forward pass
            p_hat_l, loss_int_gen = generator(x, y, s2v, lmb, train=True)
            sym_hat_l = torch.argmax(p_hat_l, dim=-1)
            out_disc_fake = discriminator(sym_hat_l)

            # Gen. loss
            loss_gen = args.lmb_G * BCE(out_disc_fake, real_ls)
            iter_loss['gen'].append(loss_gen.item())
            for lt in loss_int_gen:
                loss_gen += loss_int_gen[lt]
                iter_loss[lt].append(loss_int_gen[lt].item())
            iter_loss['gen_full'].append(loss_gen.item())

            # Gen. backward pass
            loss_gen.backward()
            optim_gen.step()

            # Update tqdm with losses for this iteration
            Tqdm.set_description('Train Gen. {:.4f} Disc. {:.4f}'.format(
                    np.mean(iter_loss['gen_full']), np.mean(iter_loss['disc'])))
            Tqdm.refresh()

            x = x.detach()
            y = y.detach()
            loss_gen = loss_gen.detach()
            loss_disc = loss_disc.detach()
            p_hat_l = p_hat_l.detach()
            sym_hat_l = sym_hat_l.detach()
            del loss_int_gen

            # Debugging by overfitting to one batch
            if count >= 0 and args.debug:
                break

        return iter_loss

    def loop_eval(generator, discriminator, data, epoch=0):
        '''Eval. Only forward pass. Also log the losses.'''
        generator.eval()
        discriminator.eval()

        iter_loss = defaultdict(list)

        Tqdm = tqdm(data, desc='eval {:.10f}'.format(0), leave=False, ncols=50)
        for count, batch in enumerate(Tqdm):

            # Prep. data
            # ------------
            X, Y, s2v = batch['input'], batch['output'], batch['desc']
            pose, trajectory, start_trajectory = X
            pose_gt, trajectory_gt, start_trajectory_gt = Y

            if isinstance(s2v, torch.Tensor):
                s2v = s2v.to(device)

            x, y = None, None
            if feats_kind != 'symbol':
                x = torch.cat((trajectory, pose), dim=-1).to(device)
                y = torch.cat((trajectory_gt, pose_gt), dim=-1).to(device)
                # Transform before the model
                x = pre.transform(x)
                y = pre.transform(y)
                x = x[..., :-4]
                y = y[..., :-4]
            else:
                # TODO: Handle the subsampling better
                x = (pose/12).type(torch.cuda.LongTensor)
                y = (pose_gt/12).type(torch.cuda.LongTensor)

            # Disc: Create label vector = [real labels, fake labels]
            real_ls = torch.ones((b_sz, 1)).to(device='cuda').double()
            fake_ls = torch.zeros((b_sz, 1)).to(device='cuda').double()
            all_ls = torch.cat((real_ls, fake_ls))

            # Disc: Create samples = [real data, fake data]
            p_hat_l, _ = generator(x, y, s2v, lmb, train=False)
            sym_hat_l = torch.argmax(p_hat_l, dim=-1)
            all_samples = torch.cat((y.squeeze(), sym_hat_l))

            # Disc. forward pass
            out_disc = discriminator(all_samples)
            loss_disc = args.lmb_D * BCE(out_disc, all_ls)
            iter_loss['disc'].append(loss_disc.item())

            # Gen. forward pass
            p_hat_l, loss_int_gen = generator(x, y, s2v, lmb, train=False)
            sym_hat_l = torch.argmax(p_hat_l, dim=-1)
            out_disc_fake = discriminator(sym_hat_l)
            loss_gen = args.lmb_G * BCE(out_disc_fake, real_ls)
            iter_loss['gen'].append(loss_gen.item())
            for lt in loss_int_gen:
                loss_gen += loss_int_gen[lt]
                iter_loss[lt].append(loss_int_gen[lt].item())
            iter_loss['gen_full'].append(loss_gen.item())

            # update tqdm
            Tqdm.set_description('Val. Gen. {:.8f} : Disc. {:.8f}'.format(
                np.mean(iter_loss['gen_full']), np.mean(iter_loss['disc'])))
            Tqdm.refresh()

            x = x.detach()
            y = y.detach()
            loss_gen = loss_gen.detach()
            loss_disc = loss_disc.detach()
            p_hat_l = p_hat_l.detach()
            sym_hat_l = sym_hat_l.detach()

            # Debugging by overfitting
            if count >= 0 and args.debug:
                break

        return iter_loss

    num_epochs = args.num_epochs
    time_list = []
    time_list_idx = 0
    if curriculum:
        for power in range(1, int(np.log2(time-1)) + 1):
            time_list.append(2**power)
        data.update_dataloaders(time_list[0])
    time_list.append(time)
    tqdm.write('Training up to time: {}'.format(time_list[time_list_idx]))


    best_val_loss, best_epoch = np.inf, -1

    # Training Loop
    for epoch in tqdm(range(num_epochs), ncols=50):

        spl_loss = {spl: defaultdict(float) for spl in ['train', 'val', 'test']}
        spl_loss['train'] = loop_train(generator, discriminator, train, epoch=epoch)
        spl_loss['val'] = loop_eval(generator, discriminator, dev, epoch=epoch)
        spl_loss['test'] = loop_eval(generator, discriminator, test, epoch=epoch)
        scheduler.step()  # Change the Learning Rate

        # Log in wandb
        wb_dict = {}
        for spl in ['train', 'val', 'test']:
            wb_dict = log_wb(spl_loss[spl], spl, wb_dict)
        if wb_dict['val_gen_full'] < best_val_loss:
            best_val_loss = wb_dict['val_gen_full']
            best_epoch = epoch
            wandb.run.summary['best_val_gen_full'] = best_val_loss
            wandb.run.summary['best_val_gen_full_epoch'] = best_epoch
        wandb.log(wb_dict, step=epoch)

        # save results
        book.update_res({   'epoch': epoch,
                            'train': wb_dict['train_gen_full'],
                            'dev': wb_dict['val_gen_full'],
                            'test': wb_dict['test_gen_full']})

        # print results
        book.print_res(epoch, key_order=[
            'train', 'dev', 'test'], exp=exp_num, lr=scheduler.get_last_lr())
        if book.stop_training(generator, epoch):
            # if early_stopping criterion is met,
            # start training with more time steps
            time_list_idx += 1
            book.stop_count = 0  # reset the threshold counter
            book.best_dev_score = np.inf
            generator.load_state_dict(copy.deepcopy(book.best_model))
            if len(time_list) > time_list_idx:
                time_ = time_list[time_list_idx]
                data_obj.update_dataloaders(time_)
                tqdm.write('Training up to time: {}'.format(time_))


def log_wb(iter_loss, spl, wb_dict):
    '''Wandb logging utils.'''
    for lt in iter_loss:
        wb_dict['{0}_{1}'.format(spl, lt)] = np.mean(iter_loss[lt])
    return wb_dict

if __name__ == '__main__':
    argparseNloop(train)
