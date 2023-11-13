from fnmatch import translate
from typing import List, Tuple, Iterable, Optional, Dict, Union
import math
import pdb
import sys
from tqdm import tqdm

from omegaconf import DictConfig, ListConfig
from hydra.utils import instantiate

import torch
from torch import Tensor, nn
from torchmetrics import MetricCollection, Accuracy, BLEUScore

from temos.model.metrics.compute_asymov import Perplexity, ReconsMetrics
from torchmetrics import MetricCollection, Accuracy, BLEUScore, SumMetric
from temos.model.base import BaseModel
from temos.model.utils.tools import create_mask, remove_padding, greedy_decode, batch_greedy_decode, batch_beam_decode

class AsymovMT(BaseModel):
    def __init__(self, traj: bool,
                 transformer: DictConfig,
                 losses: DictConfig,
                 metrics: DictConfig,
                 optim: DictConfig,
                 text_vocab_size: int,
                 mw_vocab_size: int,
                 special_symbols: Union[List[str],ListConfig],
                #  fps: float,
                 max_frames: int,
                 metrics_start_epoch: int,
                 metrics_every_n_epoch: int,
                 decoding_scheme: str,
                 beam_width: int,
                 best_ckpt_monitors: List,
                 **kwargs):
        super().__init__()

        self.PAD_IDX, self.BOS_IDX, self.EOS_IDX, self.UNK_IDX = \
            special_symbols.index('<pad>'), special_symbols.index('<bos>'), special_symbols.index('<eos>'), special_symbols.index('<unk>')
        self.num_special_symbols = len(special_symbols)
        
        # self.fps = fps
        self.max_frames = max_frames
        
        self.metrics_start_epoch = metrics_start_epoch
        self.metrics_every_n_epoch = metrics_every_n_epoch
        self.best_ckpt_monitors = best_ckpt_monitors
        
        self.transformer = instantiate(transformer)
        # self.src_vocab_size = text_vocab_size
        self.tgt_vocab_size = mw_vocab_size
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # pdb.set_trace()

        self.optimizer = instantiate(optim, params=self.parameters())

        self._losses = MetricCollection({split: instantiate(losses, vae=False, _recursive_=False)
                                         for split in ["losses_train", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "val"]}

        self.train_metrics = {
                              'acc_teachforce': Accuracy(num_classes=mw_vocab_size, mdmc_average='samplewise',
                                              ignore_index=self.PAD_IDX, multiclass=True,# subset_accuracy=True
                                              ),
                              'bleu_teachforce': BLEUScore(),
                              'ppl_teachforce': Perplexity(self.PAD_IDX),
                             }
        self.val_metrics = {
                              'acc_teachforce': Accuracy(num_classes=mw_vocab_size, mdmc_average='samplewise',
                                              ignore_index=self.PAD_IDX, multiclass=True,# subset_accuracy=True
                                              ),
                              'bleu_teachforce': BLEUScore(),
                              'ppl_teachforce': Perplexity(self.PAD_IDX),
                            
                            #   'acc': Accuracy(num_classes=mw_vocab_size, mdmc_average='samplewise',
                            #                   ignore_index=self.PAD_IDX, multiclass=True,# subset_accuracy=True
                            #                   ),
                              'bleu': BLEUScore(),
                            #   'ppl': Perplexity(self.PAD_IDX),
                              'mpjpe': instantiate(metrics)
                             }

        self.metrics={key: getattr(self, f"{key}_metrics") for key in ["train", "val"]}

        self.decoding_scheme = decoding_scheme
        self.beam_width = beam_width 
        
        self.__post_init__()

    #TODO: add comments to understand the interleaved output of batch_beam_decode
    def batch_translate(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor, max_len: int, decoding_scheme:str = "diverse", beam_width: int = 5) -> Union[List[Tensor],Tuple[List[Tensor]]]: # no teacher forcing, takes batched input but gives unbatched output
        # src: [Frames, Batch size]
        if self.hparams.traj:
            if decoding_scheme == "greedy":
                tgt_list, traj_list = batch_greedy_decode(self.transformer, src, max_len, self.BOS_IDX, self.EOS_IDX,
                                                      src_mask, src_padding_mask)
            else:
                tgt_list, traj_list = batch_beam_decode(self.transformer, src, max_len, self.BOS_IDX, self.EOS_IDX, decoding_scheme,
                                                        src_mask, src_padding_mask, beam_width=beam_width)
            assert len(tgt_list) == len(traj_list)
            return tgt_list, traj_list #Tuple[List[Tensor[Frames]]]
        else:
            if decoding_scheme == "greedy":
                tgt_list= batch_greedy_decode(self.transformer, src, max_len, self.BOS_IDX, self.EOS_IDX,
                                                        src_mask, src_padding_mask, traj=False)
            else:
                tgt_list= batch_beam_decode(self.transformer, src, max_len, self.BOS_IDX, self.EOS_IDX, decoding_scheme,
                                                        src_mask, src_padding_mask, traj=False, beam_width=beam_width)
            return tgt_list #List[Tensor[Frames]]

    def translate(self, src_list: List[Tensor], max_len: Union[int, List[int]]) -> Union[List[Tensor],Tuple[List[Tensor]]]: # no teacher forcing
        if type(max_len)==int:
            max_len_list = [max_len]*len(src_list)
        else:
            assert len(src_list)==len(max_len)
            max_len_list = max_len
        
        if self.hparams.traj:
            traj_list=[]
        tgt_list = []
        # pdb.set_trace()
        for src, max_len in tqdm(zip(src_list, max_len_list), "translating", len(src_list), None, position=0):
            src = src.view(-1,1) #[Frames, 1]
            if self.hparams.traj:
                tgt_tokens, traj = [i.flatten() for i in greedy_decode(self.transformer, src, max_len, self.BOS_IDX, self.EOS_IDX)] #[[Frames], [Frames]]
                traj_list.append(traj)
            else:
                tgt_tokens = greedy_decode(self.transformer, src, max_len, self.BOS_IDX, self.EOS_IDX, traj=False).flatten() #[Frames]
            tgt_list.append(tgt_tokens) 
        
        if self.hparams.traj:
            assert len(tgt_list) == len(traj_list)
            return tgt_list, traj_list # Tuple[List[Tensor]]
        return tgt_list # List[Tensor[Frames]]
    
    def allsplit_step(self, split: str, batch: Dict, batch_idx):
        src: Tensor = batch["text"] #[Frames, Batch size]
        tgt: Tensor = batch["motion_words"] #[Frames, Batch size]
        tgt_input = tgt[:-1, :] #[Frames-1, Batch size]
        tgt_out = tgt[1:, :].permute(1,0) #[Batch size, Frames-1]
        if self.hparams.traj:
            tgt_traj: Tensor = batch["traj"] #[Frames, Batch size, 3]
            tgt_traj_input: Tensor = tgt_traj[:-1] #[Frames-1, Batch size, 3]
            tgt_traj_out: Tensor = remove_padding(tgt_traj[1:].permute(1,0,2), batch["length"])  #[Batch size, Frames-1, 3]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.PAD_IDX)
        
        if self.hparams.traj:
            mw_logits, traj = self.transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask, tgt_traj_input)
            #[Frames, Batch size, 3]
            traj = traj.permute(1,0,2) #[Batch size, Frames, 3]
            traj = remove_padding(traj, batch["length"]) #List[Tensor[Frames, 3]]
        else:
            mw_logits = self.transformer(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)    
        mw_logits = mw_logits.permute(1,2,0) #[Batch size, Classes, Frames]
        
        # Compute the losses
        if self.hparams.traj:
            loss = self.losses[split].update(ds_text=mw_logits, ds_ref=tgt_out, traj_text=traj, traj_ref=tgt_traj_out)
        else:
            loss = self.losses[split].update(ds_text=mw_logits, ds_ref=tgt_out)

        if loss is None:
            raise ValueError("Loss is None, this happend with torchmetrics > 0.7")
        # pdb.set_trace()

        ### Compute the metrics
        probs = mw_logits.detach().softmax(dim=1) 
        # bs, _, frames = probs.shape
        target = tgt_out.detach()
        traj = [i.detach() for i in traj]

        self.metrics[split]['acc_teachforce'].update(probs, target)

        # predicted through teacher forcing, target motion word ids without padding for BLEU
        pred_mw_tokens_teachforce = remove_padding(torch.argmax(probs, dim=1).int(), batch["length"])
        pred_mw_sents_teachforce = [" ".join(map(str, mw.int().tolist())) for mw in pred_mw_tokens_teachforce]
        target_mw_sents = [[" ".join(map(str, mw.int().tolist()))] for mw in remove_padding(target, batch["length"])]
        self.metrics[split]['bleu_teachforce'].update(pred_mw_sents_teachforce, target_mw_sents)
        
        self.metrics[split]['ppl_teachforce'].update(mw_logits.detach().cpu(), target.cpu())

        epoch = self.trainer.current_epoch
        if split == "val":
            if (self.trainer.global_step==0 or (epoch>=self.metrics_start_epoch and (epoch+1)%self.metrics_every_n_epoch==0)):
                # inferencing translations without teacher forcing
                #TODO: add max_len buffer frames to config
                # max_len = [i+int(self.fps*5) for i in batch["length"]]
                # pdb.set_trace()
                
                if self.hparams.traj:
                    pred_mw_tokens, pred_traj = self.batch_translate(src, src_mask, src_padding_mask, self.max_frames, self.decoding_scheme, self.beam_width)       #passed none so it'll pick the
                    pred_traj = [i.detach() for i in pred_traj]                                                                                                     #default values "diverse" and 5
                else:
                    pred_mw_tokens = self.batch_translate(src, src_mask, src_padding_mask, self.max_frames, self.decoding_scheme, self.beam_width)
                # pred_mw_tokens2 = self.translate(remove_padding(src.permute(1,0), batch["text_length"]), self.max_frames)
                # for mw_tokens, mw_tokens2 in zip(pred_mw_tokens, pred_mw_tokens2):
                #     assert torch.equal(mw_tokens, mw_tokens2)
                # assert len(pred_mw_tokens) == len(pred_mw_tokens2) 
                
                #TODO: aggregate BLEU over beams
                # add EOS/BOS as they were removed during translation
                pred_mw_sents = [" ".join(map(str, [self.BOS_IDX] + mw.int().tolist() + [self.EOS_IDX])) for mw in pred_mw_tokens]
                self.metrics[split]['bleu'].update(pred_mw_sents, target_mw_sents)
                
                # shift for special symbols (EOS/BOS and padding already removed)
                pred_mw_clusters = [mw - self.num_special_symbols for mw in pred_mw_tokens]
                if self.hparams.traj:
                    self.metrics[split]['mpjpe'].update(batch['keyid'], pred_mw_clusters, pred_traj)
                else:
                    self.metrics[split]['mpjpe'].update(batch['keyid'], pred_mw_clusters)

        return loss

    def allsplit_epoch_end(self, split: str, outputs):
        losses = self.losses[split]
        loss_dict = losses.compute(split)
        losses.reset()
        dico = {losses.loss2logname(loss, split): value.item()
                for loss, value in loss_dict.items()}

        #Accuracy, BLEU and Perplexity Teacher-forced
        metrics_dict_teachforce = {f"Metrics/{name}/{split}": metric.compute() for name, metric in self.metrics[split].items() if name.endswith('_teachforce')}
        _ = [metric.reset() for name, metric in self.metrics[split].items() if name.endswith('_teachforce')] 
        dico.update(metrics_dict_teachforce)

        epoch = self.trainer.current_epoch
        if split == "val":
            if (self.trainer.global_step==0 or (epoch>=self.metrics_start_epoch and (epoch+1)%self.metrics_every_n_epoch==0)):
                # pdb.set_trace()
                metrics_dict = {f"Metrics/{name}/{split}": metric.compute() for name, metric in self.metrics[split].items() if (name!='mpjpe' and not name.endswith('_teachforce'))}
                mpjpe_dict = self.metrics[split]['mpjpe'].compute()
                metrics_dict.update({f"Metrics/{name}/{split}": metric for name, metric in mpjpe_dict.items()})
                _ = [metric.reset() for name, metric in self.metrics[split].items() if not name.endswith('_teachforce')] 
                dico.update(metrics_dict)
            
        # pdb.set_trace()
        # print(dico['Metrics/acc_teachforce/val'])
        nan_metrics = {monitor: float('nan') for monitor, _ in self.best_ckpt_monitors if (monitor.split('/')[-1]==split and monitor not in dico)}
        dico.update(nan_metrics)
        # print(split, nan_metrics)
            
        dico.update({"epoch": float(self.trainer.current_epoch),
                    "step": float(self.trainer.global_step)})
        self.log_dict(dico)
