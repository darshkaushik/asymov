from typing import Tuple, List, Union
import torch
from torch import Tensor
from torch.nn import Module, Transformer as T
from tqdm import tqdm
import pdb
from .beam_search import beam_search_auto, diverse_beam_search_auto, diverse_beam_search_unit, beam_search_unit

def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

# dummy wrapper to make the usage clear
def remove_padding_and_EOS(tensors, lengths):
    return remove_padding(tensors, lengths)
    
def remove_padding_asymov(tensors, lengths):
    return [tensor[:, :tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

def create_mask(src: Tensor, tgt: Tensor, PAD_IDX: int) -> Tuple[Tensor]:
    # src: [Frames, Batch size], tgt: [Frames-1, Batch size]
    src_seq_len = src.shape[0] #Frames
    tgt_seq_len = tgt.shape[0] #Frames-1

    tgt_mask = T.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device) #[tgt_seq_len, tgt_seq_len]
    src_mask = src.new_zeros((src_seq_len, src_seq_len), dtype=torch.bool) #[src_seq_len, src_seq_len]

    src_padding_mask = (src == PAD_IDX).transpose(0, 1) #[Batch size, Frames]
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1) #[Batch size, Frames-1]
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# function to generate output sequence using greedy algorithm
def greedy_decode(model: Module, src: Tensor, max_len: int, start_symbol: int, end_symbol: int, traj: bool = True) -> Tensor:
    # src: [Frames, 1]
    num_tokens = src.shape[0]
    src_mask = src.new_zeros((num_tokens, num_tokens), dtype=torch.bool) # [Frames, Frames]
    memory = model.encode(src, src_mask) #[Frames, 1, *]
    
    # pdb.set_trace()
    tgt = src.new_full((1, 1), start_symbol, dtype=torch.long)
    if traj:
        tgt_traj = src.new_zeros((1, 3), dtype=torch.long)

    for i in tqdm(range(max_len), leave=False):
        tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))
                    .to(tgt.device, dtype=torch.bool))
        if traj:
            out = model.decode(tgt, memory, tgt_mask, tgt_traj = tgt_traj) #[Frames, 1, *]
            next_root = model.traj_generator(out[-1]) #[3]
            tgt_traj = torch.cat([tgt_traj, next_root.unsqueeze(0)]) #[Frames+1, 3]
        else:
            out = model.decode(tgt, memory, tgt_mask) #[Frames, 1, *]
        logits = model.generator(out[-1]) #[1, Classes]
        _, next_word = torch.max(logits, dim=-1)
        next_word = next_word.item()

        tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break
    
    if traj:
        return tgt, tgt_traj
    return tgt #[Frames, 1]

def batch_greedy_decode(model: Module, src: Tensor, max_len: int, start_symbol: int, end_symbol: int,
                  src_mask: Tensor = None, src_padding_mask: Tensor = None, traj: bool = True) -> Union[List[Tensor], Tuple[List[Tensor]]]:
    # src: [Frames, Batches]
    if src_mask is None:
        num_tokens = src.shape[0]
        src_mask = src.new_zeros(num_tokens, num_tokens, dtype=torch.bool) # [Frames, Frames]
    
    memory = model.encode(src, src_mask, src_padding_mask) #[Frames, Batches, *]
    
    # pdb.set_trace()
    batch_size = src.shape[1]
    tgt = src.new_full((1, batch_size),  start_symbol, dtype=torch.long) #[1, Batch size], 1 as for 1st frame
    tgt_len = tgt.new_full((batch_size,), max_len) #[Batch Size] #same dtype as tgt
    if traj:
        tgt_traj = src.new_zeros((1, batch_size, 3), dtype=torch.long) #[1, Batch size, 3], 1 as for 1st frame

    # effective frame predictions
    for i in tqdm(range((max_len)), "autoregressive translation", None):
        tgt_mask = (T.generate_square_subsequent_mask(tgt.size(0))
                    .to(tgt.device, dtype=torch.bool))
        if i==0:
            tgt_padding_mask = tgt.new_full((batch_size, 1), False, dtype=torch.bool) #[Batch Size, 1], 1 as for 1st frame
        else:
            tgt_padding_mask = torch.cat([tgt_padding_mask, (tgt_len<=i).unsqueeze(1)], dim=1)

        if traj:
            out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask, tgt_traj = tgt_traj) #[Frames, Batch Size, *]
            next_root = model.traj_generator(out[-1]) #[Batch Size, 3]
            tgt_traj = torch.cat([tgt_traj, next_root.unsqueeze(0)]) #[Frames+1, Batch size, 3]
        else:
            out = model.decode(tgt, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask) #[Frames, Batch Size, *]
        logits = model.generator(out[-1]) #[Batch Size, Classes]
        next_word = torch.argmax(logits, dim=-1) #[Batch Size]
        tgt = torch.cat([tgt, next_word.unsqueeze(0)]) #[Frames+1, Batch size]
        # tgt2 = torch.argmax(model.generator(out), dim=-1)
        # assert torch.equal(tgt[1:], tgt2)
        
        # if EOS then effective length of o/p = i (for (i+1)th iter), else same as init (max_len)
        tgt_len = torch.where(torch.logical_and(next_word==end_symbol, tgt_len==max_len), i, tgt_len)
        if (tgt_len>i).sum()==0: #break if each batch prediction got an EOS
            break
    
    #remove BOS ([1:] slicing), EOS and padding and return effective frame predictions
    tgt_list = remove_padding_and_EOS(tgt[1:].permute(1, 0), tgt_len)
    if traj:
        tgt_traj_list =  remove_padding_and_EOS(tgt_traj[1:].permute(1, 0, 2), tgt_len)
        return tgt_list, tgt_traj_list #Tuple[List[Tensor[Frames]]]
    return tgt_list #List[Tensor[Frames]]

def beam_decode(model: Module, src: Tensor, max_len: int, start_symbol: int, end_symbol: int, decoding_scheme: str ="diverse",
                      src_mask: Tensor = None, src_padding_mask: Tensor = None, traj: bool = True, beam_width: int = 5) -> Tensor:
  #It'll use diverse by default
  decode_dict = {
      "diverse": diverse_beam_search_unit, 
      "beam": beam_search_unit,
  }

  # src: [Frames, Batches]
  if src_mask is None:
      num_tokens = src.shape[0]
      src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)  # [Frames, Frames]

  # batch_size = src.shape[1]
  tgt = src.new_full((1, 1), start_symbol, dtype=torch.long)   #start symb, 1st frame

  if traj:
    pdb.set_trace()           ##
    tgt_list, tgt_traj_list = decode_dict[decoding_scheme](model, src, tgt, src_mask,src_padding_mask, 
                                                          end_symbol, max_len, beam_width)
    return tgt_list, tgt_traj_list  # List[Tensor[Frames]]; List_len-> batch*beam
                                    #b:batch_element,B:beam; b1B1,b2B1,b1B2,b2B2.... 
  else:
    tgt_list = decode_dict[decoding_scheme](model, src, tgt, src_mask,src_padding_mask, 
                                          end_symbol, max_len, beam_width, traj=False)
    return tgt_list  # List[Tensor[Frames]]


def batch_beam_decode(model: Module, src: Tensor, max_len: int, start_symbol: int, end_symbol: int, decoding_scheme: str ="diverse",
                        src_mask: Tensor = None, src_padding_mask: Tensor = None, traj: bool = True, beam_width: int = 5) -> Tensor:
    #It'll use diverse by default
    decode_dict = {
        "diverse": diverse_beam_search_auto, 
        "beam": beam_search_auto,
    }

    # src: [Frames, Batches]
    if src_mask is None:
        num_tokens = src.shape[0]
        src_mask = (src.new_zeros(num_tokens, num_tokens)).type(torch.bool)  # [Frames, Frames]

    batch_size = src.shape[1]
    tgt = src.new_ones(1, batch_size).fill_(start_symbol).type(torch.long)  # [1, Batch size], 1 as for 1st frame

    if traj:
        tgt_list, tgt_traj_list = decode_dict[decoding_scheme](model, src, tgt, src_mask, src_padding_mask, end_symbol, 
                                                              max_len, batch_size, beam_width)
        return tgt_list, tgt_traj_list  # List[Tensor[Frames]]; List_len-> beam*batch
                                        #b:batch_element,B:beam; b1B1,b1B2,b1B3,b1B4,b1B5,b2B1,b2B2,b2B3... 
    else:
        tgt_list = decode_dict[decoding_scheme](model, src, tgt, src_mask,src_padding_mask, end_symbol,
                                        max_len, batch_size, beam_width, traj=False)
        return tgt_list  # List[Tensor[Frames]]
        
        # if (i + 2) < max_len:
        #     break

    # tgt_list = remove_padding(tgt.permute(1, 0), tgt_len)
    # return tgt_list  # List[Tensor[Frames]]

# TODO: Unified search fucntion
# def batch_decode(model: Module, src: Tensor, max_len: int, start_symbol: int, end_symbol: int, beam: bool, beam_width: int,
#                   src_mask: Tensor = None, src_padding_mask: Tensor = None) -> Tensor:
#     if beam:
#         assert beam_width>1
#     else:
#         assert beam_width==1

