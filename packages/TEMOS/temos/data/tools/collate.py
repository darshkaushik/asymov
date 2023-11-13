from typing import Callable, List, Dict
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import pdb


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_datastruct_and_text(lst_elements: List) -> Dict:
    collate_datastruct = lst_elements[0]["datastruct"].transforms.collate

    batch = {
        # Collate with padding for the datastruct
        "datastruct": collate_datastruct([x["datastruct"] for x in lst_elements]),
        # Collate normally for the length
        "length": [x["length"] for x in lst_elements],
        # Collate the text
        "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


def collate_text_and_length(lst_elements: Dict) -> Dict:
    batch = {"length": [x["length"] for x in lst_elements],
             "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch and x != "datastruct"]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch

def motion_words_padding(batch: List[Tensor], vocab_size: int) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_ones(size=size)*vocab_size
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b-vocab_size)
    return canvas

def collate_motion_words_and_text(lst_elements: List, vocab_size: int) -> Dict:
    batch = {
        # Collate with padding for the datastruct
        "motion_words": motion_words_padding([x["motion_words"] for x in lst_elements], vocab_size),
        # Collate normally for the length
        "length": [x["length"] for x in lst_elements],
        # Collate the text
        "text": [x["text"] for x in lst_elements]}
    
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
def text_transform(text: str, text_vocab: Vocab):
    text_tokens = tokenizer(text) #Tokenization
    text_token_ids = text_vocab(text_tokens) #Numericalization
    # pdb.set_trace()
    text_token_ids = torch.tensor(text_vocab(['<bos>'])+text_token_ids+text_vocab(['<eos>'])) # Add BOS/EOS
    return text_token_ids

def mw_transform(mw_cluster_ids: List[int], special_symbols: List[str]): 
    mw_token_ids = mw_cluster_ids + len(special_symbols) # Shift token id to accomodate special symbols in mw vocab
    mw_token_ids = torch.cat((torch.tensor([special_symbols.index('<bos>')]), 
                             mw_token_ids,
                             torch.tensor([special_symbols.index('<eos>')]))) # Add BOS/EOS assuming special first
    return mw_token_ids

def traj_transform(traj_xyz: Tensor):
    traj_xyz = traj_xyz.float() #double to float
    traj_xyz = torch.cat((traj_xyz.new_zeros((1,3)),
                          traj_xyz,
                          traj_xyz.new_zeros((1,3))))
    return traj_xyz

def collate_motion_words_and_text_mt(lst_elements: List, text_vocab:Vocab, special_symbols: List[str], traj: bool = True) -> Dict:

    text_batch, mw_batch = [], []
    for x in lst_elements:
        text_sample, mw_sample = x['text'], x['motion_words']
        text_batch.append(text_transform(text_sample.rstrip("\n"), text_vocab))
        mw_batch.append(mw_transform(mw_sample, special_symbols))

    PAD_IDX = text_vocab.__getitem__('<pad>')
    assert PAD_IDX == special_symbols.index('<pad>')
    
    batch = {
        "text": pad_sequence(text_batch, padding_value=PAD_IDX), #[Frames, Batch size]
        "motion_words": pad_sequence(mw_batch, padding_value=PAD_IDX), #[Frames, Batch size]
        "text_length": [len(x) for x in text_batch],
        "length": [len(x) for x in mw_batch]
        }
    
    if traj:
        # pdb.set_trace()
        traj_batch = [traj_transform(x['traj']) for x in lst_elements] #List[Tensor[Frames, 3]]
        # batch["traj"] = traj_batch
        batch["traj"] = pad_sequence(traj_batch, padding_value=0.0) #[Frames, Batch size, 3]
    
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch
    