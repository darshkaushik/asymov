from pathlib import Path
import pdb
from typing import List
import torch
from torch import Tensor

def get_split_keyids(path: str, split: str):
    # pdb.set_trace()
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")

# def sequential_transforms(*transforms):
#     def func(txt_input):
#         for transform in transforms:
#             txt_input = transform(txt_input)
#         return txt_input
#     return func

# def mt_terminal_transform(token_ids: List[int], BOS_IDX: int, EOS_IDX: int)->Tensor:
#     return torch.cat((torch.tensor([BOS_IDX]),
#                       torch.tensor(token_ids),
#                       torch.tensor([EOS_IDX])))

# def mt_mwid_transform(motion_words: List[int], num_special: int)->List[int]:
#     return [i+num_special for i in motion_words]