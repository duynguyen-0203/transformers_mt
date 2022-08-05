import random

import numpy as np
import torch


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_device(batch: dict, device: torch.device) -> dict:
    converted_batch = dict()
    for key in batch.keys():
        converted_batch[key] = batch[key].to(device)

    return converted_batch


def extend_tensor(tensor: torch.tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape
    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


def padded_stack(tensors: torch.tensor, padding=0):
    dim_count = len(tensors[0].shape)
    max_shape = [max([tensor.shape[d] for tensor in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for tensor in tensors:
        e = extend_tensor(tensor, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)

    return stacked


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return mask


def create_mask(src: torch.Tensor, tgt: torch.Tensor, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    return src_mask, tgt_mask
