import torch

from torch.nn.functional import pad
from torch.utils.data import Dataset


# Custom collate_fn function using pad_sequence
def pad_collate_fn(batch: Dataset, target_num: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    # Separate the sequences and labels in the batch
    w_seq, labels = zip(*batch)

    # Convert each sequence to a tensor
    padded_w_seq = []
    for seq in w_seq:
        seq_t = torch.tensor(seq)

        pad_size = target_num - seq_t.shape[0]
        padded_seq = pad(seq_t, (0, 0, 0, pad_size))

        padded_w_seq.append(padded_seq)

    # Stack padded sequences into a tensor
    padded_w_seq = torch.stack(padded_w_seq)

    # Stack labels into a tensor
    labels = torch.stack(labels)

    return padded_w_seq, labels


# Custom collate_fn function to flatten the website sequences
def flatten_collate_fn(batch: Dataset, target_num: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    # Padded sequences
    padded_w_seq, labels = pad_collate_fn(batch, target_num)

    # Flatten the padded_w_seq tensor
    batch_size = padded_w_seq.size(0)
    flattened_w_seq = padded_w_seq.view(batch_size, 1, -1)

    return flattened_w_seq, labels
