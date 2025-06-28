import torch


def make_weights_for_balanced_classes(labels):
    unique_labels, counts = torch.unique(labels, return_counts=True)
    weight_per_class = torch.sum(counts.to(torch.float)) / counts.to(
        torch.float
    )
    weights = [0] * len(labels)
    for i, val in enumerate(labels):
        weights[i] = weight_per_class[torch.where(unique_labels == val)[0][0]]
    return weights