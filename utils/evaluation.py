import torch

from sklearn import metrics


# Compute the AUC for pair classification
def auc(scores, labels, mode='pairwise'):
    """Compute the AUC for pair classification.

    See `tf.metrics.auc` for more details about this metric.

    Args:
      scores: [n_examples] float.  Higher scores mean higher preference of being
        assigned the label of +1.
      labels: [n_examples] int.  Labels are either +1 or -1.

    Returns:
      auc: the area under the ROC curve.
    """
    scores_max = torch.max(scores)
    scores_min = torch.min(scores)

    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    if (mode == 'pairwise'):
        labels = (labels + 1) / 2

    fpr, tpr, thresholds = metrics.roc_curve(
        labels.cpu().detach().numpy(), scores.cpu().detach().numpy())
    return metrics.auc(fpr, tpr)
