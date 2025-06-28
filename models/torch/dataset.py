import torch
import itertools
import numpy as np

from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from models.torch.data import WebsiteNamePair, WebsitePair


# Define WebsiteFeatureDataset
class WebsiteFeatureDataset(Dataset):
    def __init__(self, company_labels, labels_map, features):
        self.data = []
        self.company_labels = company_labels
        self.labels_map = labels_map
        self.features = features.item()

        for site_name, label in self.company_labels.items():
            self.data.append(
                {
                    'feature': np.array(self.features[site_name]),
                    'label': str(label)
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded_number = torch.tensor(self.labels_map[self.data[idx]['label']])
        return self.data[idx]['feature'], one_hot(encoded_number, len(self.labels_map)).float()


# Define WebsitePairwiseDataset
class WebsitePairwiseDataset(Dataset):
    def __init__(self, company_labels, features, loss='pairwise'):
        self.company_labels = company_labels
        self.features = features.item()
        self.loss = loss
        self.pairs, self.labels = self._build_website_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return WebsitePair(
            w1_feat=np.array(self.features[self.pairs[idx].w1]),
            w2_feat=np.array(self.features[self.pairs[idx].w2]),
            labels=self.labels[idx]
        )

    def _build_website_pairs(self):
        pairs = []
        labels = []

        for (w1, l1), (w2, l2) in itertools.combinations(self.company_labels.items(), 2):
            pairs.append(WebsiteNamePair(
                w1=w1,
                w2=w2
            ))
            if (l1 == l2):
                labels.append(1)
            else:
                labels.append(-1 if self.loss == 'pairwise' else 0)

        return pairs, labels


# Define PairwiseSynthesizedFeatureDataset
class PairwiseSynthesizedFeatureDataset(Dataset):
    def __init__(self, features, loss='pairwise'):
        self.X = features['X']
        self.Y = features['Y']
        self.loss = loss
        self.pairs, self.labels = self._build_website_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return WebsitePair(
            w1_feat=np.array(self.pairs[idx][0], dtype='float32'),
            w2_feat=np.array(self.pairs[idx][1], dtype='float32'),
            labels=self.labels[idx]
        )

    def _build_website_pairs(self):
        pairs = []
        labels = []

        for (v1, l1), (v2, l2) in itertools.combinations(zip(self.X, self.Y), 2):
            pairs.append((v1, v2))

            if (l1 == l2):
                labels.append(1)
            else:
                labels.append(-1 if self.loss == 'pairwise' else 0)

        return pairs, labels


# Define SynthesizedFeatureDataset
class SynthesizedFeatureDataset(Dataset):
    def __init__(self, features):
        # Extract X, Y
        self.X = torch.Tensor(features['X'])
        self.Y = features['Y']

        # Create a mapping of string categories to integer labels
        self.label_map = {label: idx for idx, label in enumerate(set(self.Y))}
        
        # Convert string categories to integer categories
        self.Y = torch.Tensor([self.label_map[label] for label in self.Y]).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def get_label_map(self):
        return self.label_map