import torch

from torch import nn
from torch_geometric.nn import GlobalAttention
import torch.nn.functional as F

class WSimNet(nn.Module):
    def __init__(self, in_channels=459, out_channels=128):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),

            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),

            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),

            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )

        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )

        self.pool = GlobalAttention(
            gate_nn=self.gate_nn
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, out_channels),
            nn.BatchNorm1d(out_channels),
        )

        self.last_att_weights = None

    def forward(self, x, batch):
        x = self.node_encoder(x)           # Encoding each node using MLP

        # Compute gate score manually to extract attention weights
        gate_scores = self.gate_nn(x).squeeze(-1)  # shape: [N]

        # Softmax within each graph in batch
        att_weights = torch.zeros_like(gate_scores)
        for b in torch.unique(batch):
            idx = (batch == b)
            att_weights[idx] = F.softmax(gate_scores[idx], dim=0)

        self.last_att_weights = att_weights.detach()  # store for debugging

        x = self.pool(x, batch)            # Attention Pooling
        x = self.mlp(x)                    # Projection MLP Layer

        return x                           # 輸出嵌入（與原本相同介面）