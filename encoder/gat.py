from torch import nn
import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
import torch.nn.functional as F

from encoder.gat_conv import MyGATConv

from functional.pooling import my_global_add_pool, center_node_pooling


class GAT(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            s_channels,
            num_layers,
            dropout,
            return_emb=False,
            center_add_pooling=True,
    ):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(MyGATConv(in_channels, hidden_channels))
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(MyGATConv(in_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        if not center_add_pooling:
            self.lin2 = Linear(hidden_channels * num_layers, s_channels)
        else:
            self.lin2 = Linear(hidden_channels * num_layers* 2, s_channels)
        self.center_add_pooling = center_add_pooling

        self.dropout = dropout
        self.act = nn.ReLU(hidden_channels)
        self.num_layers = num_layers
        self.return_emb = return_emb
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            # x = self.convs[i](x, edge_index, edge_weight)
            x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1 and self.return_emb:
                return x
            if i == self.num_layers - 1:
                break
            x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)
        return x