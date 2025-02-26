from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor, NoneType
from torch_geometric.utils import spmm


class MyGATConv(GATConv):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.1,
            add_self_loops: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        super(MyGATConv, self).__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loops, bias, **kwargs)
        self.dropout=dropout
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(
            self,
            x: Union[Tensor, OptPairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            size: Size = None,
            return_attention_weights: NoneType = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # if self.project and hasattr(self, 'lin'):
        #     x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        # out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = F.dropout(x=x, training=self.training)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')