import torch
from torch.nn import BatchNorm1d, Sequential, Linear, ReLU

class ViewLearner(torch.nn.Module):
    def __init__(self, encoder, hidden_c, mlp_edge_model_dim=64):
        """
        初始化方法 __init__ 接收 encoder、hidden_c 和 mlp_edge_model_dim 参数。
        调用父类的初始化方法。
        设置 encoder 和 input_dim 属性。
        定义一个多层感知机（MLP）模型 mlp_edge_model，包括两个线性层和一个 ReLU 激活函数。
        调用 init_emb 方法初始化嵌入层。
        """
        super(ViewLearner, self).__init__()
        self.encoder = encoder
        self.input_dim = hidden_c
        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1)
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, graph):
        """
        定义 forward 方法，执行前向传播。
        使用 encoder 对图进行编码，得到节点嵌入 node_emb。
        获取图的边索引 edge_index，并分解为源节点索引 src 和目标节点索引 dst。
        获取源节点和目标节点的嵌入 emb_src 和 emb_dst。
        将源节点和目标节点的嵌入拼接成边嵌入 edge_emb。
        使用 MLP 模型 mlp_edge_model 计算边的 logits 值 edge_logits。
        返回边的 logits 值。`
        """
        node_emb = self.encoder(graph)
        edge_index = graph.edge_index
        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]
        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)
        return edge_logits
