import torch
import torch.nn as nn
from encoder.gnn import GraphEncoder
from encoder.linear_encoder import LinearEncoder
from functional.ghratio import node_level_homo, node_level_struc_ratio


class AttrAtk(nn.Module):
    def __init__(
            self,
            gnn,
            in_channels,
            hidden_channels,
            gnn_layers,
            num_classes,
            dropout,
            pe_types,
            num_nodes,
            tolerance,
    ):
        super(AttrAtk, self).__init__()
        feature_encoders = []

        if 'trans_x' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['x_in'], hidden_channels['x_hidden'], 'x'))
            gnn_in = hidden_channels['x_hidden']
        else:
            gnn_in = in_channels['x_in']

        if 'mix' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['mix_in'], hidden_channels['mix_hidden'], 'mix'))
            gnn_in += hidden_channels['mix_hidden']

        if 'rwse' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['rwse_in'], hidden_channels['rwse_hidden'], 'rwse'))
            gnn_in += hidden_channels['rwse_hidden']



        '''
        列表/元组解包：在列表或元组前使用 * 号，可以将其元素解包到另一个列表或元组中。  
        list1 = [1, 2, 3]
        list2 = [*list1, 4, 5]  # 结果是 [1, 2, 3, 4, 5]
        在你的代码中，*feature_encoders 用于将 feature_encoders 列表中的元素解包并传递给 nn.Sequential 构造函数
        '''
        self.feature_encoders = nn.Sequential(*feature_encoders) #

        p_gnn, s_gnn = gnn[0], gnn[1]
        self.p_encoder = GraphEncoder(
            gnn_model=p_gnn,
            in_channels=in_channels['x_in'],
            hidden_channels=hidden_channels['gnn_hidden'] * 4, #全图的隐藏层
            s_channels=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            return_emb=False,
        )
        self.s_encoder = GraphEncoder(
            gnn_model=s_gnn,
            in_channels=gnn_in,
            hidden_channels=hidden_channels['gnn_hidden'], # TODO:*2
            s_channels=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            return_emb=False,
        )
        self.p_gnn, self.s_gnn = p_gnn, s_gnn
        self.norms = nn.ModuleList()
        self.norms.append(nn.BatchNorm1d(num_classes)) #
        self.norms.append(nn.BatchNorm1d(num_classes)) #一维归一化
        self.p_weight = torch.ones(num_nodes).unsqueeze(1) * 0.5 # 形状为 num_nodes * 1 的张量，值全为 0.5
        self.s_weight = torch.ones(num_nodes).unsqueeze(1) * 0.5
        self.tolerance = tolerance

    def forward(self, batch_p, batch_s, batch_idx):
        x_p = self.p_encoder(batch_p)[batch_idx]
        x_p = self.norms[0](x_p)

        batch_s = self.feature_encoders(batch_s)
        x_s = self.s_encoder(batch_s)
        x_s = self.norms[1](x_s)

        x = self.p_weight[batch_idx] * x_p + self.s_weight[batch_idx] * x_s

        return x

    """
    计算邻接相似性权重：  
    p_weight = node_level_homo(edge_index, label).unsqueeze(1)
    通过 node_level_homo 函数计算每个节点的邻接相似性，并将结果扩展为形状为 (num_nodes, 1) 的张量。  
    计算结构相似性权重：  
    s_weight = node_level_struc_ratio(edge_index, label, self.tolerance).to(p_weight.device)
    s_weight = torch.masked_fill(s_weight, torch.isnan(s_weight), 0)
    通过 node_level_struc_ratio 函数计算每个节点的结构相似性，并将结果转换为与 p_weight 相同的设备。然后，将所有 NaN 值替换为 0。
    """
    def update_weights(self, edge_index, label):
        p_weight = node_level_homo(edge_index, label).unsqueeze(1)             # proximity homo 邻接相似性
        s_weight = node_level_struc_ratio(edge_index, label, self.tolerance).to(p_weight.device)  # 结构相似性
        s_weight = torch.masked_fill(s_weight, torch.isnan(s_weight), 0)

        tol = p_weight + s_weight
        p_weight = torch.where(tol != 0, torch.div(p_weight, tol), 0.5)
        s_weight = torch.where(tol != 0, torch.div(s_weight, tol), 0.5)
        self.p_weight = p_weight
        self.s_weight = s_weight



class AttrAtkCommunity(nn.Module):
    def __init__(
            self,
            gnn,
            in_channels,
            hidden_channels,
            gnn_layers,
            num_classes,
            dropout,
            pe_types,
            num_nodes,
            tolerance,
    ):
        super(AttrAtk, self).__init__()
        feature_encoders = []

        if 'trans_x' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['x_in'], hidden_channels['x_hidden'], 'x'))
            gnn_in = hidden_channels['x_hidden']
        else:
            gnn_in = in_channels['x_in']

        if 'mix' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['mix_in'], hidden_channels['mix_hidden'], 'mix'))
            gnn_in += hidden_channels['mix_hidden']

        if 'rwse' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['rwse_in'], hidden_channels['rwse_hidden'], 'rwse'))
            gnn_in += hidden_channels['rwse_hidden']



        #列表/元组解包：在列表或元组前使用 * 号，可以将其元素解包到另一个列表或元组中。
        self.feature_encoders = nn.Sequential(*feature_encoders) #

        p_gnn, s_gnn = gnn[0], gnn[1]
        self.p_encoder = GraphEncoder(
            gnn_model=p_gnn,
            in_channels=in_channels['x_in'],
            hidden_channels=hidden_channels['gnn_hidden'] * 4, #全图的隐藏层
            s_channels=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            return_emb=False,
        )

        # community encoder
        self.c_encoder = GraphEncoder(
            gnn_model=s_gnn,
            in_channels=gnn_in,
            hidden_channels=hidden_channels['gnn_hidden']*4,  # community 隐藏层
            s_channels=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            return_emb=False,
        )

        self.p_gnn, self.s_gnn = p_gnn, s_gnn
        self.norms = nn.ModuleList()
        self.norms.append(nn.BatchNorm1d(num_classes)) #
        self.norms.append(nn.BatchNorm1d(num_classes)) #一维归一化
        self.p_weight = torch.ones(num_nodes).unsqueeze(1) * 0.5 # 形状为 num_nodes * 1 的张量，值全为 0.5
        self.s_weight = torch.ones(num_nodes).unsqueeze(1) * 0.5
        self.tolerance = tolerance

    def forward(self, batch_p, batch_s, batch_idx):
        x_p = self.p_encoder(batch_p)[batch_idx]
        x_p = self.norms[0](x_p)
        batch_s = self.feature_encoders(batch_s)
        x_s = self.s_encoder(batch_s)
        x_s = self.norms[1](x_s)
        x = self.p_weight[batch_idx] * x_p + self.s_weight[batch_idx] * x_s
        return x

    """
    计算邻接相似性权重：  
    p_weight = node_level_homo(edge_index, label).unsqueeze(1)
    通过 node_level_homo 函数计算每个节点的邻接相似性，并将结果扩展为形状为 (num_nodes, 1) 的张量。  
    计算结构相似性权重：  
    s_weight = node_level_struc_ratio(edge_index, label, self.tolerance).to(p_weight.device)
    s_weight = torch.masked_fill(s_weight, torch.isnan(s_weight), 0)
    通过 node_level_struc_ratio 函数计算每个节点的结构相似性，并将结果转换为与 p_weight 相同的设备。然后，将所有 NaN 值替换为 0。
    """
    def update_weights(self, edge_index, label):
        p_weight = node_level_homo(edge_index, label).unsqueeze(1)             # proximity homo 邻接相似性
        s_weight = node_level_struc_ratio(edge_index, label, self.tolerance).to(p_weight.device)  # 结构相似性
        s_weight = torch.masked_fill(s_weight, torch.isnan(s_weight), 0)

        tol = p_weight + s_weight
        p_weight = torch.where(tol != 0, torch.div(p_weight, tol), 0.5)
        s_weight = torch.where(tol != 0, torch.div(s_weight, tol), 0.5)
        self.p_weight = p_weight
        self.s_weight = s_weight