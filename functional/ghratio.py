import torch
from torch_geometric.utils import subgraph, degree
from torch_scatter import scatter
import networkx as nx


def node_level_homo(edge_index, label):
    """
    #某个节点的相邻节点中有多少比例与自身具有相同的标签
    :param edge_index:
    :param label:
    :return:
    """
    row, col = edge_index
    out = torch.zeros(row.size(0), device=row.device)
    out[label[row] == label[col]] = 1.
    out = scatter(out, col, 0, dim_size=label.size(0), reduce='mean')
    return out

def node_level_struc_ratio(edge_index, label, tolerance):
    """
    #计算每个节点的 **结构性比例**。具体来说，这个函数会忽略某些“缺失标签”（用值 `-1` 表示）的节点，并针对剩余部分重新构建子图，再调用 `struc_ratio` 方法计算每个节点的结构性比例。

    :param edge_index:
    :param label:
    :param torr:
    :return:
    """
    non_missing = torch.where(label != -1)[0]
    new_edge_index = subgraph(non_missing, edge_index, relabel_nodes=True, num_nodes=label.shape[0])[0]
    new_label = label[non_missing]
    temp = struc_ratio(degree(new_edge_index[0], num_nodes=non_missing.shape[0]), new_label, tolerance)
    s_weight = torch.zeros(label.shape[0])
    s_weight[non_missing] = temp
    return s_weight.unsqueeze(1)


def struc_ratio(properties, label, tolerance=0):
    properties = properties.cpu()
    label = label.cpu()
    mask = ~torch.eye(properties.size(0)).bool()    # ~ 按位取反，创建一个掩码矩阵，去掉对角线元素（对角线位置表示节点自身与自身的比较）。
    properties_same = torch.abs(properties.unsqueeze(1) - properties.unsqueeze(0)) <= tolerance     # 计算属性的相似性：如果两个节点的属性之差小于等于容差，则认为它们相似。
    # properties_same = torch.eq(properties.unsqueeze(1), properties.unsqueeze(0))
    properties_same = properties_same[mask].reshape(properties_same.size(0), properties_same.size(0) - 1) # 去掉对角线元素后，重塑矩阵形状。


    label_same = torch.eq(label.unsqueeze(1), label.unsqueeze(0))
    label_same = label_same[mask].reshape(label_same.size(0), label_same.size(0) - 1)

    properties_label_same = properties_same & label_same
    same_count = torch.sum(properties_label_same.float(), dim=-1) # dim=-1 表示沿每一行求和
    total_count = torch.sum(properties_same.float(), dim=-1)

    ratios = same_count / total_count
    return ratios





def struc_ratio_by_core(edge_index, label, tolerance=0):
    """
    基于节点核心数（core number）计算结构相似性比例。

    参数：
        edge_index: 图的边索引，形状为[2, num_edges]。
        label: 节点的标签，形状为[num_nodes]。
        tolerance: 容差值，用于判断核心数是否相似。

    返回：
        ratios: 每个节点的结构相似性比例，形状为[num_nodes]。
    """
    # 转换 edge_index 为 NetworkX 图
    graph = nx.Graph()
    graph.add_edges_from(edge_index.T.tolist())

    # 计算节点的核心数
    core_numbers = nx.core_number(graph)
    core_array = torch.tensor([core_numbers[node] for node in range(len(label))])  # 转换为张量

    # 和原始 struc_ratio 函数的逻辑一致进行相似性计算
    core_array = core_array.cpu()
    label = label.cpu()
    mask = ~torch.eye(core_array.size(0)).bool()  # 创建掩码矩阵，移除对角线项

    # 判断核心数相似性（属性）
    core_same = torch.abs(core_array.unsqueeze(1) - core_array.unsqueeze(0)) <= tolerance
    core_same = core_same[mask].reshape(core_same.size(0), core_same.size(0) - 1)  # 去掉对角线元素并重塑形状

    # 判断标签相同（意义）
    label_same = torch.eq(label.unsqueeze(1), label.unsqueeze(0))
    label_same = label_same[mask].reshape(label_same.size(0), label_same.size(0) - 1)

    # 属性和标签均相同
    core_label_same = core_same & label_same
    same_count = torch.sum(core_label_same.float(), dim=-1)  # 核心数相似且标签相同的节点数
    total_count = torch.sum(core_same.float(), dim=-1)  # 核心数相似的总节点数

    # 结构性比例
    ratios = same_count / total_count
    return ratios
