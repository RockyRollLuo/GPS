import argparse
import os

import matplotlib.pyplot as plt
import networkx
import numpy as np  # 常用的数值计算库
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split as sk_split
from torch_geometric.data import Data
from torch_geometric.io.planetoid import index_to_mask
from torch_geometric.utils import to_networkx

from functional.dataset import load_data  # 自定义模块，用于加载数据集

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 设置CUDA的工作空间大小配置，优化性能
torch.use_deterministic_algorithms(True)  # 强制使用确定性算法以确保复现实验结果



# 主实验入口
def main(args):
    data,labels = load_data(data_name=args.dataset, target=args.sens_attr, train_ratio=args.train_ratio,
                            path=args.root + '/raw/')  # 加载完整图数据
    labels=labels.numpy()

    classes, counts = np.unique(labels, return_counts=True)
    netx_graph = to_networkx(data, to_undirected=True)


    print(f'====whole graph info:====') # 输出完整图数据信息
    print(netx_graph)
    print(f'Number of classes: {len(classes)}')
    print(f'Classes and their counts: {dict(zip(classes, counts))}')


    # draw_fraction(netx_graph,labels)

    draw_graph_with_labels(netx_graph, labels)


def draw_graph_with_labels(graph, labels):
    # pos = networkx.kamada_kawai_layout(graph)  # 计算节点的布局

    pos = networkx.nx_pydot.graphviz_layout(graph)

    unique_labels, counts = np.unique(labels, return_counts=True)


    color_map = {label: plt.cm.tab20(i / len(unique_labels)) for i, label in enumerate(unique_labels)}  # 为每个标签分配颜色

    node_colors = [color_map[labels[node]] for node in graph.nodes()]  # 根据标签设置节点颜色

    plt.figure(figsize=(20, 20))
    # networkx.draw(graph, pos, node_color=node_colors, with_labels=False, node_size=300, font_size=10, font_color='black')
    networkx.draw(graph, pos, node_color=node_colors, with_labels=False, edge_color='gray', node_size=80)
    plt.show()







def draw_fraction(graph,lables):
    netx_graph = graph
    y=lables
    # Calculate the fraction of neighbors sharing the same label for each node
    fractions1 = []
    num_ture1=0
    total_same1=0
    for node in netx_graph.nodes():
        neighbors = list(netx_graph.neighbors(node))

        labels=[y[neighbor] for neighbor in neighbors]
        if len(neighbors) > 0:
            predicted_label = max(set(labels), key=labels.count)
            if predicted_label == y[node]:
                num_ture1 += 1

            same_label_count = sum(1 for neighbor in neighbors if y[neighbor] == y[node])
            total_same1+=same_label_count

            fraction = same_label_count / len(neighbors)
            fractions1.append(fraction)
    print()
    print(f'==true rate1=={num_ture1/len(netx_graph.nodes())}')
    print(f'==total_same1=={total_same1}')
    print(f'==total_same1 rate =={total_same1/sum(dict(netx_graph.degree()).values())}')

    # Plot the distribution
    plt.hist(fractions1, bins=100, edgecolor='black')
    plt.axvline(np.median(fractions1), color='r', linestyle='dashed', linewidth=1, label='Median')
    plt.axvline(np.mean(fractions1), color='g', linestyle='dashed', linewidth=1, label='Mean')
    plt.xlabel('Fraction of neighbors sharing the same label')
    plt.ylabel('Number of nodes')
    plt.title('Distribution of the fraction 1-hop neighbr')
    plt.legend()
    plt.show()

    # Calculate the label for nodes sharing the same label for two-hop neighbors
    unknown_labels = []
    fractions2=[]
    num_ture2 = 0
    total_same2=0
    for node in netx_graph.nodes():

        two_hop_neighbors = set()
        for neighbor in netx_graph.neighbors(node):
            two_hop_neighbors.update(netx_graph.neighbors(neighbor))

        two_hop_neighbors.discard(node)

        if two_hop_neighbors:
            labels = [y[neighbor] for neighbor in two_hop_neighbors]
            predicted_label = max(set(labels), key=labels.count)
            if predicted_label == y[node]:
                 num_ture2+=1

            same_label_count=sum(1 for neighbor in two_hop_neighbors if y[neighbor] == y[node])

            total_same2+=same_label_count
            fraction=same_label_count/len(two_hop_neighbors)
            fractions2.append(fraction)

    print()
    print(f'==true rate 2=={num_ture2/len(netx_graph.nodes())}')
    print(f'==total_same2=={total_same2}')
    print(f'==total_same2 rate =={total_same2/sum(dict(netx_graph.degree()).values())}*2')

    # Plot the distribution
    plt.hist(fractions2, bins=100, edgecolor='black')
    plt.axvline(np.median(fractions2), color='r', linestyle='dashed', linewidth=1, label='Median')
    plt.axvline(np.mean(fractions2), color='g', linestyle='dashed', linewidth=1, label='Mean')
    plt.xlabel('Fraction of neighbors sharing the same label')
    plt.ylabel('Number of nodes')
    plt.title('Distribution of the fraction of 1-hop core')
    plt.show()




    # Calculate the label for nodes sharing the same label for the neighbors with same core number
    fractions3 = []
    num_ture3 = 0
    total_same3 = 0

    core_numbers = networkx.core_number(netx_graph)
    for node in netx_graph.nodes():
        neighbors = list(netx_graph.neighbors(node))

        labels = [y[neighbor] for neighbor in neighbors if core_numbers[neighbor] >= core_numbers[node]]
        if len(neighbors) > 0:
            predicted_label = max(set(labels), key=labels.count)
            if predicted_label == y[node]:
                num_ture3 += 1

            same_label_count = sum(1 for neighbor in neighbors if y[neighbor] == y[node])
            total_same3 += same_label_count

            fraction = same_label_count / len(neighbors)
            fractions3.append(fraction)
    print()
    print(f'==true rate3=={num_ture3 / len(netx_graph.nodes())}')
    print(f'==total_same3=={total_same3}')
    print(f'==total_same3 rate =={total_same3 / sum(dict(netx_graph.degree()).values())}')

    # Plot the distribution
    plt.hist(fractions3, bins=100, edgecolor='black')
    plt.axvline(np.median(fractions3), color='r', linestyle='dashed', linewidth=1, label='Median')
    plt.axvline(np.mean(fractions3), color='g', linestyle='dashed', linewidth=1, label='Mean')
    plt.xlabel('Fraction of neighbors sharing the same label')
    plt.ylabel('Number of nodes')
    plt.title('Distribution of the fraction 2-hop neighobr')
    plt.show()



    # Calculate the label for nodes sharing the same label for two-hop neighbors and same core number
    unknown_labels = []
    fractions4=[]
    num_ture4 = 0
    total_same4=0
    for node in netx_graph.nodes():

        two_hop_neighbors = set()
        for neighbor in netx_graph.neighbors(node):
            two_hop_neighbors.update(netx_graph.neighbors(neighbor))

        two_hop_neighbors.discard(node)

        if two_hop_neighbors:
            labels = [y[neighbor] for neighbor in two_hop_neighbors if core_numbers[neighbor] >= core_numbers[node]]
            predicted_label = max(set(labels), key=labels.count)
            if predicted_label == y[node]:
                 num_ture4+=1

            same_label_count=sum(1 for neighbor in two_hop_neighbors if y[neighbor] == y[node])

            total_same4+=same_label_count
            fraction=same_label_count/len(two_hop_neighbors)
            fractions4.append(fraction)

    print()
    print(f'==true rate 4=={num_ture4/len(netx_graph.nodes())}')
    print(f'==total_same4=={total_same4}')
    print(f'==total_same4 rate =={total_same4/sum(dict(netx_graph.degree()).values())}*2')

    # Plot the distribution
    plt.hist(fractions4, bins=100, edgecolor='black')
    plt.axvline(np.median(fractions4), color='r', linestyle='dashed', linewidth=1, label='Median')
    plt.axvline(np.mean(fractions4), color='g', linestyle='dashed', linewidth=1, label='Mean')
    plt.xlabel('Fraction of neighbors sharing the same label')
    plt.ylabel('Number of nodes')
    plt.title('Distribution of the fraction 2-hop core')
    plt.show()

def load_data(data_name, target, train_ratio=0.1, path=None, split=True):
    if data_name.startswith('pokec'):
        data, y = load_pokec(dataset=data_name, target_attr=target, train_ratio=train_ratio, path=path, split=split)
    elif data_name.lower() == 'nba':
        data, y = load_nba(data_name=data_name, target_attr=target, train_ratio=train_ratio, path=path, split=split)
    else:
        raise NotImplementedError
    return data, y

def load_nba(data_name, target_attr, train_ratio, path, split=None):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(data_name)))

    # sensitive attribute
    sens = idx_features_labels[target_attr].values
    sens = torch.LongTensor(sens)

    # train test split
    non_missing = np.where(sens.numpy() >= 0)[0]
    non_missing_sens = sens.numpy()[non_missing]
    res = sk_split(non_missing, non_missing_sens, train_size=train_ratio, stratify=non_missing_sens, random_state=0)
    idx_train, idx_test = res[0], res[1]
    idx_train = index_to_mask(torch.LongTensor(idx_train), size=sens.shape[0])
    idx_test = index_to_mask(torch.LongTensor(idx_test), size=sens.shape[0])

    # feature matrix
    header = list(idx_features_labels.columns)
    for attr in ['user_id', 'country', 'SALARY']: header.remove(attr)
    features = np.array(idx_features_labels[header])
    features = torch.FloatTensor(features)
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    #因为如果直接除以范围，可能会导致输出数据范围为 [-1, 1] 或 [0, 1]。但是，在某些神经网络中（如残差块），需要输入数据范围为 [-1, 1]。因此，通过乘以 2，可以将输出数据范围从 [0, 1] 变为 [-1, 1]。
    features = 2 * (features - min_values).div(max_values - min_values) - 1

    # edge index
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(data_name)), dtype=int)
    edges = torch.tensor(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape).t()

    # num of cls
    classes = set(sens.numpy())
    classes.discard(-1)
    num_classes = len(classes)

    # pyg graph
    data = Data(x=features, edge_index=edges, y=sens, train_mask=idx_train, test_mask=idx_test, num_classes=num_classes)
    data = T.ToUndirected()(data)

    return data,sens


def load_pokec(dataset, target_attr, train_ratio, path, split):
    if dataset == 'pokec_n':
        dataset = 'region_job_2'
    elif dataset == 'pokec_z':
        dataset = 'region_job'
    else:
        raise NotImplementedError
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    label = idx_features_labels['I_am_working_in_field'].values
    non_missing_label = np.where(label != -1)[0]
    # private attribute or downstream label
    if target_attr is not None:
        target = idx_features_labels[target_attr].values
        target = torch.LongTensor(target)
        if target_attr == "AGE":
            young = torch.where((target >= 18) & (target < 25))[0]
            young_adult = torch.where((target >= 25) & (target < 35))[0]
            middle = torch.where((target >= 35) & (target < 50))[0]
            senior = torch.where((target >= 50) & (target <= 80))[0]
            target[:] = -1
            target[young] = 0
            target[young_adult] = 1
            target[middle] = 2
            target[senior] = 3
        elif target_attr == "I_am_working_in_field":
            target[target > 1] = 1

        classes = set(target.numpy())
        classes.discard(-1)
        num_classes = len(classes)
    else:
        target, num_classes = None, None

    # node attribute
    header = list(idx_features_labels.columns)
    for attr in ['user_id', 'I_am_working_in_field', 'AGE', 'gender', 'region', 'completion_percentage']: header.remove(
        attr)
    features = np.array(idx_features_labels[header])
    features = torch.FloatTensor(features)

    # edge index
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)
    # edges = torch.tensor(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape).t()
    edges = torch.tensor(list(map(lambda x: idx_map.get(x, -1), edges_unordered.flatten()))).reshape(
        edges_unordered.shape).t()

    data = Data(x=features, edge_index=edges, y=target, num_classes=num_classes)
    data = T.ToUndirected()(data)
    data = data.subgraph(torch.tensor(non_missing_label))

    # split train and test
    if split:
        non_missing_idx = torch.where(data.y >= 0)[0]
        non_missing_sens = data.y[non_missing_idx]
        res = sk_split(non_missing_idx, non_missing_sens, train_size=train_ratio, stratify=non_missing_sens)
        idx_train = res[0]
        idx_test = res[1]
        idx_train = index_to_mask(torch.LongTensor(idx_train), size=data.num_nodes)
        idx_test = index_to_mask(torch.LongTensor(idx_test), size=data.num_nodes)
        data.train_mask, data.test_mask = idx_train, idx_test
    return data, target



if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--dataset', type=str, default='nba', choices=['nba', 'pokec_n', 'pokec_z'])  # 数据集名称
    parser.add_argument('--sens_attr', type=str, default='country', choices=['country', 'SALARY',  'gender', 'age'])  # 敏感属性名称
    parser.add_argument('--train_ratio', type=float, default=0.1)  # 敏感属性名称
    parser.add_argument('--root', type=str, default='./dataset/nba_sub', choices=['./dataset/nba_sub', './dataset/pokec_sub'])  # 敏感属性名称
    args = parser.parse_args()  # 从命令行解析参数

    # print(f'====experiment args:====')




    main(args)  # 运行主实验流程


