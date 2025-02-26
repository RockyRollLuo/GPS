import argparse
import copy
import os
from functional.config import load_config
from torch_geometric.transforms import ToSparseTensor  # PyTorch Geometric模块，将图转换为稀疏张量表示
from torch_geometric.loader import DataLoader  # PyTorch Geometric模块，用于批量加载图数据
import torch
import torch.nn.functional as F  # 常用的神经网络函数模块，如激活函数和损失函数
import torch.optim as optim  # PyTorch优化器模块
import numpy as np  # 常用的数值计算库
from sklearn.metrics import roc_auc_score, accuracy_score  # 导入SKLearn的评价指标，计算AUC和准确率

from functional.dataset import LoadSubgraph, load_data  # 自定义模块，用于加载数据集
from functional.seed import set_seed  # 设置随机种子以保证实验的可重复性
from functional.transform import transform_in_memory  # 执行数据转换的自定义函数
from model.atk_model import AttrAtk  # 自定义GNN模型，用于攻击实验

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 设置CUDA的工作空间大小配置，优化性能
torch.use_deterministic_algorithms(True)  # 强制使用确定性算法以确保复现实验结果


# 定义训练函数
def train(model, device, optimizer, train_loader, whole_graph, train_idx):
    model.train()  # 设置模型为训练模式
    train_loss_accum = 0  # 初始化训练损失累计值
    for step, (batch_s, batch_idx) in enumerate(train_loader):  # 遍历每个训练数据批次
        glo_batch_idx = train_idx[batch_idx]  # 将局部索引转换为全局索引
        batch_s = ToSparseTensor()(batch_s.to(device))  # 将训练图转换为稀疏张量形式并传输到设备
        batch_p = ToSparseTensor()(whole_graph)  # 将整体图转换为稀疏张量形式

        optimizer.zero_grad()  # 清空优化器的累积梯度
        x = model.forward(batch_p, batch_s, glo_batch_idx)  # 前向传播，获取输出预测值
        batch_loss = F.cross_entropy(x, batch_s.y)  # 计算交叉熵损失
        batch_loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 使用优化器更新模型参数

        train_loss_accum += batch_loss.item()  # 累加当前批次的损失
    train_loss = train_loss_accum / (step + 1)  # 计算平均训练损失
    return train_loss  # 返回该轮训练的平均损失


# 定义测试函数
@torch.no_grad()  # 禁用梯度计算以加快推断速度和节省内存
def test(model, device, train_loader, test_loader, num_classes, whole_graph, glo_idx):
    model.eval()  # 设置模型为评估模式
    train_idx = glo_idx[0]  # 获取训练集的全局索引
    test_idx = glo_idx[1]  # 获取测试集的全局索引
    y_pred = torch.ones(whole_graph.num_nodes, dtype=torch.long) * -1  # 初始化预测标签为-1
    y_pred = y_pred.to(device)  # 将预测张量传递到设备

    # 以下评估训练集表现
    train_res_accum = 0  # 初始化训练集累计评估结果
    for step, (batch_s, batch_idx) in enumerate(train_loader):  # 遍历训练数据
        glo_batch_idx = train_idx[batch_idx]  # 获取该批次对应的全局索引
        batch_s = ToSparseTensor()(batch_s.to(device))  # 转换为稀疏张量
        batch_p = ToSparseTensor()(whole_graph)  # 获取稀疏图表示
        x = model.forward(batch_p, batch_s, glo_batch_idx)  # 模型前向传播
        pred = F.softmax(x, dim=-1)  # 计算Softmax概率分布
        true = batch_s.y  # 获取真实标签

        if num_classes == 2:  # 如果是二分类任务
            train_res = roc_auc_score(true.cpu(), pred[:, 1].cpu())  # 计算AUC分数
        else:  # 如果是多分类任务
            pred = torch.argmax(pred, dim=-1)  # 取最大概率的类别作为预测值
            train_res = accuracy_score(true.cpu(), pred.cpu())  # 计算准确率
        train_res_accum += train_res  # 累加训练集的评估结果
        y_pred[glo_batch_idx] = torch.argmax(x, dim=-1)  # 更新预测结果
    train_res = train_res_accum / (step + 1)  # 计算平均训练集指标

    # 以下评估测试集表现
    test_res_accum = 0  # 初始化测试集累计评估结果
    for step, (batch_s, batch_idx) in enumerate(test_loader):  # 遍历测试数据
        glo_batch_idx = test_idx[batch_idx]  # 获取该批次的全局索引
        batch_s = ToSparseTensor()(batch_s.to(device))  # 转换为稀疏张量
        batch_p = ToSparseTensor()(whole_graph)  # 获取稀疏图表示
        x = model.forward(batch_p, batch_s, glo_batch_idx)  # 模型前向传播
        pred = F.softmax(x, dim=-1)  # 计算Softmax概率分布
        true = batch_s.y  # 获取真实标签
        if num_classes == 2:  # 如果是二分类任务
            test_res = roc_auc_score(true.cpu(), pred[:, 1].cpu())  # 计算AUC分数
        else:  # 如果是多分类任务
            pred = torch.argmax(pred, dim=-1)  # 取最大概率的类别作为预测值
            test_res = accuracy_score(true.cpu(), pred.cpu())  # 计算准确率
        test_res_accum += test_res  # 累加测试集的评估结果
        y_pred[glo_batch_idx] = torch.argmax(x, dim=-1)  # 更新测试集的预测结果
    test_res = test_res_accum / (step + 1)  # 计算平均测试集指标

    return train_res, test_res, y_pred  # 返回训练集、测试集表现和预测标签


# 主实验入口
def main(exp_num, opt):
    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")  # 设置设备

    sub_graph = LoadSubgraph(root=opt.root, data_name=opt.dataset, sens_attr=opt.sens_attr, hops=opt.hops,
                             device=opt.device)  # 加载子图
    whole_graph = load_data(data_name=opt.dataset, target=opt.sens_attr, train_ratio=opt.train_ratio,
                            path=opt.root + '/raw/').to(device)  # 加载完整图数据
    const_edge_index = whole_graph.edge_index  # 获取静态边索引

    # print('=====load dataset=====')  # 输出加载数据集信息
    # print(sub_graph, whole_graph)  # 打印子图和完整图的基本信息

    transform_in_memory(sub_graph=sub_graph, whole_graph=copy.deepcopy(whole_graph), opt=opt)  # 执行数据预处理操作


    train_idx, test_idx = whole_graph.train_mask, whole_graph.test_mask  # 获取训练集和测试集掩码
    train_idx = torch.where(train_idx)[0]  # 提取训练集索引
    test_idx = torch.where(test_idx)[0]  # 提取测试集索引

    train_loader = DataLoader(sub_graph[train_idx], batch_size=opt.batch_size, num_workers=opt.num_workers,
                              shuffle=True)  # 定义训练数据加载器
    test_loader = DataLoader(sub_graph[test_idx], batch_size=opt.batch_size, num_workers=opt.num_workers,
                             shuffle=True)  # 定义测试数据加载器

    # 设置模型及其超参数
    opt.in_channels['x_in'] = sub_graph.num_features  # 设置模型输入特征维度
    model = AttrAtk(
        gnn=(opt.p_encoder, opt.s_encoder),  # 自定义GNN编码器
        in_channels=opt.in_channels,
        hidden_channels=opt.hidden_channels,
        gnn_layers=opt.num_layers,
        num_classes=sub_graph.num_classes,
        dropout=opt.dropout,
        pe_types=opt.pe_types,
        num_nodes=whole_graph.num_nodes,
        torr=opt.torr,
    ).to(device)  # 实例化模型并传输至设备

    model.p_weight, model.s_weight = model.p_weight.to(device), model.s_weight.to(device)  # 将模型附加权重传输至设备
    # print(model)  # 输出模型结构信息

    # 配置优化器
    optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)  # AdamW优化器
    step_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.scheduler)  # 指数学习率调度器
    # print(optimizer)  # 打印优化器信息

    for epoch in range(1, opt.num_epochs + 1):  # 遍历每一个epoch
        train_loss = train(model, device, optimizer, train_loader, whole_graph, train_idx)  # 进行模型训练
        step_scheduler.step()  # 更新学习率

        if epoch % 10 == 1 or epoch == 1:  # 每10轮或第1轮进行模型评估


            train_res, test_res, y_pred = test(model, device, train_loader, test_loader, sub_graph.num_classes,
                                               whole_graph, (train_idx, test_idx))  # 评估训练和测试表现
            print(
                f'exp {exp_num}, epoch {epoch:03d}, train loss {train_loss:.4f}, train res {train_res:.4f}, test res {test_res:.4f}'
                # 输出训练和测试结果
            )

        if epoch > opt.init and epoch % opt.upd_gap == 1:  # 条件更新模型权重
            model.update_weights(const_edge_index, y_pred)  # 使用预测结果更新模型权重,根据pro和struc的比例

    return test_res  # 返回最终测试结果





# 脚本入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--device', type=int, default=3)  # 设备编号
    parser.add_argument('--num_layers', type=int, default=2)  # 模型层数
    parser.add_argument('--train_ratio', type=float, default=0.1)  # 训练集比例
    parser.add_argument('--p_encoder', type=str, default="SAGE")  # structure 编码器类型
    parser.add_argument('--s_encoder', type=str, default="SAGE")  # proximity 编码器类型
    parser.add_argument('--num_workers', type=int, default=1)  # 数据加载的线程数
    parser.add_argument('--dataset', type=str, default='nba')  # 数据集名称
    parser.add_argument('--sens_attr', type=str, default='country')  # 敏感属性名称
    args = parser.parse_args()  # 从命令行解析参数

    param = load_config(source=f'{args.dataset}_{args.sens_attr}.json')  # 从配置文件加载超参数
    for key, value in param.items(): setattr(args, key, value)  # 设置参数中的键值对到命令行参数变量中

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 根据GPU可用性设置设备

    print(f'====experiment args:====')
    print(args)  # 输出当前实验参数配置
    set_seed(args.seed)  # 设置随机种子以确保结果复现
    test_l = []  # 初始化测试结果列表

    for i in range(2):  # 重复实验1次以计算平均值和方差
        print(f'====exp:{i}====')
        test_res = main(i, args)  # 运行主实验流程
        test_l.append(test_res)  # 记录结果

    ave, std = np.mean(test_l), np.std(test_l)  # 计算结果的平均值和标准差
    print(f'final result: {ave:.4f}±{std:.4f}')  # 输出实验结果

