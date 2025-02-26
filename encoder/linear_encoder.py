import torch
import torch.nn as nn


"""
该程序定义了一个 LinearEncoder 类，用于实现线性编码器或多层感知机（MLP）编码器
它根据输入通道数、隐藏通道数、变量名称、模型类型、层数和归一化类型来构建编码器。
编码器可以对输入数据进行归一化处理，并通过线性层或MLP层进行特征提取，最终将结果存储到 batch 对象中。
"""

class LinearEncoder(torch.nn.Module):
    kernel_type = None
    def __init__(
            self,
            in_channel,
            hidden_channel,
            var_name,
            model_type='linear',
            num_layers=None,
            norm_type=None,
    ):
        super().__init__()
        self.var_name = var_name
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(in_channel)
        else:
            self.raw_norm = None

        activation = nn.ReLU
        if model_type == 'mlp':
            layers = []
            if num_layers == 1:
                layers.append(nn.Linear(in_channel, hidden_channel))
                layers.append(activation())
            else:
                layers.append(nn.Linear(in_channel, 2 * hidden_channel))
                layers.append(activation())
                for _ in range(num_layers - 2):
                    layers.append(nn.Linear(2 * hidden_channel, 2 * hidden_channel))
                    layers.append(activation())
                layers.append(nn.Linear(2 * hidden_channel, hidden_channel))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)

        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(in_channel, hidden_channel)

    def forward(self, batch):
        var = getattr(batch, self.var_name)
        if self.raw_norm:
            var = self.raw_norm(var)
        var = self.pe_encoder(var)


        """
        如果 self.var_name 为 'x'，则将结果直接赋值给 batch.x。
        否则，将 batch.x 与 var 在维度 1 上进行拼接，并将结果赋值给 batch.x。
        """
        if self.var_name == 'x':
            batch.x = var
        else:
            h = batch.x
            batch.x = torch.cat((h, var), 1)

        return batch

