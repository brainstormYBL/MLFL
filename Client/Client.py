"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Defining the action of the clients in each epoch.
"""
import numpy as np
import torch
from torch import optim, nn


class Client:
    def __init__(self, args):
        self.args = args

    '''
    # 输入：当前Client本地数据集的特征与标签，本地模型 训练一个Client调用一次，可以采取并行多个同时计算
    # 输出：本地模型经过多次训练后的模型参数 训练过程中的平均Loss
    # 功能：训练本地模型
    '''
    def train_local(self, train_loader, model, index_fl, index_uav, index_mu):
        # num_data = feature.shape[0]
        # loss_fun = self.args.loss_func  # 本地模型训练的损失函数
        # optimizer = torch.optim.Adam(model.parameters(), self.args.lr)  # 本地模型训练的优化器
        # loss_re = []
        # # 本地多次训练 减少通信的场景
        # for index_echo in range(self.args.epochs_local):
        #     data_selected_idx = np.array(np.random.choice(np.linspace(0, num_data - 1, num_data), self.args.batch_size)
        #                                  , dtype=int)
        #     feature_train = feature[data_selected_idx]
        #     label_train = torch.reshape(label[data_selected_idx], (-1, 1))
        #     # for param in model.parameters():
        #     #     print(f"Parameter Data Type: {param.dtype}")
        #     # print(feature_train.dtype)
        #     # 类型转换 转换为与模型参数一致的类型 一般为float32
        #     feature_train = feature_train.to(torch.float32)
        #     label_train = label_train.to(torch.float32)
        #     # 预测
        #     y_pre = model(feature_train)
        #     loss_val = loss_fun(label_train, y_pre)
        #     loss_re.append(loss_val.item())
        #     # print(loss_val.item())
        #     optimizer.zero_grad()
        #     loss_val.backward()
        #     optimizer.step()
        # return model.state_dict(), sum(loss_re) / len(loss_re)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.CrossEntropyLoss()
        # model.train()
        total_loss = 0
        for index_local in range(5):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print('在第{}次联邦学习中，第{}个RW-UAV下第{}个MU在本地训练下的平均损失为: {:.6f}'.format(index_fl + 1, index_uav + 1, index_mu + 1, total_loss / len(train_loader)))
        return model.state_dict(), total_loss / len(train_loader)


    '''
    # 输入：通信的速率 bit/s 参数的总大小 bit
    # 输出：上传参数需要花费的时间 s
    # 功能：计算上传参数的时间
    '''

    @staticmethod
    def calculate_time_to_server_parameters(rate, size):
        return rate / size

