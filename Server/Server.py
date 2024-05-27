"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Defining the function of the server in the FL structure.
"""
import copy
import sys

import numpy as np
import torch
import torch.nn as nn

from Models.Models import LinearRegression, Net


class Server(nn.Module):
    def __init__(self, par):
        super(Server, self).__init__()
        self.par = par
        self.model_name = self.par.model_name  # 训练的模型的名称
        self.dim_input = self.par.dim_input  # 输入的维度 即特征维度
        self.dim_output = self.par.dim_output  # 输出的维度 即标签维度
        self.global_model = self.build_model()  # 构建全局模型
        self.idx_client_selected = self.select_clients_join_fl()
        self.num_client_selected = len(self.idx_client_selected)

    '''
    # 输入：无
    # 输出：全局模型
    # 功能：根据模型名称构建全局模型，未识别输出None
    '''

    def build_model(self):
        model = None
        if self.model_name == "LR":
            model = LinearRegression(self.dim_input, self.dim_output)
        elif self.model_name == "MINIST":
            model = Net()
        return model

    '''
    # 输入：无
    # 输出：被选择参与联邦学习的Client的ID
    # 功能：确定参与联邦学习的Client
    '''

    def select_clients_join_fl(self):
        # 所有的Client均被训中
        if self.par.flag:
            idx_selected_clients = np.array(np.linspace(0, self.par.num_client - 1, self.par.num_client), dtype=int)
        # 按比例选中Client，比例为self.args.frac
        else:
            num_selected = int(self.par.num_client * self.par.frac)
            idx_selected_clients = np.random.choice(np.array(np.linspace(0, self.par.num_client - 1,
                                                                         self.par.num_client), dtype=int),
                                                    num_selected)
        return idx_selected_clients

    '''
    # 输入：无
    # 输出：全局模型的参数，字典
    # 功能：获取全局模型参数
    '''

    def get_parameters_global_model(self):
        return self.global_model.state_dict()

    '''
    # 输入：所有的本地参数
    # 输出：计算后的全局模型参数
    # 功能：计算全局模型参数
    '''

    @staticmethod
    def calculate_newest_parameters_global_model(w_local):
        # 深度复制第一个Client的网络参数
        w_avg = copy.deepcopy(w_local["mu1"])
        # keys是网络参数的键，因为是字典，例如network.0.weight表示第一层网络的权重
        for k in w_avg.keys():
            for i in range(1, len(w_local)):
                w_avg[k] += w_local["mu" + str(i + 1)][k]
            w_avg[k] = torch.div(w_avg[k], len(w_local))
        return w_avg

    @staticmethod
    def calculate_newest_parameters_global_model_fw(w_local):
        # 深度复制第一个Client的网络参数
        w_avg = copy.deepcopy(w_local["rw-uav1"])
        # keys是网络参数的键，因为是字典，例如network.0.weight表示第一层网络的权重
        for k in w_avg.keys():
            for i in range(1, len(w_local)):
                w_avg[k] += w_local["rw-uav" + str(i + 1)][k]
            w_avg[k] = torch.div(w_avg[k], len(w_local))
        return w_avg

    '''
    # 输入：最新的全局模型参数
    # 输出：无
    # 功能：更新全局模型参数
    '''

    def load_parameters_to_global_model(self, newest_par):
        self.global_model.load_state_dict(newest_par)

    '''
    # 输入：通信的速率 bit/s 参数的总大小 bit
    # 输出：广播参数需要花费的时间 s
    # 功能：计算广播参数的时间
    '''

    @staticmethod
    def calculate_time_broadcast_parameters(rate, size):
        return rate / size


