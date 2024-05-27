"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Train the model by the FedAvg
"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import visdom

from Client.Client import Client
from Server.Server import Server
from Utils.Parameters import parameters
from Utils.ProcessData import ProcessData
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


def ml_hfl(num_rw_uav, num_sub_carrier_each_uav, num_data_uav_mu, data_loader):
    # 联邦学习过程 #
    # 1. 初始化 参数 Server Client 在线打印窗口
    viz = None
    if par.visdom:
        viz = visdom.Visdom()
        viz.close()
    # data_holder = ProcessData(path_data_set, par.ratio_train, par.num_client)
    worker_mu = Client(par)
    server_rw_uav = dict()
    for index_uav in range(num_rw_uav):
        server_rw_uav["rw-uav" + str(index_uav + 1)] = Server(par)
    server_fw_uav = Server(par)
    # 2. 进行多次的FL
    loss_dis_train = np.zeros((num_rw_uav, par.epochs))
    loss_rw_uav = np.zeros((num_rw_uav, par.epochs))
    loss_fw_uav = np.zeros(par.epochs)

    for index_fl in range(par.epochs):
        print("-------------开始第 " + str(index_fl + 1) + " 个epoch的训练-------------\n")
        # 2.1 本地模型获取最新的全局模型参数
        w_worker_mu_newest = dict()
        for index_uav in range(par.num_rw_uav):
            w_worker = dict()
            for index_mu in range(par.num_sub_carrier_each_uav):
                w_worker["mu" + str(index_mu + 1)] = server_rw_uav[
                    "rw-uav" + str(index_uav + 1)].get_parameters_global_model()
            w_worker_mu_newest["rw-uav" + str(index_uav + 1)] = w_worker
        # 2.2 针对每个Client并行执行1-5个Echos，并进行梯度下降，更新参数，并返回各自最新的参数
        loss_value = np.zeros((par.num_rw_uav, par.num_sub_carrier_each_uav))
        for index_uav in range(par.num_rw_uav):
            for index_mu in range(par.num_sub_carrier_each_uav):
                # 使用copy.deepcopy创建了一个同参数的模型副本 copy.deepcopy是深度拷贝 与原始对象相互独立
                w_worker_mu_newest["rw-uav" + str(index_uav + 1)]["mu" + str(index_mu + 1)], loss_value[index_uav][
                    index_mu] = worker_mu.train_local(
                    data_loader["rw-uav" + str(index_uav + 1)]["mu" + str(index_mu + 1)],
                    copy.deepcopy(server_rw_uav["rw-uav" + str(index_uav + 1)].global_model), index_fl, index_uav, index_mu)
            loss_dis_train[index_uav, index_fl] = np.sum(loss_value[index_uav]) / len(loss_value[index_uav])
            radio = num_data_uav_mu[index_uav] / np.sum(num_data_uav_mu[index_uav])
            loss_rw_uav[index_uav, index_fl] = np.sum(radio * loss_value[index_uav])
        radio1 = np.sum(num_data_uav_mu, 1) / np.sum(num_data_uav_mu)
        for index_uav in range(par.num_rw_uav):
            loss_fw_uav[index_fl] += radio1[index_uav] * loss_rw_uav[index_uav][index_fl]
        if par.visdom:
            for index_uav in range(par.num_rw_uav):
                viz.line(X=[index_fl + 1], Y=[loss_dis_train[index_uav][index_fl]],
                         win='MUs loss of the RW-UAV' + str(index_uav + 1),
                         opts={
                             'title': 'The average training Loss of the MUs served by the RW-UAV' + str(index_uav + 1)},
                         update='append')
                viz.line(X=[index_fl + 1], Y=[loss_rw_uav[index_uav][index_fl]],
                         win='loss of the RW-UAV' + str(index_uav + 1),
                         opts={
                             'title': 'The training Loss of the RW-UAV' + str(index_uav + 1)},
                         update='append')
                viz.line(X=[index_fl + 1], Y=[loss_fw_uav[index_fl]],
                         win='loss of the FW-UAV',
                         opts={
                             'title': 'The training Loss of the FW-UAV'},
                         update='append')
        # 2.3 根据本地参数进行全局模型参数的聚合 聚合方式很多 这里采用平均
        w_global_newest_rw = dict()
        for index_uav in range(par.num_rw_uav):
            w_local = w_worker_mu_newest["rw-uav" + str(index_uav + 1)]
            w_global_newest = server_rw_uav["rw-uav" + str(index_uav + 1)].calculate_newest_parameters_global_model(
                w_local)
            w_global_newest_rw["rw-uav" + str(index_uav + 1)] = w_global_newest
            # 2.4 更新全局模型参数
            server_rw_uav["rw-uav" + str(index_uav + 1)].load_parameters_to_global_model(w_global_newest)
        # 2.4 第二次全局模型聚合
        w_global_newest_fw = server_fw_uav.calculate_newest_parameters_global_model_fw(w_global_newest_rw)
        server_fw_uav.load_parameters_to_global_model(w_global_newest_fw)
        # 3. 在当前的模型和数据集下，分别计算RW-UAVs和FW-UAV模型的损失函数
        print("-------------第 " + str(index_fl + 1) + " 个epoch训练结束 " + "-------------\n")
    # # 3. 画图 LOSS
    # plt.figure()
    # plt.plot(loss_dis)
    # plt.show()
    # # 4.保存模型参数
    # for index_uav in range(par.num_rw_uav):
    #     torch.save(server_rw_uav.global_model["rw-uav" + str(index_uav + 1)].state_dict(), 'model.pth')


if __name__ == "__main__":
    # 准备每次FL的数据集

    par = parameters()
    num_data_uav_mu = np.zeros((par.num_rw_uav, par.num_sub_carrier_each_uav))
    # 设置随机种子，以确保结果的可重复性
    torch.manual_seed(0)
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化图像
    ])
    # 加载MNIST数据集
    data_set_all = datasets.MNIST('data', train=True, download=True, transform=transform)  # 训练数据集
    data_loader = dict()
    for index_uav_ in range(par.num_rw_uav):
        data_mu = dict()
        for index_mu_ in range(par.num_sub_carrier_each_uav):
            num_data = np.random.randint(par.num_fe_min, par.num_fe_max)
            num_data_uav_mu[index_uav_][index_mu_] = num_data
            data_select = RandomSampler(data_set_all, replacement=False, num_samples=num_data)
            bath_size = 10  # 您可以根据需要调整
            train_loader = DataLoader(data_set_all, batch_size=bath_size, sampler=data_select)
            data_mu["mu" + str(index_mu_ + 1)] = train_loader
        data_loader["rw-uav" + str(index_uav_ + 1)] = data_mu
    ml_hfl(par.num_rw_uav, par.num_sub_carrier_each_uav, num_data_uav_mu, data_loader)
