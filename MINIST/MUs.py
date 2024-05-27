# 导入所需的库
import numpy as np
import torch
import visdom
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat


# 定义神经网络模型（LeNet）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 360, 5, 1)
        self.conv2 = nn.Conv2d(360, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


# 训练模型
def train(model_train, opt_train, loss_train, uav_index, mu_index):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        opt_train.zero_grad()
        output = model_train(data)
        loss = loss_train(output, target)
        loss.backward()
        opt_train.step()
        total_loss += loss.item()
    # 输出每个epoch的平均损失
    viz.line(X=[epoch + 1], Y=[total_loss / len(train_loader)],
             win='The loss of ' + str(mu_index + 1) + ' MU in RW_UAV' + str(uav_index),
             opts={
                 'title': 'The loss of ' + ' MU' + str(mu_index + 1) +  ' in RW_UAV' + str(uav_index)},
             update='append')
    # print('Epoch: {} 平均损失: {:.6f}'.format(epoch + 1, total_loss / len(train_loader)))
    return total_loss / len(train_loader)


if __name__ == '__main__':
    # 进行模型训练
    viz = visdom.Visdom()
    viz.close()
    num_rw_uav = 3
    num_mu = 10
    num_epi = 400
    # 创建神经网络模型实例
    model = dict()
    optimizer = dict()
    loss_fun = dict()
    num_data = np.zeros((num_rw_uav, num_mu))
    for index_model in range(num_rw_uav):
        model_mu = dict()
        optimizer_mu = dict()
        loss_fun_mu = dict()
        for index_mu in range(num_mu):
            model_mu["mu" + str(index_mu)] = Net()
            optimizer_mu["mu" + str(index_mu)] = optim.SGD(model_mu["mu" + str(index_mu)].parameters(), lr=0.001, momentum=0.5)
            loss_fun_mu["mu" + str(index_mu)] = nn.CrossEntropyLoss()
        model["uav" + str(index_model)] = model_mu
        optimizer["uav" + str(index_model)] = optimizer_mu
        loss_fun["uav" + str(index_model)] = loss_fun_mu
    # 设置随机种子，以确保结果的可重复性
    torch.manual_seed(0)
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化图像
    ])
    # 加载MNIST数据集
    data_set_all = datasets.MNIST('data', train=True, download=True, transform=transform)  # 训练数据集
    # 定义优化器和损失函数
    loss_res = np.zeros((num_rw_uav, num_mu, num_epi))
    for index_uav in range(num_rw_uav):
        for index_mu in range(num_mu):
            num_data_select = np.random.randint(100, 1000)  # 您可以根据需要修改 n 的值
            num_data[index_uav][index_mu] = num_data_select
            data_select = RandomSampler(data_set_all, replacement=False, num_samples=num_data_select)
            # 自定义训练样本数量
            bath_size = 50  # 您可以根据需要调整
            # 创建数据加载器
            train_loader = DataLoader(data_set_all, batch_size=bath_size, sampler=data_select)
            for epoch in range(num_epi):
                loss_res[index_uav][index_mu][epoch] = train(model["uav" + str(index_model)]["mu" + str(index_mu)], optimizer["uav" + str(index_model)]["mu" + str(index_mu)], loss_fun["uav" + str(index_model)]["mu" + str(index_mu)], index_uav, index_mu)
    mat_dict = {'loss': loss_res}

    # 保存为MAT文件
    savemat('loss.mat', mat_dict)

    mat_dict = {'num_data': num_data}
    savemat('num_data.mat', mat_dict)
