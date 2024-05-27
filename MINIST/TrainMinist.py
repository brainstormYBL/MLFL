# 导入所需的库
import torch
import visdom
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

# 设置随机种子，以确保结果的可重复性
torch.manual_seed(0)

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化图像
])
# 加载MNIST数据集
data_set_all = datasets.MNIST('data', train=True, download=True, transform=transform)  # 训练数据集


num_data_select = 100  # 您可以根据需要修改 n 的值
data_select = RandomSampler(data_set_all, replacement=False, num_samples=num_data_select)

# 自定义训练样本数量
bath_size = 10  # 您可以根据需要调整

# 创建数据加载器
train_loader = DataLoader(data_set_all, batch_size=bath_size, sampler=data_select)


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


# 创建神经网络模型实例
model = Net()

# 定义优化器和损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()


# 训练模型
def train(epoch, viz):
    # model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 输出每个epoch的平均损失
    viz.line(X=[epoch + 1], Y=[total_loss / len(train_loader)],
             win='loss of the MUs',
             opts={
                 'title': 'The training Loss of MUs'},
             update='append')
    print('Epoch: {} 平均损失: {:.6f}'.format(epoch + 1, total_loss / len(train_loader)))


# 进行模型训练
viz = visdom.Visdom()
viz.close()
for epoch in range(1000):
    train(epoch, viz)
