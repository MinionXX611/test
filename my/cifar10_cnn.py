# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms


# 定义超参数
batch_size = 64
learning_rate = 0.0007
num_epochs = 100



# 定义数据预处理方式
# 普通的数据预处理方式
'''
transform = transforms.Compose([
    transforms.ToTensor(),])
'''
# 数据增强的数据预处理方式
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),     # 随机水平翻转
    transforms.RandomRotation(15),             # 随机旋转 ±15度
    transforms.RandomCrop(32, padding=4),      # 随机裁剪并填充
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 色彩扰动
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),  # 随机擦除
    transforms.ToTensor(),                     # 转为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 数据归一化
])



# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(nn.Module):
    '''
    定义卷积神经网络,3个卷积层,2个全连接层
    '''
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 输入通道数为3，输出通道数为32，卷积核大小为3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 输入通道数为32，输出通道数为64，卷积核大小为3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 输入通道数为64，输出通道数为128，卷积核大小为3x3
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 输入通道数为128，输出通道数为128，卷积核大小为3x3
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 输入通道数为128，输出通道数为128，卷积核大小为3x3
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # 输入通道数为128，输出通道数为128，卷积核大小为3x3
        self.fc1 = nn.Linear(128 * 2 * 2, 256)          # 全连接层1
        self.fc2 = nn.Linear(256, 10)               # 全连接层2，输出10个类别
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 最大池化层

    def forward(self,x):
        # 前向传播
        x=self.conv1(x) # 卷积层1
        x=F.relu(x)     # ReLU激活函数
        x=self.pool(x) # 最大池化层

        x=self.conv2(x) # 卷积层2
        x=F.relu(x)     # ReLU激活函数
        x=self.pool(x) # 最大池化层

        x=self.conv3(x) # 卷积层3
        x=F.relu(x)     # ReLU激活函数
        x=self.pool(x) # 最大池化层

        x=self.conv4(x) # 卷积层4
        x=F.relu(x)  # ReLU激活函数
        x=self.pool(x) # 最大池化层

        x=self.conv5(x) # 卷积层5
        x=F.relu(x)    # ReLU激活函数

        x=self.conv6(x) # 卷积层6
        x=F.relu(x)    # ReLU激活函数

        x=self.fc1(x.view(-1, 128 * 2 * 2)) # 展平并输入全连接层1
        x=F.relu(x)     # ReLU激活函数
        x=nn.Dropout(0.5)(x) # Dropout层

        x=self.fc2(x) # 输入全连接层2
        x=F.log_softmax(x, dim=1) # log_softmax 激活
        return x



# 实例化模型
model = Net()

# 检查是否支持 MLU（如寒武纪芯片），否则使用 GPU 或 CPU
use_mlu = False
try:
    use_mlu = torch.mlu.is_available()  # 检查 MLU 是否可用
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')  # 使用 MLU 设备
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # 使用 GPU 设备
    else:
        device = torch.device('cpu')  # 使用 CPU 设备

# 将模型移动到指定设备（MLU/GPU/CPU）
model = model.to(device)

# 定义损失函数 优化器 学习率调度器
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)  # 优化器（需要定义，例如 optim.Adam(model.parameters(), lr=learning_rate)）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)  # 每 30 个 epoch 学习率减小为原来的 0.1

# 训练模型
for epoch in range(num_epochs):  # 遍历每个 epoch
    # 设置模型为训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):  # 遍历训练数据
        images = images.to(device)  # 将输入数据移动到指定设备
        labels = labels.to(device)  # 将标签移动到指定设备

        # 前向传播
        outputs = model(images)  # 通过模型计算输出
        loss = criterion(outputs, labels)  # 计算损失

        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        # 计算当前批次的准确率
        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 每 100 个批次打印一次训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))

    # 测试模型
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        correct = 0  # 记录正确预测的样本数
        total = 0  # 记录总样本数
        for images, labels in test_loader:  # 遍历测试数据
            images = images.to(device)  # 将输入数据移动到指定设备
            labels = labels.to(device)  # 将标签移动到指定设备

            outputs = model(images)  # 通过模型计算输出
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 累加样本总数
            correct += (predicted == labels).sum().item()  # 累加正确预测的样本数

        # 打印测试集的准确率
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    scheduler.step()  # 更新学习率