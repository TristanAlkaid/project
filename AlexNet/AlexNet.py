import time
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 使用 GPU 进行计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 准备数据集
train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
alex_net = AlexNet()
alex_net = alex_net.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01  # 1e-2 = 1 x (10)^(2)
optimizer = torch.optim.SGD(params=alex_net.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录册数的次数
epoch = 50  # 记录训练的轮数

# 添加 tensorboard
writer = SummaryWriter("logs_train")

start_time = time.time()
for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))

    # 训练开始
    alex_net.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = alex_net(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            end_time = time.time()
            print("已运行时间：{}".format(end_time - start_time))
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    alex_net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = alex_net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    if total_train_step % 10 == 0:
        torch.save(alex_net, "alex_net_{}.pth".format(i+1))
        # torch.save(alex_net.state_dict(), "alex_net_{}.pth".format(i))
        print("模型已保存")

writer.close()
