"""
Accelerate是由Hugging Face开发的一个库，用于简化和加速在多种硬件设备（如多GPU、TPU）上进行深度学习训练的过程。
它旨在使用户能够更容易地将代码从单个设备扩展到多个设备，而无需进行大量的代码改动。以下是Accelerator库的一些主要功能和用途：
    主要功能：
    1、多设备支持：
        Accelerator可以自动处理在多个GPU或TPU上的分布式训练，使得代码更具可移植性和可扩展性。
    2、简化数据并行：
        通过自动处理数据并行和梯度累积，Accelerator简化了在多个设备上的数据同步和分发。
    3、优化器和调度器支持：
        支持主流的优化器和学习率调度器，并能够在多个设备上高效运行。
    4、混合精度训练：
        支持混合精度训练（FP16），这有助于减少内存使用并加速训练过程。
    5、兼容性好：
        与主流的深度学习框架（如PyTorch）兼容，可以无缝集成到现有的代码库中。
"""

#################################进行多GPU训练########################################
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch import nn, optim

# 初始化Accelerator
accelerator = Accelerator()

# 准备数据集
dataset = CIFAR10(root='./data', download=True, transform=ToTensor())
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 定义模型、损失函数和优化器
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(32*32*3, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 使用Accelerator包装模型、优化器和数据加载器
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

# 训练循环
for epoch in range(10):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
    
    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
    
    val_loss /= len(val_loader.dataset)
    correct /= len(val_loader.dataset)
    print(f'Epoch {epoch + 1}, Validation loss: {val_loss:.4f}, Accuracy: {correct:.4f}')
