import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


# 构造网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        # conv1输出为(16, 14, 14)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # conv2输出为(32, 7, 7)
        self.output = torch.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        prediction = self.output(x)
        return prediction



dummy_input = torch.rand(13, 1, 28, 28)  # 假设输入13张1*28*28的图片
model = Net()
with SummaryWriter(comment='LeNet') as w:
    w.add_graph(model, (dummy_input,))
