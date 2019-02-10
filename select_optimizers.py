"""View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/My Youtube Channel: https://www.youtube.com/user/MorvanZhouDependencies:torch: 0.4matplotlib"""from torch.autograd import Variablefrom torchvision import transformsimport fashion_mnist_data_ready as mnist_loadimport torchimport torch.utils.data as Dataimport torch.nn.functional as Fimport matplotlib.pyplot as pltroot="fashion_mnist/"# 读取数据train_data=mnist_load.MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())test_data=mnist_load.MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())train_loader = mnist_load.DataLoader(dataset=train_data, batch_size=64, shuffle=True)test_loader = mnist_load.DataLoader(dataset=test_data, batch_size=64)LR = 0.01BATCH_SIZE = 64EPOCH = 2# 构造网络class Net(torch.nn.Module):    def __init__(self):        super(Net, self).__init__()        self.conv1 = torch.nn.Sequential(            torch.nn.Conv2d(                in_channels=1,                out_channels=16,                kernel_size=5,                stride=1,                padding=2            ),            torch.nn.ReLU(),            torch.nn.MaxPool2d(kernel_size=2)        )        # conv1输出为(16, 14, 14)        self.conv2 = torch.nn.Sequential(            torch.nn.Conv2d(16, 32, 5, 1, 2),            torch.nn.ReLU(),            torch.nn.MaxPool2d(2)        )        # conv2输出为(32, 7, 7)        self.output = torch.nn.Linear(32 * 7 * 7, 10)    def forward(self, x):        x = self.conv1(x)        x = self.conv2(x)        x = x.view(x.size(0), -1)        prediction = self.output(x)        return predictionmodel = Net()model = model.cuda()print(model)if __name__ == '__main__':    # different nets    net_SGD         = Net()    net_Momentum    = Net()    net_RMSprop     = Net()    net_Adam        = Net()    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]    net_SGD = net_SGD.cuda()    net_Momentum = net_Momentum.cuda()    net_RMSprop = net_RMSprop.cuda()    net_Adam = net_Adam.cuda()    # different optimizers    opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)    opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)    opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)    opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]    loss_func = torch.nn.CrossEntropyLoss()    losses_his = [[], [], [], []]   # record loss    # training    for epoch in range(EPOCH):        print('Epoch: ', epoch)        for step, (batch_x, batch_y) in enumerate(train_loader):            for net, opt, l_his in zip(nets, optimizers, losses_his):                batch_x = batch_x.cuda()                batch_y = batch_y.cuda()                batch_x, batch_y = Variable(batch_x), Variable(batch_y)                output = net(batch_x)                loss = loss_func(output, batch_y)                opt.zero_grad()                loss.backward()                opt.step()                l_his.append(loss.data.cpu().numpy())    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']    for i, l_his in enumerate(losses_his):        plt.plot(l_his, label=labels[i])    plt.legend(loc='best')    plt.xlabel('Steps')    plt.ylabel('Loss')    plt.ylim((0, 2.5))    plt.show()