import torch
import time
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import fashion_mnist_data_ready as mnist_load

root="fashion_mnist/"


LR = 0.00001
EPOCH = 50
BATCH_SIZE = 128

# 读取数据
train_data=mnist_load.MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())
test_data=mnist_load.MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
train_loader = mnist_load.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = mnist_load.DataLoader(dataset=test_data, batch_size=BATCH_SIZE)



# 构造网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, 1, 2),
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



model = Net()
model = model.cuda()
print(model)

if __name__ == '__main__':

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,betas=(0.9, 0.99))
    loss_func = torch.nn.CrossEntropyLoss()

    Acc = [[],[]]
    Loss = [[],[]]

    for epoch in range(EPOCH):
        start_time = time.time()
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad() # 清空梯度
            loss.backward()
            optimizer.step() # 更新参数
        total_loss = (train_loss * float(BATCH_SIZE)) / len(train_data)
        total_acc = train_acc / (len(train_data))
        Acc[0].append(total_acc)
        Loss[0].append(total_loss)
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(total_loss,total_acc))


        # 评测
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        total_loss = (eval_loss * float(BATCH_SIZE)) / len(test_data)
        total_acc = eval_acc / (len(test_data))
        Acc[1].append(total_acc)
        Loss[1].append(total_loss)
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(total_loss,total_acc))
        print('training took %fs!' % (time.time() - start_time))

    torch.save(Net,'fashion_mnist_module.pkl'+str(epoch))

    labels = ['train-acc', 'test-acc']
    for i, acc in enumerate(Acc):
        plt.plot(acc,label=labels[i])
    plt.title('acc-0.001-512')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.show()

    labels = ['train-loss', 'test-loss']
    for i, loss in enumerate(Loss):
        plt.plot(loss, label=labels[i])
    plt.title('loss-0.001-512')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()