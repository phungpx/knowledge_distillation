import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


train_loader = DataLoader(
    datasets.MNIST(
        './data_mnist',
        train=True,
        download=True,
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    ),
    batch_size=64, shuffle=True,
)

test_loader = DataLoader(
    datasets.MNIST(
        './data_mnist',
        train=False,
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    ),
    batch_size=64, shuffle=False,
)


class TeacherNet(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.fc3(x)
        return x

model = TeacherNet(num_classes=10)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)

loss_fn = nn.CrossEntropyLoss()


def trainer(epoch, model, train_loader, loss_fn, optimizer, device='cuda'):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(f'Train Epoch: {epoch} [{i * len(x)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.data:.6f}')


def evaluator(epoch, set_name, model, test_loader, loss_fn, device='cuda'):
    model.eval()
    total_loss = 0.
    correct = 0.
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        x, y = Variable(x), Variable(y)
        with torch.no_grad():
            y_hat = model(x)
        total_loss += loss_fn(y_hat, y)
        y_hat = y_hat.data.max(dim=1, keepdim=True)[1]   # index
        correct += y_hat.eq(y.data.view_as(y_hat)).cpu().sum()

    print(f'{set_name}: {epoch} Average loss: {total_loss / len(test_loader):.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')

    
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    trainer(epoch, model, train_loader, loss_fn, optimizer, device='cuda')
    evaluator(epoch, 'Train', model, train_loader, loss_fn, device='cuda')
    evaluator(epoch, 'Test', model, test_loader, loss_fn, device='cuda')
    print('\n')

torch.save(model.state_dict(), 'weights/teacher_MLP.pth')
