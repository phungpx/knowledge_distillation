import torch
import argparse

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.Mnist import TeacherNet


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
            message = f'Train Epoch: {epoch} [{i * len(x)}/{len(train_loader.dataset)} '
            message += f'({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.data:.6f}'
            print(message)


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

    message = f'{set_name}: {epoch} Average loss: {total_loss / len(test_loader):.4f},'
    message += f'Accuracy: {correct}/{len(test_loader.dataset)} '
    message += f'({100. * correct / len(test_loader.dataset):.0f}%)'
    print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--lr', help='learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--weight-path', type=str, default='weights/teacher_MLP.pth')
    args = parser.parse_args()


    # Device
    device = args.device if ('cuda' in args.device) and torch.cuda.is_available() else 'cpu'

    # Dataloader
    train_loader = DataLoader(
        datasets.MNIST(
            args.data_dir, train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        datasets.MNIST(
            args.data_dir, train=False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )


    # Model Definition
    model = TeacherNet(num_classes=10)
    model.to(device)


    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.num_epochs + 1):
        trainer(epoch, model, train_loader, loss_fn, optimizer, device=device)
        evaluator(epoch, 'Train', model, train_loader, loss_fn, device=device)
        evaluator(epoch, 'Test', model, test_loader, loss_fn, device=device)
        print('\n')

    torch.save(model.state_dict(), args.weight_path)
