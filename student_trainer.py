import torch

from torch import nn
from torch import optim
from losses.KD_loss import KDLoss
from models.Mnist import TeacherNet, StudentNet


def distiller_train(epoch, teacher, student, data_loader, optimizer, loss_fn, device):
    teacher.eval()
    student.train()
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        x, y = Variable(x), Variable(y)
        optimizer.zero_grad()
        student_y = student(x)
        with torch.no_grad():
            teacher_y = teacher(x)
        loss = loss_fn(student_y, teacher_y, y)
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
    parser.add_argument('--temperature-scale', type=float, default=20)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--lr', help='learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--teacher-weight-path', type=str, default='weights/teacher_MLP.pth')
    parser.add_argument('--weight-path', type=str, default='weights/distiller_MLP.pth')
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


    # Teacher
    teacher = TeacherNet(num_classes=10)
    teacher.load_state_dict(torch.load(args.teacher_weight_path))
    teacher.to(device)

    # Student
    student = StudentNet(num_classes=10)
    student.to(device)

    # Optimizer
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Loss Function
    distil_loss = KDLoss(T=args.temperature_scale, alpha=args.alpha)
    ce_loss = nn.CrossEntropyLoss()


    for epoch in range(1, num_epochs + 1):
        distiller(epoch, teacher, student, train_loader, optimizer, distil_loss, device=device)
        evaluator(epoch, 'Train', student, train_loader, ce_loss, device=device)
        evaluator(epoch, 'Test', student, test_loader, ce_loss, device=device)
        print('\n')

    torch.save(model.state_dict(), args.weight_path)
