import torch.nn.functional as F
from torch import nn


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


class StudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
