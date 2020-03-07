from torch import nn
import torch.nn.functional as F


class MlpSample(nn.Module):
    def __init__(self):
        self.dset_type = {"mnist"}
        super(MlpSample, self).__init__()
        self.fc1 = nn.Linear(784,500)
        self.fc2 = nn.Linear(500, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x