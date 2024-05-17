import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ResNet(nn.Module):
    """Encoder for feature embedding"""

    def __init__(self, args):
        super(ResNet, self).__init__()
        self.args = args
        h_dim, z_dim = args.h_dim, args.z_dim

        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, h_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(1, h_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # nn.Flatten(),
            # nn.Linear(h_dim * 16, 50),
        )

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        out1 = self.cnn1(x1)
        out2 = self.cnn2(x2)
        out = torch.cat([out1.unsqueeze(-2), out2.unsqueeze(-2)], -2)

        return out

if __name__=='__main__':
    model=ResNet()
    input = torch.FloatTensor(5, 3, 80, 80)
    out = model(input)
