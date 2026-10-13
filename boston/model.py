import torch.nn as nn


class RedBoston(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.red(x)