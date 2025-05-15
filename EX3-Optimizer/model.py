import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 128, 64], dropout=0.0, use_bn=True, output_dim=10):
        super(MLP, self).__init__()
        layers = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = h

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
