import torch.nn as nn

class SpatioGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(SpatioGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return x