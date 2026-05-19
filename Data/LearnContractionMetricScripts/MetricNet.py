import torch
import torch.nn as nn

class MetricNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )
        self.state_dim = state_dim
        self.ε = 1e-3

    def forward(self, x):
        A = self.net(x)  # (B, n*n)
        A = A.view(-1, self.state_dim, self.state_dim)
        M = torch.bmm(A.transpose(1, 2), A) + self.ε * torch.eye(self.state_dim).to(x.device)
        return M
