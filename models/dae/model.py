import torch
from torch import nn

# TODO: Find a way to make hidden dimensions dynamic
class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=(200, 150, 100, 50)) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim[0]),
            nn.ReLU(),
            nn.Linear(h_dim[0], h_dim[1]),
            nn.ReLU(),
            nn.Linear(h_dim[1], h_dim[2]),
            nn.ReLU(),
            nn.Linear(h_dim[2], h_dim[3]),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(h_dim[3], h_dim[2]),
            nn.ReLU(),
            nn.Linear(h_dim[2], h_dim[1]),
            nn.ReLU(),
            nn.Linear(h_dim[1], h_dim[0]),
            nn.ReLU(),
            nn.Linear(h_dim[0], input_dim)
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x