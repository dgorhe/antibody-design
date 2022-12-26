import torch
from torch import nn

# Original Author: Aladdin Persson
# Reference: https://www.youtube.com/watch?v=VELQT1-hILo

# Input img -> Hidden dim -> mean, std -> Parametrization trick -> Decoder -> Output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20) -> None:
        super().__init__()
        self.vec_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()
    
    def encode(self, x):
        # q_phi(z | x)
        h = self.relu(self.vec_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x | z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma) # torch.randn_like(mu) should also work right?
        z_reparametrized = mu + (sigma * epsilon)
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma