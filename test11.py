from deep_rl import *
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import trange, tqdm
import random
import numpy as np

select_device(0)

class torch_reshape(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, 64, 9, 9)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class SRNetImage(nn.Module):
    def __init__(self, output_dim, hidden_units_sr=(512*4,), hidden_units_psi2q=(), gate=F.relu, config=1):
        """
        This network has two heads: SR head (SR) and reconstruction head (rec).
        config -> type of learning on top of state abstraction
            0 - typical SR with weights sharing
            1 - learning SR without weights sharing
        """
        super(SRNetImage, self).__init__()
        self.feature_dim = 512
        self.output_dim = output_dim
        self.gate = gate

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, kernel_size=3, stride=2)),  # b, 16, 10, 10
            nn.ReLU(True),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2)), 
            nn.ReLU(True),
            Flatten(),
            nn.Linear(9 * 9 * 64, self.feature_dim)
        )

        self.decoder = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, 9 * 9 * 64)),
            torch_reshape(),
            layer_init(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)),  # b, 16, 5, 5
            nn.ReLU(True),
            layer_init(nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, output_padding=1)),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.Tanh()
        )

        # layers for SR
        dims_sr = (self.feature_dim,) + hidden_units_sr + (self.feature_dim * output_dim,)
        self.layers_sr = nn.ModuleList(
            [layer_init_0(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims_sr[:-1], dims_sr[1:])])

        # SR final head layer
        if(config == 0):
            self.psi2q = Psi2QNet(output_dim, self.feature_dim)
        if(config == 1):
            self.psi2q = Psi2QNetFC(output_dim, self.feature_dim, hidden_units=hidden_units_psi2q)

        self.to(Config.DEVICE)

    def forward(self, x):

        # Finding the latent layer
        phi = self.encoder(tensor(x)) # shape: b x state_dim

        # Reconstruction
        state_est = self.decoder(phi)

        # Estimating the SR from the latent layer
        psi = phi
        for layer in self.layers_sr[:-1]:
            psi = self.gate(layer(psi))
        psi = self.layers_sr[-1](psi)
        psi = psi.view(psi.size(0), self.output_dim, self.self.feature_dim) # shape: b x action_dim x state_dim
        q_est = self.psi2q(psi)

        return phi, psi, state_est, q_est


