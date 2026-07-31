import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self , dim  , scale_factor=2 ):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(dim , dim*scale_factor),
            nn.GELU(),
            nn.Linear(dim*scale_factor , dim)
        )

    def forward(self,x):
        return self.ffn(x)
