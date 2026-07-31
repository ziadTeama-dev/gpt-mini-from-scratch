import torch
import torch.nn as nn



class LayerNormalization(nn.Module):
    def __init__(self , input_size , eps = 1e-6):
        super().__init__()

        
        self.eps = eps


        self.gamma = nn.Parameter(torch.ones(input_size,),requires_grad=True)

        self.beta = nn.Parameter(torch.zeros(input_size,),requires_grad=True)

    def forward(self, x:torch.tensor ) -> torch.tensor:
        """
        - the equation of the layer normalization (the mean & std of the Layer not the batch)
        - X' = ((X-M)/STD) * r + B
        """

        mean = torch.mean(x,dim=-1,keepdim=True)
        variance = torch.var(x,dim=-1,keepdim=True , unbiased=False)

        # the eps just incase the std = 0
        out = ( (x - mean) / (torch.sqrt( variance + self.eps)) ) * self.gamma + self.beta

        return out


