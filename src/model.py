import torch 
import torch.nn as nn
from src.decoderblock import DecoderBlock
from src.layernorm import LayerNormalization


class GPT_mini(nn.Module):
    def __init__(self, 
                 dim=512 , 
                 num_head=8 , 
                 num_layers=5 ,
                 vocab_size=None,
                 max_sequance=None,
                 scale_factor=2 , 
                 eps=1e-6 ,
                 dropout_value=.1 ,
                 padding_mask =False) :
        
        super().__init__()
        self.max_sequance = max_sequance

        self.embedding = nn.Embedding(vocab_size , dim)

        self.pos_embedding = nn.Embedding(max_sequance , dim)

        self.layers = nn.ModuleList([
                                    DecoderBlock(dim,
                                                 num_head,
                                                 scale_factor,
                                                 eps,
                                                 dropout_value)
            for _ in range(num_layers)
        ])

        self.layernorm = LayerNormalization(dim , eps)


        self.fc = nn.Linear(dim , vocab_size)


    def forward(self,x ,mask = True , padding_mask=None , noise_std=.2):

        attention_list = []
        B,S = x.shape
        embedding_out = self.embedding(x)

        # learnable pos emp
        postional_indices = torch.arange(0,S,device=x.device).unsqueeze(0).expand(B,S)
        postional_embedding_values = self.pos_embedding(postional_indices)

        # now the embedding have info about word location in sentence
        out = embedding_out + postional_embedding_values

        # adding guassian noise as augmentation step
        if self.training :
            if noise_std > 0 :
                if torch.randint(0,2,(1,)).item() == 1:
                    guassian_noise = torch.randn_like(out,device=x.device) * torch.rand(B,S,1,device=x.device) * .2

                    out = out + guassian_noise


        # passing the input throght all layers
        for layer in self.layers:
            out , attention_score= layer(out , mask ,padding_mask )
            attention_list.append(attention_score)

        out = self.layernorm(out)

        # prediction
        return self.fc(out) ,attention_list



    

