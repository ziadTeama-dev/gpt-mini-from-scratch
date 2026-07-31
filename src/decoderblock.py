import torch.nn as nn

from src.mha import MHA
from src.layernorm import LayerNormalization
from src.feedforward import FeedForwardNetwork



class DecoderBlock(nn.Module):
    def __init__(self, dim = 512 , num_head = 8,scale_factor=2 , eps = 1e-6 , drop_out_value=.1):
        super().__init__()

        self.multiheadattention = MHA(dim,
                                      num_head)
        
        self.layernorm_1 = LayerNormalization(dim,
                                              eps)
        
        self.ffn = FeedForwardNetwork(dim,
                                      scale_factor)
        
        self.layernorm_2 = LayerNormalization(dim,
                                              eps)

        self.dropout = nn.Dropout(drop_out_value)

    def forward(self,x ,mask=True,padding_mask = None):
        # norm
        out_norm = self.layernorm_1(x)

        out_attention,attention_scores = self.multiheadattention(out_norm,out_norm,out_norm,
                                                       mask=mask,
                                                       padding_mask=padding_mask)
        out_attention = self.dropout(out_attention)
        # (res 1)
        out_res = out_attention + x
        out_norm = self.layernorm_2(out_res)
        # feedforward
        out = self.ffn(out_norm)
        out = self.dropout(out)
        # res 2
        out = out + out_res

        return out , attention_scores