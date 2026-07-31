import torch
import math 
import torch.nn as nn



class MHA(nn.Module):
    def __init__(self , dim : int , num_head:int ):
        super().__init__()

        self.num_head = num_head
        self.head_dim = dim // num_head

        assert(self.head_dim * self.num_head == dim) ,"d_MODEL should be devisible by heads"

        self.query = nn.Linear(dim,dim)
        self.key = nn.Linear(dim,dim)
        self.value = nn.Linear(dim,dim)

        self.out_put_layer = nn.Linear(dim,dim)



    def forward(self, query:torch.tensor , key:torch.tensor , value:torch.tensor , mask : bool = True , padding_mask: torch.tensor =None):
        # setup attention component
        B_q,S_q,_ = query.shape
        B_k,S_k,_ = key.shape
        B_v,S_v,d_v = value.shape
        # 
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)


        # divide each dimension into head
        query = query.reshape(B_q,S_q,self.num_head,self.head_dim).transpose(1,2)
        key = key.reshape(B_k,S_k,self.num_head,self.head_dim).transpose(1,2)
        value = value.reshape(B_v,S_v,self.num_head,self.head_dim).transpose(1,2)

        qk = torch.einsum("BHQD,BHKD->BHQK", query, key)
        
        qk = qk / math.sqrt(self.head_dim)

        if mask :
            # creating the mask
            attention_mask = torch.triu( torch.ones(S_q,S_k,device=query.device) , diagonal=1)
            # applying the mask
            qk = qk.masked_fill(attention_mask == 1 , float("-inf"))

        if padding_mask is not None :
            padding_mask = padding_mask[:,None,None,:].to(query.device)
            qk = qk.masked_fill(padding_mask == 0 , float("-inf"))




        attention_score = torch.softmax( qk , dim=-1)

        output = torch.einsum("BHQK,BHKD->BHQD", attention_score , value )
        output = output.transpose(1,2).reshape( B_q, S_q, self.num_head*self.head_dim)

        output = self.out_put_layer(output)

        return output , attention_score




