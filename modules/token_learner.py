'''
Implementation of A3 module based on TokenLearner.

Copyright 2022 Alibaba
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    
    def __init__(self, input_embed_dim, out_token=30):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size = (1,1), stride=1, groups=8, bias=False),
                                          nn.Conv2d(input_embed_dim, out_token, kernel_size = (1,1), stride=1, bias=False))
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size = (1,1), stride=1, groups=8, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x)
        x = x.transpose(1, 2).unsqueeze(-1)
        selected = self.tokenLearner(x)
        selected = selected.flatten(2)
        selected = F.softmax(selected, dim=-1) 
        feat = self.feat(x)
        feat = feat.flatten(2).transpose(1,2)
        x = torch.einsum('...si,...id->...sd', selected, feat)
        
        x = self.norm(x)
        return selected, x