import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def attention(query, value, key, mask, dropout):
    d_k = query.size(-1)  # dim of a head
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.cpu()
    if mask is not None:
        scores = scores.masked_fill(Variable(mask) == 0, -1e9).cuda()
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultuHieadedAttention(nn.Module):
    def __init__(self, num_head=10, d_model=300, dropout=0.1):
        super(MultuHieadedAttention, self).__init__()
        self.d_k = d_model // num_head
        self.h = num_head
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        query, key, value = [
            x.view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
            for x in (query, key, value)
        ]
        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.h * self.d_k)
        )
        return x


if __name__ == "__main__":
    MultuHieadedAttention_test = MultuHieadedAttention()
    input = torch.randn((1, 10, 300)).cuda()
    mask = torch.Tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    output = MultuHieadedAttention_test(input, input, input, mask)
    print(output)
