from random import random

import torch.nn as nn

class MaskAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MaskAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x, mask=None):
        # x shape: (seq_len, batch_size, d_model)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        return attn_output

class FeedForward(nn.Module): # two layer fully connected feed foward network
    def __init__(self, d_model, dim_feedforward=2048):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        # Main feed forward
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class DecoderBlock(nn.Module): #no dropout
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super(DecoderBlock, self).__init__()
        self.self_attn = MaskAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Main decoder
        attn_output = self.self_attn(x, mask)
        x = self.norm(x + attn_output) # residual connection
        feed_forward_out = self.feed_forward(x)
        x = self.norm(x + feed_forward_out) # residual connection
        return x

class DecoderTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048):
        super(DecoderTransformer, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        # add any embeddings here e.g. from other data source (like spatio)
        return self.norm(x) # should I just do softmax here?