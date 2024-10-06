import torch
from torch import nn
from .Transformer import MultiHeadAttention, AddNorm, PositionWiseFFN

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, X, pos = None):
        if self.training:
            X = X + self.P[:, : X.shape[1], :].to(X.device)
        else:
            X = X + self.P[:, pos : pos + 1, :].to(X.device)
        return self.dropout(X)

class DecoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, mha_num_hiddens, ffn_num_hiddens, norm_shape, num_heads,
                 dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(query_size, key_size, value_size, mha_num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout, norm_shape)
        self.ffn = PositionWiseFFN(mha_num_hiddens, ffn_num_hiddens, value_size)
        self.addnorm2 = AddNorm(dropout, norm_shape)

    def forward(self, X, state):
        if state[self.i] == None:
            key_values = X
        else:
            key_values = torch.cat((state[self.i], X), dim = 1)
        state[self.i] = key_values
        batch_size, num_steps, _ = X.shape
        if self.training:
            dec_valid_lens = torch.arange(1, num_steps + 1, device = X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        Y = self.addnorm1(X, self.attention1(X, key_values, key_values, dec_valid_lens))
        return self.addnorm2(Y, self.ffn(Y)), state

class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, value_size, mha_num_hiddens, ffn_num_hiddens, norm_shape,
                 num_heads, num_blocks, dropout):
        super().__init__()
        self.num_hiddens, self.num_blocks = mha_num_hiddens, num_blocks
        self.embed = nn.Embedding(vocab_size, mha_num_hiddens)
        self.pos_encoding = PositionalEncoding(mha_num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blocks):
            self.blks.add_module('block' + str(i),
                                 DecoderBlock(query_size, key_size, value_size, mha_num_hiddens, ffn_num_hiddens,
                                              norm_shape, num_heads, dropout, i))
        self.dense = nn.Linear(value_size, vocab_size)

    def init_state(self):
        return [None] * self.num_blocks

    def forward(self, X, state):
        if self.training:
            X = self.pos_encoding(self.embed(X))
        else:
            X = self.pos_encoding(self.embed(X), state[0].shape[1])
        self._attention_weights = [[None] * self.num_blocks for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
