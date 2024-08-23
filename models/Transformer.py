import torch
import math
from torch import nn
from d2l import torch as d2l

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype = torch.float32).reshape(-1, 1) / torch.pow(1000, torch.arange(0, num_hiddens, 2,
                                                                                                     dtype = torch.float32) / num_hiddens)
        self.P[:, :, 0:: 2] = torch.sin(X)
        self.P[:, :, 1:: 2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(num_inputs, num_hiddens)
        self.relu = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        # return X

class AddNorm(nn.Module):
    def __init__(self, dropout, norm_shape):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


def transpose_qkv(X, num_heads):
    # X(batch_size, query/key/value数量, num_hiddens)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, - 1)
    # X(batch_size, query/key/value数量, num_heads, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(- 1, X.shape[2], X.shape[3])
    # X(batch_size * num_heads, query/key/value数量, num_hiddens / num_heads)


def transpose_output(X, num_heads):
    X = X.reshape(- 1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], - 1)


class MultiHeadAttention(nn.Module):
    # 默认使用点积注意力
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.attention = d2l.DotProductAttention(dropout)
        # self.W_q = nn.Sequential(nn.Linear(query_size, num_hiddens, bias = False), nn.ReLU(), nn.Linear(num_hiddens, num_hiddens, bias = False))
        # self.W_k = nn.Sequential(nn.Linear(key_size, num_hiddens, bias = False), nn.ReLU(), nn.Linear(num_hiddens, num_hiddens, bias = False))
        # self.W_v = nn.Sequential(nn.Linear(value_size, num_hiddens, bias = False), nn.ReLU(), nn.Linear(num_hiddens, num_hiddens, bias = False))
        self.W_q = nn.Linear(query_size, num_hiddens, bias = False)
        self.W_k = nn.Linear(key_size, num_hiddens, bias = False)
        self.W_v = nn.Linear(value_size, num_hiddens, bias = False)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias = False)
        # 默认最后输出大小为num_hiddens

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats = self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class EncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, mha_num_hiddens, ffn_num_hiddens, norm_shape, num_heads,
                 dropout):
        super().__init__()
        self.attention = MultiHeadAttention(query_size, key_size, value_size, mha_num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout, norm_shape)
        self.ffn = PositionWiseFFN(mha_num_hiddens, ffn_num_hiddens, value_size)
        self.addnorm2 = AddNorm(dropout, norm_shape)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y)) #两个addnorm是一样的woc


class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, query_size, key_size, value_size, mha_num_hiddens, ffn_num_hiddens, norm_shape,
                 num_heads, num_blocks, dropout):
        super().__init__()
        self.num_hiddens = mha_num_hiddens
        self.embed = nn.Embedding(vocab_size, mha_num_hiddens)
        self.pos_encoding = PositionalEncoding(mha_num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blocks):
            self.blks.add_module('block' + str(i),
                                 EncoderBlock(query_size, key_size, value_size, mha_num_hiddens, ffn_num_hiddens,
                                              norm_shape, num_heads, dropout))

    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embed(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, mha_num_hiddens, ffn_num_hiddens, norm_shape, num_heads,
                 dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(query_size, key_size, value_size, mha_num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout, norm_shape)
        self.attention2 = MultiHeadAttention(query_size, key_size, value_size, mha_num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(dropout, norm_shape)
        self.ffn = PositionWiseFFN(mha_num_hiddens, ffn_num_hiddens, value_size)
        self.addnorm3 = AddNorm(dropout, norm_shape)

    def forward(self, X, state):
        # state = [enc_outputs, enc_valid_lens, 每个解码块上一时间步的输出]
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] == None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        batch_size, num_steps, _ = X.shape
        if self.training:
            dec_valid_lens = torch.arange(1, num_steps + 1, device = X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        Y = self.addnorm1(X, self.attention1(X, key_values, key_values, dec_valid_lens))
        Z = self.addnorm2(Y, self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens))
        # enc_valid_lens是为了忽略'<pad>'，dec_valid_len是为了忽略未到达的时间步
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.Decoder):
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

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blocks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embed(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * self.num_blocks for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights