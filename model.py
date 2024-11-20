import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.optim as optim

from data_convert import data_split, TransDataset


def get_attn_pad_mask(seq_q, seq_k):
    """
    此时字还没表示成嵌入向量
    句子0填充
    seq_q中的每个字都要“看”一次seq_k中的每个字
    Args: 在Encoder_self_att中，seq_q，seq_k 就是enc_input
            seq_q (_type_): [batch, enc_len] [batch, 中文句子长度]
            seq_k (_type_): [batch, enc_len] [batch, 中文句子长度] 主要的
          在Decoder_self_att中，seq_q，seq_k 就是dec_input, dec_input
            seq_q (_type_): [batch, tgt_len] [batch, 英文句子长度]
            seq_k (_type_): [batch, tgt_len] [batch, 英文句子长度]
          在Decoder_Encoder_att中，seq_q，seq_k 就是dec_input, enc_input
            seq_q (_type_): [batch, tgt_len] [batch, 中文句子长度]
            seq_k (_type_): [batch, enc_len] [batch, 英文句子长度]

    Returns:
        _type_: [batch_size, len_q, len_k]  元素：T or F
    """
    batch_size, len_q = seq_q.size()  # seq_q 用于升维，为了做attention，mask score矩阵用的
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0)  # 判断 输入那些词index含有P(=0),用1标记 [len_k, d_model]元素全为T,F
    pad_attn_mask = pad_attn_mask.unsqueeze(1)  # [batch, 1, len_k]
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # 扩展成多维度   [batch_size, len_q, len_k]
    return pad_attn_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, dropout=0.1, max_len=5000):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        pos_list = []
        for pos in range(max_len):
            if pos != 0:
                pos_list.append(np.zeros(d_model))
            else:
                pos_list.append([pos / np.power(10000, 2 * i / d_model) for i in range(d_model)])
        pos_table = np.array(pos_list)

        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
        pos_table[1:, 1::2] = np.sin(pos_table[1:, 1::2])  # [seq_len,d_model]

        self.pos_table = torch.FloatTensor(pos_table)

    def forward(self, embedding_input):
        """_summary_

        Args:
            embedding_input (_type_): _description_
            [seq_len,batch_size,d_model]
        """

        seq_len = embedding_input.size(0)
        enc_inputs = embedding_input + self.pos_table[:seq_len, :].unsqueeze(1).to(self.device)
        enc_inputs = self.dropout(enc_inputs)
        return enc_inputs


class Attention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attention_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V)

        return context, attn


class MUltiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        # d_k*n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.attention = Attention(d_k=d_k)
        self.fc = nn.Linear(n_heads * d_k, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, input_Q, input_K, input_V, attn_mask):
        res, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q.float())
        # 变为多头 [batch_size,seq_len,n_heads,d_k]->[batch_size,n_heads,seq_len,d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K.float())
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V.float())
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, len_q, len_k]

        context, attn = self.attention(Q, K, V, attn_mask)
        # concat mul head information

        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_k)
        output = self.fc(context)
        return self.ln(output + res), attn


class FF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): [batch_size,seq_len,d_model]
        """
        res = inputs
        output = self.linear1(inputs)
        output = self.linear2(self.relu(output))

        return self.ln(output + res)


class Encoderlayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.mul_attention = MUltiHeadAttention(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)
        self.feed = FF(d_model=d_model, d_ff=d_ff)

    def forward(self, enc_inputs, enc_attention_mask):
        """_summary_

        Args:
            enc_inputs (_type_): [batch_size,seq_len,d_model]
            enc_attention_mask (_type_): [batch_size,seq_len,seq_len]
        """

        enc_outputs, attn = self.mul_attention(enc_inputs, enc_inputs, enc_inputs, enc_attention_mask)
        enc_outputs = self.feed(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, encode_layers, d_model, n_heads, d_k, d_v, d_ff, device):
        super().__init__()
        # embedding 编码
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.position = PositionalEncoding(d_model=embedding_dim, device=device)
        self.layers = nn.ModuleList(
            [Encoderlayer(d_model, n_heads, d_k, d_v, d_ff) for i in range(encode_layers)])
        self.ff = FF(d_model, d_ff)
        self.device = device

    def forward(self, enc_inputs):
        """_summary_

        Args:
            enc_inputs (_type_): _description_
            [batch_size,src_len]
        """
        # 词embedding
        enc_outputs = self.src_embedding(enc_inputs)  # [batch,src_len,d_model]
        # 位置编码与词embedding进行汇总
        enc_outputs = self.position(enc_outputs.transpose(0, 1)).transpose(0, 1)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs).to(self.device)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attention = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attention)

        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super().__init__()
        self.dec_mask_attention = MUltiHeadAttention(d_model, n_heads, d_k, d_v)
        self.cross_attention = MUltiHeadAttention(d_model, n_heads, d_k, d_v)
        self.ff = FF(d_model, d_ff)

    def forward(self, dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_mask):
        """_summary_

        Args:
            dec_outputs (_type_): _description_
            enc_outputs (_type_): _description_
            dec_self_attn_mask (_type_): _description_
            dec_enc_mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        dec_outputs, dec_self_attention = self.dec_mask_attention(dec_outputs, dec_outputs, dec_outputs,
                                                                  dec_self_attn_mask)

        dec_outputs, dec_enc_attention = self.cross_attention(dec_outputs, enc_outputs, enc_outputs, dec_enc_mask)
        dec_outputs = self.ff(dec_outputs)

        return dec_outputs, dec_self_attention, dec_enc_attention


# 搭建decoder


def get_subsequence_mask(seq):
    """_summary_

    Args:
        seq (_type_): [batch_size,tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    sub_mask = np.triu(np.ones(attn_shape), k=1)
    sub_mask = torch.from_numpy(sub_mask).byte()
    return sub_mask


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_len, d_model, n_layers, n_heads, d_k, d_v, d_ff, device):
        super().__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_len, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model, device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)])
        self.device = device

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """_summary_

        Args:
            dec_inputs (_type_): decoder的输入
            enc_inputs (_type_): encoder的输出,来计算mask的
            enc_outputs (_type_): encoder的输出
        """
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)
        dec_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(self.device)
        # 这是mask attention
        # 上三角mask
        dec_sub_mask = get_subsequence_mask(dec_inputs).to(self.device)

        # 两个mask 相加，来mask掉pad的部分
        dec_self_attn_mask = torch.gt((dec_sub_mask + dec_pad_mask),
                                      0)

        # 这是decoder的 mul attention

        dec_enc_mask = get_attn_pad_mask(dec_inputs, enc_inputs).to(self.device)

        dec_self_attn_sum = []

        dec_enc_attn_sum = []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_mask)
            dec_enc_attn_sum.append(dec_enc_attn)
            dec_self_attn_sum.append(dec_self_attn)

        return dec_outputs, dec_self_attn_sum, dec_enc_attn_sum


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, encode_layers, d_model, tgt_vocab_len, n_heads, d_k, d_v, d_ff,
                 n_layers,
                 device):
        """

        :param src_vocab_size:
        :param embedding_dim:
        :param encode_layers:
        :param d_model:
        :param tgt_vocab_len:
        :param n_heads:
        :param d_k:
        :param d_v:
        :param d_ff:
        :param n_layers:
        :param device:
        """

        super().__init__()

        self.Encoder = Encoder(src_vocab_size, embedding_dim, encode_layers, d_model, n_heads, d_k, d_v, d_ff, device)
        self.Decoder = Decoder(tgt_vocab_len, d_model, n_layers, n_heads, d_k, d_v, d_ff, device)

        self.linear = nn.Linear(d_model, tgt_vocab_len, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)

        dec_outputs, dec_self_attn_sum, dec_enc_attn_sum = self.Decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_outputs = self.linear(dec_outputs)
        # 将batch数据展平为一句话
        dec_outputs = dec_outputs.view(-1, dec_outputs.size(-1))
        return dec_outputs
