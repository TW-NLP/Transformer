import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_convert import data_split, TransDataset
from model import Transformer
from test import test
from train import train

if __name__ == '__main__':
    device = "cpu" if torch.cuda.is_available else "cpu"

    # Encoder_input    Decoder_input          Decoder_output(预测下一个字符)
    sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],  # S: 开始符号
                 ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
                 ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]  # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0

    src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}  # 词源字典  字：索引
    src_idx2word = {v: k for k, v in src_vocab.items()}
    src_vocab_len = len(src_vocab.keys())
    tgt_vocab = {'S': 0, 'E': 1, 'P': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
    tgt_vocab_len = len(tgt_vocab.keys())
    idx2word = {v: k for k, v in tgt_vocab.items()}

    enc_inputs, dec_inputs, dec_outputs = data_split(src_vocab, tgt_vocab, sentences)

    data_set = TransDataset(enc_inputs, dec_inputs, dec_outputs)

    data_loader = DataLoader(data_set, batch_size=2, shuffle=False)

    # transformers参数设置

    d_model = 512  # 字 Embedding 的维度
    d_ff = 2048  # 前向传播隐藏层维度
    d_k = d_v = 64  # K(=Q), V的维度. V的维度可以和K=Q不一样
    n_layers = 6  # 有多少个encoder和decoder
    n_heads = 8  # Multi-Head Attention设置为8
    epochs = 100

    model = Transformer(src_vocab_len, d_model, n_layers, d_model, tgt_vocab_len, n_heads, d_k, d_v, d_ff, n_layers,
                        device)
    model = model.to(device)
    loss = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)

    for epoch in range(epochs):
        train(model, data_loader, loss, optimizer, device)

    with torch.no_grad():

        enc_inputs, _, _ = next(iter(data_loader))
        # 取一个样本
        enc_inputs = enc_inputs[0]
        test_line = enc_inputs.cpu().numpy().tolist()
        enc_str = ""
        for en_i in test_line:
            enc_str += src_idx2word[en_i]

        print(enc_str)

        enc_inputs = enc_inputs.to(device)
        # 预测dec_input
        predict_dec_input = test(model, tgt_vocab_len, enc_inputs[1].view(1, -1), start_symbol=tgt_vocab["S"],
                                 device=device)  # [1, tgt_len]

        out_puts = ""
        test_out = predict_dec_input[0].cpu().numpy().tolist()
        for i in test_out:
            out_puts += idx2word[i]

        print(out_puts)
