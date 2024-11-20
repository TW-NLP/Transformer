import torch


def test(model, tgt_len, enc_input, start_symbol, device):
    """

    :param model:
    :param tgt_len:
    :param enc_input:
    :param start_symbol:
    :param device:
    :return:
    """
    enc_input = enc_input.to(device)

    # 先得到Encoder的输出
    enc_outputs, enc_self_attns = model.Encoder(enc_input)  # [1,src_len, d_model] []
    enc_outputs = enc_outputs.to(device)

    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data).to(device)  # [1, tgt_len]

    next_symbol = start_symbol

    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol

        # 然后一个一个解码
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)  # [1, tgt_len, d_model]

        projected = model.linear(dec_outputs)  # [1, tgt_len, tgt_voc_size]
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]  # [tgt_len][索引]
        next_word = prob.data[i]  # 不断地预测所有字，但是只取下一个字
        next_symbol = next_word.item()
    return dec_input
