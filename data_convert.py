import torch
from torch.utils.data import Dataset


def data_split(src_dict, tgt_dict, input_data):
    """

    :param src_dict: 输入词典
    :param tgt_dict: 输出词典
    :param input_data:
    :return:
    """

    input_list = []
    decoder_input = []
    decoder_output = []
    for sent_i in input_data:
        input_list.append([src_dict[i] for i in sent_i[0].split()])
        decoder_input.append([tgt_dict[i] for i in sent_i[1].split()])
        decoder_output.append([tgt_dict[i] for i in sent_i[-1].split()])
    return torch.tensor(input_list), torch.tensor(decoder_input), torch.tensor(decoder_output)


class TransDataset(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        """

        :param enc_inputs: encoder输入
        :param dec_inputs: decoder输入
        :param dec_outputs: decoder输出
        """
        super().__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, index):
        return self.enc_inputs[index], self.dec_inputs[index], self.dec_outputs[index]
