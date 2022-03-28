# Learner: 王振强
# Learn Time: 2022/2/3 23:27
import torch
import torch.nn.functional as F
from config.config import *

"""
    CTC编解码及CTC loss
"""


# CTC 字符串编码
# 输入文本 [xcsc,3sd4,...],
def encode_text_batch(text_batch):
    # 得到文本长度列表[4,4,4,...]
    text_batch_targets_lens = [len(text) for text in text_batch]
    # 转换为int类型
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)

    text_batch_concat = "".join(text_batch)
    text_batch_targets = [config.char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)

    return text_batch_targets, text_batch_targets_lens


# CTC 字符串解码
def decode_predictions(text_batch_logits):

    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2) # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T # [batch_size, T]
    text_batch_tokens_new = []
    # 遍历batch size
    for text_tokens in text_batch_tokens:
        text = []
        # 遍历T方向
        for i in range(len(text_tokens)):
            if text_tokens[i] != 0 and (not (i > 0 and text_tokens[i - 1] == text_tokens[i])):
                text.append(config.idx2char[text_tokens[i]])
        text = "".join(text)
        text_batch_tokens_new.append(text)

    # 传回 batch size 大小的 list
    return text_batch_tokens_new


# CTC loss
# text_batch->target   text_batch_logits->output
def compute_loss(text_batch, text_batch_logits):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])  (8,bz,63)
    """
    # 激活函数
    text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32).to(config.device) # [batch_size]

    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)

    # CTC loss
    '''
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        log_probs: shape为(T, N, C)的模型输出张量其中，T表示CTCLoss的输入长度也即输出序列长度，
                   N表示训练的batch size长度，C则表示包含有空白标签的所有要预测的字符集总长度，
                   log_probs一般需要经过torch.nn.functional.log_softmax处理后再送入到CTCLoss中；
        targets: shape为(N, S) 或(sum(target_lengths))的张量，
                 N表示训练的batch size长度，S则为标签长度，
        input_lengths:  shape为(N)的张量或元组，但每一个元素的长度必须等于T即输出序列长度，一般来说模型输出序列固定后则该张量或元组的元素值均相同；
        target_lengths: shape为(N)的张量或元组，其每一个元素指示每个训练输入序列的标签长度，但标签长度是可以变化的；
    '''
    loss = config.criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss


# 解码并计算得分
def calc_acc(target, output):
    correct = []
    # 解码输出 output
    output = decode_predictions(output.cpu())
    for i, j in zip(target, output):
        if i == j:
            correct.append(1)
        else:
            correct.append(0)
    return sum(correct) / len(correct)




