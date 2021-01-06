# Reference: https://github.com/yourh/AttentionXML/blob/master/deepxml/modules.py

"""
Created on 2020/12/31
@author Sangwoo Han
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size=None,
        emb_size=None,
        emb_init=None,
        emb_trainable=True,
        padding_idx=0,
        dropout=0.2,
    ):
        super(Embedding, self).__init__()
        if emb_init is not None:
            if vocab_size is not None:
                assert vocab_size == emb_init.shape[0]
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            vocab_size, emb_size = emb_init.shape
        self.emb = nn.Embedding(
            vocab_size,
            emb_size,
            padding_idx=padding_idx,
            sparse=True,
            _weight=torch.from_numpy(emb_init).float()
            if emb_init is not None
            else None,
        )
        self.emb.weight.requires_grad = emb_trainable
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

    def forward(self, inputs):
        emb_out = self.dropout(self.emb(inputs))
        lengths, masks = (inputs != self.padding_idx).sum(
            dim=-1
        ), inputs != self.padding_idx
        return emb_out[:, : inputs.size(-1)], lengths, masks[:, : inputs.size(-1)]


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers_num, dropout):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, layers_num, batch_first=True, bidirectional=True
        )
        self.init_state = nn.Parameter(torch.zeros(2 * 2 * layers_num, 1, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths, **kwargs):
        self.lstm.flatten_parameters()
        init_state = self.init_state.repeat([1, inputs.size(0), 1])
        cell_init, hidden_init = (
            init_state[: init_state.size(0) // 2],
            init_state[init_state.size(0) // 2 :],
        )
        idx = torch.argsort(lengths, descending=True)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs[idx], lengths.cpu()[idx], batch_first=True
        )
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            self.lstm(packed_inputs, (hidden_init, cell_init))[0], batch_first=True
        )
        return self.dropout(outputs[torch.argsort(idx)])


class MLAttention(nn.Module):
    def __init__(self, labels_num, hidden_size):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, labels_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)  # N, 1, L
        attention = (
            self.attention(inputs).transpose(1, 2).masked_fill(~masks, -np.inf)
        )  # N, labels_num, L
        attention = F.softmax(attention, -1)
        return attention @ inputs  # N, labels_num, hidden_size


class MLLinear(nn.Module):
    def __init__(self, linear_size, output_size):
        super(MLLinear, self).__init__()
        self.linear = nn.ModuleList(
            nn.Linear(in_s, out_s)
            for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
        )
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(linear_size[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.squeeze(self.output(linear_out), -1)
