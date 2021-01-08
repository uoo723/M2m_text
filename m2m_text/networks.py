# Reference: https://github.com/yourh/AttentionXML/blob/master/deepxml/networks.py

"""
Created on 2020/12/31
@author Sangwoo Han
"""

import torch.nn as nn

from .modules import *


class Network(nn.Module):
    def __init__(
        self,
        emb_size,
        vocab_size=None,
        emb_init=None,
        emb_trainable=True,
        padding_idx=0,
        emb_dropout=0.2,
        **kwargs
    ):
        super(Network, self).__init__()
        self.emb = Embedding(
            vocab_size, emb_size, emb_init, emb_trainable, padding_idx, emb_dropout
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AttentionRNN(Network):
    def __init__(
        self,
        labels_num,
        emb_size,
        hidden_size,
        layers_num,
        linear_size,
        dropout,
        **kwargs
    ):
        super(AttentionRNN, self).__init__(emb_size, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, layers_num, dropout)
        self.attention = MLAttention(labels_num, hidden_size * 2)
        self.linear = MLLinear([hidden_size * 2] + linear_size, 1)

    def forward(
        self,
        inputs,
        return_emb=False,
        pass_emb=False,
        return_hidden=False,
        pass_hidden=False,
        rnn_training=False,
        **kwargs
    ):
        if return_emb and pass_emb:
            raise ValueError("`return_emb` and `pass_emb` both cannot be True")

        if return_hidden and pass_hidden:
            raise ValueError("`return_hidden` and `pass_hidden` both cannot be True")

        if return_emb and return_hidden:
            raise ValueError("`return_emb` and `return_hidden` both cannot be True")

        if not pass_emb and not pass_hidden:
            emb_out, lengths, masks = self.emb(inputs, **kwargs)
        elif not pass_hidden:
            emb_out, lengths, masks = inputs
        else:
            emb_out, lengths, masks = None, None, None

        if return_emb:
            return emb_out, lengths, masks

        if emb_out is not None:
            emb_out, masks = emb_out[:, : lengths.max()], masks[:, : lengths.max()]

        if not pass_hidden:
            rnn_out = self.lstm(emb_out, lengths, training=rnn_training)  # N, L, hidden_size * 2
        else:
            rnn_out, lengths, masks = inputs

        if return_hidden:
            return rnn_out, lengths, masks

        attn_out = self.attention(rnn_out, masks)  # N, labels_num, hidden_size * 2

        return self.linear(attn_out)
