# Reference: https://github.com/yourh/AttentionXML/blob/master/deepxml/networks.py

"""
Created on 2020/12/31
@author Sangwoo Han
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Embedding, LSTMEncoder, MLAttention, MLLinear


class Network(nn.Module):
    def __init__(
        self,
        emb_size: int,
        vocab_size: Optional[int] = None,
        emb_init: Optional[int] = None,
        emb_trainable: bool = True,
        padding_idx: int = 0,
        emb_dropout: float = 0.2,
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
        labels_num: int,
        emb_size: int,
        hidden_size: int,
        layers_num: int,
        linear_size: List[int],
        dropout: float,
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
            rnn_out = self.lstm(
                emb_out, lengths, training=rnn_training
            )  # N, L, hidden_size * 2
        else:
            rnn_out, lengths, masks = inputs

        if return_hidden:
            return rnn_out, lengths, masks

        attn_out = self.attention(rnn_out, masks)  # N, labels_num, hidden_size * 2

        return self.linear(attn_out)


class FCNet(nn.Module):
    """FCNet

    Args:
        labels_num (int): Number of labels.
        linear_size (list[int]): List of size (dimension) of fc layer.
        input_size (int, optional): Input size, if None input size is inferred.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        labels_num: int,
        linear_size: List[int],
        input_size: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.labels_num = labels_num
        self.linear_size = linear_size
        self.input_size = input_size
        self.dropout_p = dropout
        self._built = False

        if input_size:
            self._build()

    def _build(self):
        input_size = self.input_size
        linear_size = self.linear_size
        labels_num = self.labels_num
        dropout = self.dropout_p

        self.linear = MLLinear([input_size] + linear_size[:-1], linear_size[-1])
        self.output = MLLinear(linear_size[-1:], labels_num)
        self.dropout = nn.Dropout(dropout)

        self._built = True

    def forward(
        self,
        inputs: torch.Tensor,
        return_hidden: bool = False,
        pass_hidden: bool = False,
    ) -> torch.Tensor:
        """Forward inputs

        Args:
            inputs (torch.Tensor): Input tensor with shape (batch_size, input_size).
            return_hidden (bool): Return hidden if true. default: False.
            pass_hidden (bool): Treat inputs tensor as a hidden if true. default: False.

        Returns:
            outputs (torch.Tensor): Output tensor.
        """
        if not self._built:
            self.input_size = inputs.size(-1)
            self._build()

        if return_hidden and pass_hidden:
            raise ValueError("`return_hidden` and `pass_hidden` both cannot be True")

        if return_hidden:
            return self.linear(inputs)

        if pass_hidden:
            return self.output(self.dropout(F.relu(inputs)))

        outputs = self.linear(inputs)
        return self.output(self.dropout(F.relu(outputs)))
