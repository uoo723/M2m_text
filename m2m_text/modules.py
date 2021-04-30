# Reference: https://github.com/yourh/AttentionXML/blob/master/deepxml/modules.py

"""
Created on 2020/12/31
@author Sangwoo Han
"""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def forward(self, inputs):
        return inputs


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        emb_size: Optional[int] = None,
        emb_init: Optional[Union[np.ndarray, str]] = None,
        emb_trainable: bool = True,
        padding_idx: int = 0,
        dropout: bool = 0.2,
    ):
        super(Embedding, self).__init__()
        if emb_init is not None:
            if type(emb_init) == str:
                emb_init = np.load(emb_init)
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


class LabelEmbedding(nn.Module):
    def __init__(
        self,
        num_labels: int,
        emb_size: Optional[int] = None,
        emb_init: Optional[Union[np.ndarray, str]] = None,
        emb_trainable: bool = True,
        dropout: bool = 0.2,
    ):
        super().__init__()
        if emb_init is not None:
            if type(emb_init) == str:
                emb_init = np.load(emb_init)
            if emb_size is not None:
                assert emb_size == emb_init.shape[1]
            assert num_labels == emb_init.shape[0]
        else:
            assert emb_size is not None

        self.emb = nn.Embedding(
            num_labels,
            emb_size,
            sparse=True,
            _weight=torch.from_numpy(emb_init).float()
            if emb_init is not None
            else None,
        )
        self.emb.weight.requires_grad = emb_trainable
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return self.dropout(self.emb(inputs))


class LSTMEncoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.init_state = nn.Parameter(torch.zeros(2 * 2 * num_layers, 1, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths, training=False, **kwargs):
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

        training_state = self.lstm.training

        if training:
            self.lstm.train()

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            self.lstm(packed_inputs, (hidden_init, cell_init))[0],
            total_length=inputs.size(1),
            batch_first=True,
        )

        if training:
            self.lstm.train(training_state)

        return self.dropout(outputs[torch.argsort(idx)])


class MLAttention(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, num_labels, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)  # N, 1, L
        attention = (
            self.attention(inputs).transpose(1, 2).masked_fill(~masks, -np.inf)
        )  # N, labels_num, L
        attention = F.softmax(attention, -1)
        return attention @ inputs  # N, labels_num, hidden_size


class MLLinear(nn.Module):
    def __init__(self, linear_size: List[int], output_size: int):
        super(MLLinear, self).__init__()
        self.linear = nn.ModuleList(
            nn.Linear(in_s, out_s)
            for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
        )
        self.output = nn.Linear(linear_size[-1], output_size)
        self.init_weights()

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.squeeze(self.output(linear_out), -1)

    def init_weights(self):
        """Initialize weights"""
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        nn.init.xavier_uniform_(self.output.weight)


# https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
class GraphConvolution(nn.Module):
    """GCN layer"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = inputs @ self.weight
        output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features}"


class Readout(nn.Module):
    def __init__(self, linear_size: List[int], output_size: int):
        super().__init__()
        self.linear = nn.ModuleList(
            nn.Linear(in_s, out_s)
            for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
        )
        self.output = nn.Linear(linear_size[-1], output_size)
        self.init_weights()

    def forward(self, inputs):
        outputs = inputs.sum(dim=1)
        for linear in self.linear:
            outputs = F.relu(linear(outputs))
        return self.output(outputs)

    def init_weights(self):
        """Initialize weights"""
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        nn.init.xavier_uniform_(self.output.weight)


class GCNLayer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_size: List[int],
        dropout: float,
        init_adj: Optional[torch.Tensor] = None,
        adj_trainable: bool = False,
        gcn_adj_dropout: Optional[float] = None,
    ):
        super(GCNLayer, self).__init__()
        self.gc = nn.ModuleList(
            GraphConvolution(in_s, out_s)
            for in_s, out_s in zip(hidden_size[:-1], hidden_size[1:])
        )
        self.adj = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.dropout = nn.Dropout(dropout)

        if init_adj is not None:
            self.adj.data = init_adj
        else:
            nn.init.xavier_uniform_(self.adj.data)

        self.adj.requires_grad = adj_trainable

        if gcn_adj_dropout is not None:
            self.gcn_adj_dropout = nn.Dropout(gcn_adj_dropout)
        else:
            self.gcn_adj_dropout = Identity()

    def forward(self, inputs):
        outputs = inputs
        for layer in self.gc:
            outputs = F.relu(layer(outputs, self.gcn_adj_dropout(self.adj)))
            outputs = self.dropout(outputs)

        return outputs


# Refernce: https://github.com/XunGuangxu/CorNet/blob/master/deepxml/cornet.py
class CorNetBlock(nn.Module):
    def __init__(self, context_size: int, output_size: int):
        super(CorNetBlock, self).__init__()
        self.dstbn2cntxt = nn.Linear(output_size, context_size)
        self.cntxt2dstbn = nn.Linear(context_size, output_size)

    def forward(self, output_dstrbtn: torch.Tensor):
        identity_logits = output_dstrbtn
        output_dstrbtn = torch.sigmoid(output_dstrbtn)
        context_vector = self.dstbn2cntxt(output_dstrbtn)
        context_vector = F.elu(context_vector)
        output_dstrbtn = self.cntxt2dstbn(context_vector)
        output_dstrbtn = output_dstrbtn + identity_logits
        return output_dstrbtn


# Refernce: https://github.com/XunGuangxu/CorNet/blob/master/deepxml/cornet.py
class CorNet(nn.Module):
    def __init__(self, output_size: int, context_size: List[int], **kwargs):
        super(CorNet, self).__init__()
        self.intlv_layers = nn.ModuleList(
            [CorNetBlock(size, output_size, **kwargs) for size in context_size]
        )
        for layer in self.intlv_layers:
            nn.init.xavier_uniform_(layer.dstbn2cntxt.weight)
            nn.init.xavier_uniform_(layer.cntxt2dstbn.weight)

    def forward(self, logits):
        for layer in self.intlv_layers:
            logits = layer(logits)
        return logits


class GateAttention(nn.Module):
    def __init__(self, hidden_size: int, n_gates: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, n_gates, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs):
        attn = F.softmax(self.attention(inputs).transpose(1, 2), dim=-1)
        return attn @ inputs
