# Reference: https://github.com/yourh/AttentionXML/blob/master/deepxml/modules.py

"""
Created on 2020/12/31
@author Sangwoo Han
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import check_eq_shape, dgl_warning, expand_as_pair


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


class MLAttention2(nn.Module):
    def __init__(self, num_labels: int, hidden_size: int, sparse: bool = True):
        super().__init__()
        self.attn_weight = nn.Parameter(torch.Tensor(num_labels, hidden_size))
        self.sparse = sparse
        nn.init.xavier_uniform_(self.attn_weight)

    def forward(self, inputs: torch.Tensor, label_ids: torch.LongTensor = None):
        if label_ids:
            attn_weight = F.embedding(label_ids, self.attn_weight, sparse=self.sparse)
        else:
            attn_weight = self.attn_weight
        attn = (inputs @ attn_weight.T).transpose(1, 2)  # N, num_labels, L
        attn = F.softmax(attn, -1)
        return attn @ inputs  # N, num_labels, hidden_size


class MLAttentionForSBert(nn.Module):
    def __init__(self, num_embeddings: int, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, num_embeddings, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"].bool()

        attention_mask = torch.unsqueeze(attention_mask, 1)  # N, 1, L
        attention = (
            self.attention(token_embeddings)
            .transpose(1, 2)
            .masked_fill(~attention_mask, -np.inf)
        )  # N, num_embeddings, L
        attention = F.softmax(attention, -1)

        # N, num_embeddings, hidden_size
        features["sentence_embedding"] = (attention @ token_embeddings).squeeze()
        return features


class MLLinear(nn.Module):
    def __init__(
        self, linear_size: List[int], output_size: int, enable_layer_norm: bool = False
    ):
        super(MLLinear, self).__init__()
        self.linear = nn.ModuleList(
            nn.Linear(in_s, out_s)
            for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
        )
        self.output = nn.Linear(linear_size[-1], output_size)

        self.layer_norm = nn.ModuleList(
            nn.LayerNorm(out_s) if enable_layer_norm else Identity()
            for out_s in linear_size[1:]
        )

        self.init_weights()

    def forward(self, inputs):
        linear_out = inputs
        for linear, layer_norm in zip(self.linear, self.layer_norm):
            linear_out = F.relu(layer_norm(linear(linear_out)))
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


# https://github.com/Extreme-classification/GalaXC/blob/1cdf1908025a854bdc7d8697220e7813592466e7/network.py#L430
class Residual(nn.Module):
    """Residual layer implementation"""

    def __init__(self, input_size, output_size, dropout, init="eye"):
        super(Residual, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init = init
        self.dropout = dropout
        self.padding_size = self.output_size - self.input_size
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.BatchNorm1d(self.output_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.initialize(self.init)

    def forward(self, embed):
        temp = F.pad(embed, (0, self.padding_size), "constant", 0)
        embed = self.hidden_layer(embed) + temp
        return embed

    def initialize(self, init_type):
        if init_type == "random":
            nn.init.xavier_uniform_(
                self.hidden_layer[0].weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)
        else:
            nn.init.eye_(self.hidden_layer[0].weight)
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)


# https://docs.dgl.ai/en/0.5.x/_modules/dgl/nn/pytorch/conv/sageconv.html
class SAGEConv(nn.Module):
    """GraphSAGE layer."""

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        aggregator_type: str,
        feat_drop: float = 0.0,
        bias: bool = True,
        norm: Optional[Callable] = None,
        activation: Optional[Callable] = None,
    ):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )
        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, "bias"):
            dgl_warning(
                "You are loading a GraphSAGE model trained from a old version of DGL, "
                "DGL automatically convert it to be compatible with latest version."
            )
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, "fc_self"):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(
        self,
        graph: dgl.DGLGraph,
        feat: Union[torch.Tensor, Tuple[torch.Tensor]],
        edge_weight: Optional[torch.Tensor] = None,
    ):
        """Compute GraphSAGE layer.

        Args:
            graph (dgl.DGLGraph): The graph.
            feat (Tensor | Tuple[Tensor]):
                If a torch.Tensor is given, it represents the input feature of shape
                :math:`(N, D_{in})`
                where :math:`D_{in}` is size of input feature, :math:`N` is the number
                of nodes. If a pair of torch.Tensor is given, the pair must contain t
                wo tensors of shape :math:`(N_{in}, D_{in_{src}})` and
                :math:`(N_{out}, D_{in_{dst}})`.
            edge_weight (optional, Tensor):
                Optional tensor on the edge. If given, the convolution will weight
                with regard to the message.

        Returns:
            outputs (Tensor):
                The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
                is size of output feature.
        """
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_src("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(self._aggre_type)
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
            else:
                rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
