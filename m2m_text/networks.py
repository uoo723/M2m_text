# Reference: https://github.com/yourh/AttentionXML/blob/master/deepxml/networks.py

"""
Created on 2020/12/31
@author Sangwoo Han
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock
from dgl.nn import GATConv, GINConv
from sentence_transformers import SentenceTransformer, models
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaModel,
    RobertaPreTrainedModel,
)

from .datasets._base import Dataset
from .modules import (
    CNNLayer,
    CorNet,
    Embedding,
    GateAttention,
    GCNLayer,
    Identity,
    LabelEmbedding,
    LSTMEncoder,
    MLAttention,
    MLAttention2,
    MLAttentionForSBert,
    MLLinear,
    MultiHeadAttention,
    Readout,
    Residual,
)
from .roberta import RobertaModel4Mix
from .utils.graph import MultiLayerNeighborSampler, get_ease_weight, stack_embed


class Network(nn.Module):
    def __init__(
        self,
        emb_size: int,
        vocab_size: Optional[int] = None,
        emb_init: Optional[int] = None,
        emb_trainable: bool = True,
        padding_idx: int = 0,
        emb_dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.emb = Embedding(
            vocab_size, emb_size, emb_init, emb_trainable, padding_idx, emb_dropout
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class AttentionRNN(Network):
    def __init__(
        self,
        num_labels: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        linear_size: List[int],
        dropout: float,
        mp_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(emb_size, **kwargs)
        # self.batch_m = nn.BatchNorm1d(max_length)
        # self.batch_m2 = nn.BatchNorm1d(num_labels)
        self.lstm = LSTMEncoder(emb_size, hidden_size, num_layers, dropout)
        self.attention = MLAttention(num_labels, hidden_size * 2)
        self.linear = MLLinear([hidden_size * 2] + linear_size, 1)
        self.mp_enabled = mp_enabled

    def forward(
        self,
        inputs,
        return_emb=False,
        pass_emb=False,
        return_hidden=False,
        pass_hidden=False,
        return_attn=False,
        pass_attn=False,
        return_attention_score=False,
        rnn_training=False,
        mp_enabled: Optional[bool] = None,
        **kwargs,
    ):
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        if return_emb and pass_emb:
            raise ValueError("`return_emb` and `pass_emb` both cannot be True")

        if return_hidden and pass_hidden:
            raise ValueError("`return_hidden` and `pass_hidden` both cannot be True")

        if return_attn and pass_attn:
            raise ValueError("`return_attn` and `pass_attn` both cannot be True")

        return_kwargs = {
            "return_emb": return_emb,
            "return_hidden": return_hidden,
            "return_attn": return_attn,
        }

        pass_kwargs = {
            "pass_emb": pass_emb,
            "pass_hidden": pass_hidden,
            "pass_attn": pass_attn,
        }

        for kw1, kw2 in combinations(return_kwargs.keys(), 2):
            if return_kwargs[kw1] and return_kwargs[kw2]:
                raise ValueError(f"`{kw1}` and `{kw2}` both cannot be True")

        for kw1, kw2 in combinations(pass_kwargs.keys(), 2):
            if pass_kwargs[kw1] and pass_kwargs[kw2]:
                raise ValueError(f"`{kw1}` and `{kw2}` both cannot be True")

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            if not pass_emb and not pass_hidden and not pass_attn:
                emb_out, lengths, masks = self.emb(inputs, **kwargs)
            elif pass_emb:
                emb_out, lengths, masks = inputs
            else:
                emb_out, lengths, masks = None, None, None

            if return_emb:
                return emb_out, lengths, masks

            if emb_out is not None:
                emb_out, masks = emb_out[:, : lengths.max()], masks[:, : lengths.max()]
                # emb_out = self.batch_m(emb_out)

                rnn_out = self.lstm(
                    emb_out, lengths, training=rnn_training
                )  # N, L, hidden_size * 2
            elif pass_hidden:
                rnn_out, lengths, masks = inputs
            else:
                rnn_out, lengths, masks = None, None, None

            if return_hidden:
                return rnn_out, lengths, masks

            if rnn_out is not None:
                attn_out = self.attention(
                    rnn_out,
                    masks,
                    return_attention_score,
                )  # N, labels_num, hidden_size * 2
            elif pass_attn:
                attn_out = inputs[0]
            else:
                attn_out = None

            if return_attn:
                return attn_out if type(attn_out) == tuple else (attn_out,)

            # attn_out = self.batch_m2(attn_out)

            if return_attention_score and type(attn_out) == tuple:
                return (self.linear(attn_out[0]), attn_out[1])
            else:
                return (self.linear(attn_out),)


class AttentionRNN4Mix(Network):
    def __init__(
        self,
        num_labels: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        linear_size: List[int],
        dropout: float,
        mp_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(emb_size, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, num_layers, dropout)
        self.attention = MLAttention(num_labels, hidden_size * 2)
        self.linear = MLLinear([hidden_size * 2] + linear_size, 1)
        self.mp_enabled = mp_enabled

    def forward(
        self,
        inputs: torch.Tensor,
        trace_grad: bool = False,
        mp_enabled: Optional[bool] = None,
        **kwargs,
    ):
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            emb_out, lengths, masks = self.emb(inputs, **kwargs)

            if trace_grad:
                emb_out = emb_out.detach().requires_grad_(True)

            rnn_out = self.lstm(emb_out, lengths)
            attn_out = self.attention(rnn_out, masks)
            linear_out = self.linear(attn_out)

            return (linear_out, emb_out)


class AttentionRNNEncoder(Network):
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_heads: int,
        mp_enabled: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(emb_size, *args, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, num_layers, dropout)
        self.attn = MultiHeadAttention(hidden_size * 2, hidden_size * 2, num_heads)

        self.mp_enabled = mp_enabled

    def forward(
        self,
        inputs,
        mp_enabled: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            emb_out, lengths, masks = self.emb(inputs, *args, **kwargs)
            emb_out, masks = emb_out[:, : lengths.max()], masks[:, : lengths.max()]
            outputs = self.attn(self.lstm(emb_out, lengths), masks)

        return (outputs,)


class RNNEncoder(Network):
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        mp_enabled: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(emb_size, *args, **kwargs)
        self.lstm = LSTMEncoder(emb_size, hidden_size, num_layers, dropout)
        self.mp_enabled = mp_enabled
        self.hidden_size = hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
        mp_enabled: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            emb_out, lengths, masks = self.emb(inputs, *args, **kwargs)
            emb_out, masks = emb_out[:, : lengths.max()], masks[:, : lengths.max()]
            outputs = self.lstm(emb_out, lengths)

        return outputs, masks


class AttentionRNN2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        linear_size: List[int],
        mp_enabled: bool = False,
    ) -> Tuple[torch.Tensor]:
        super().__init__()

        self.attn = MLAttention(num_labels, hidden_size)
        self.linear = MLLinear([hidden_size] + linear_size, 1)
        self.mp_enabled = mp_enabled

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        mp_enabled: Optional[bool] = None,
    ):
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            outputs = self.linear(self.attn(inputs, masks))

        return (outputs,)


class AttentionRNNEncoder2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mp_enabled: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, hidden_size, num_heads)
        self.mp_enabled = mp_enabled

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        mp_enabled: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            outputs = self.attn(inputs, masks)

        return (outputs,)


class AttentionRNNEncoder3(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        linear_size: List[int],
        output_linear_size: List[int],
        mp_enabled: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, hidden_size, num_heads)
        self.linear = MLLinear([hidden_size] + linear_size[:-1], linear_size[-1])
        self.output_linear = MLLinear(
            [hidden_size + linear_size[-1]] + output_linear_size[:-1],
            output_linear_size[-1],
        )
        self.mp_enabled = mp_enabled

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        mp_enabled: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            hidden = self.attn(inputs, masks)
            outputs = self.linear(hidden)
            outputs2 = self.output_linear(torch.cat([hidden, outputs], dim=-1))

        return (outputs2, outputs)


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
        num_labels: int,
        encoder_linear_size: List[int],
        linear_size: List[int],
        input_size: Optional[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.encoder_linear_size = encoder_linear_size
        self.linear_size = linear_size
        self.input_size = input_size
        self.dropout_p = dropout
        self._built = False

        if input_size:
            self._build()

    def _build(self):
        input_size = self.input_size
        encoder_linear_size = self.encoder_linear_size
        linear_size = self.linear_size
        num_labels = self.num_labels
        dropout = self.dropout_p

        self.encoder = MLLinear(
            [input_size] + encoder_linear_size[:-1], encoder_linear_size[-1]
        )
        self.linear = MLLinear(encoder_linear_size[-1:] + linear_size, num_labels)
        self.dropout = nn.Dropout(dropout)

        self._built = True

    def forward(
        self,
        inputs: torch.Tensor,
        return_emb: bool = False,
        pass_emb: bool = False,
    ) -> torch.Tensor:
        """Forward inputs

        Args:
            inputs (torch.Tensor): Input tensor with shape (batch_size, input_size).
            return_emb (bool): Return hidden if true. default: False.
            pass_emb (bool): Treat inputs tensor as a hidden if true. default: False.

        Returns:
            outputs (torch.Tensor): Output tensor.
        """
        if not self._built:
            self.input_size = inputs.size(-1)
            self._build()

        if return_emb and pass_emb:
            raise ValueError("`return_hidden` and `pass_hidden` both cannot be True")

        if return_emb:
            return self.encoder(inputs)

        if pass_emb:
            return self.linear(self.dropout(F.relu(inputs)))

        outputs = self.encoder(inputs)
        return self.linear(self.dropout(F.relu(outputs)))

    def init_linear(self):
        """Initialized linear"""
        self.linear.init_weights()


class RobertaForSeqClassification(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config: PretrainedConfig, freeze_encoder: bool = False):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.batch_m = nn.BatchNorm1d(config.max_length)
        self.classifier = RobertaClassificationHead(config)

        if freeze_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_emb=False,
        pass_emb=False,
        outputs=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if return_emb and pass_emb:
            raise ValueError("`return_hidden` and `pass_hidden` cannot be both true.")

        if not pass_emb:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs[0]

        if return_emb:
            return (sequence_output, *outputs[1:])

        sequence_output = self.batch_m(sequence_output)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LaRoberta(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(
        self,
        config: PretrainedConfig,
        linear_size: List[int],
        freeze_encoder: bool = False,
        mp_enabled: bool = False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.batch_m = nn.BatchNorm1d(config.max_length)
        # self.batch_m2 = nn.BatchNorm1d(config.num_labels)
        self.attention = MLAttention(self.num_labels, config.hidden_size)
        self.linear = MLLinear([config.hidden_size] + linear_size, 1)

        self.mp_enabled = mp_enabled

        if freeze_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.init_weights()

    def forward(
        self,
        inputs=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        return_emb=False,
        pass_emb=False,
        return_attn=False,
        pass_attn=False,
        outputs=None,
        return_attention_score=False,
        mp_enabled: Optional[bool] = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        if inputs is not None and type(inputs) == dict:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
        else:
            input_ids = None
            attention_mask = None

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if return_emb and pass_emb:
            raise ValueError("`return_hidden` and `pass_hidden` cannot be both true.")

        sequence_output, attn_out = None, None

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            if not pass_emb and not pass_attn:
                outputs = self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                sequence_output = outputs[0]
            elif pass_emb:
                sequence_output, attention_mask = inputs[0], inputs[1]
            else:
                attn_out = inputs[0]

            if return_emb:
                return (sequence_output, attention_mask, *outputs[1:])

            # if sequence_output is not None:
            #     sequence_output = self.batch_m(sequence_output)

            if attn_out is None:
                attn_out = self.attention(
                    sequence_output,
                    attention_mask.bool(),
                    return_attention_score,
                )  # N, labels_num, hidden_size

            if return_attn:
                return attn_out if type(attn_out) == tuple else (attn_out,)

            # attn_out = self.batch_m2(attn_out)

            logits = (
                self.linear(attn_out[0])
                if type(attn_out) == tuple
                else self.linear(attn_out)
            )

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            if return_attention_score and type(attn_out) == tuple:
                output += (attn_out[1],)

            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LaRoberta4Mix(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(
        self,
        config: PretrainedConfig,
        linear_size: List[int],
        freeze_encoder: bool = False,
        mp_enabled: bool = False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel4Mix(config, add_pooling_layer=False)
        self.attention = MLAttention(self.num_labels, config.hidden_size)
        self.linear = MLLinear([config.hidden_size] + linear_size, 1)

        self.mp_enabled = mp_enabled

        if freeze_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.init_weights()

    def forward(
        self,
        inputs=None,
        inputs2=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        trace_grad=False,
        mix_lambda=None,
        mix_layer=None,
        mix_embedding=False,
        return_attention_score=False,
        mp_enabled: Optional[bool] = None,
    ):
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        if inputs is not None and type(inputs) == dict:
            attention_mask = inputs["attention_mask"]

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            outputs = self.roberta(
                inputs=inputs,
                inputs2=inputs2,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                trace_grad=trace_grad,
                mix_lambda=mix_lambda,
                mix_layer=mix_layer,
                mix_embedding=mix_embedding,
            )
            sequence_output = outputs[0]
            attn_out = self.attention(
                sequence_output, attention_mask.bool(), return_attention_score
            )
            logits = (
                self.linear(attn_out[0])
                if type(attn_out) == tuple
                else self.linear(attn_out)
            )

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            if return_attention_score and type(attn_out) == tuple:
                output += (attn_out[1],)
            output += outputs[1:]

            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LaRobertaV2(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(
        self,
        config: PretrainedConfig,
        linear_size: List[int],
        freeze_encoder: bool = False,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.batch_m = nn.BatchNorm1d(config.max_length)
        self.attention = MLAttention(self.num_labels, config.hidden_size)
        self.linear = MLLinear([config.hidden_size] + linear_size, 1)
        self.classifier = RobertaClassificationHead(config)

        if freeze_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_emb=False,
        pass_emb=False,
        outputs=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if return_emb and pass_emb:
            raise ValueError("`return_hidden` and `pass_hidden` cannot be both true.")

        if not pass_emb:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
        else:
            sequence_output, attention_mask = outputs[0], outputs[1]

        if return_emb:
            return (sequence_output, *outputs[1:])

        sequence_output = self.batch_m(sequence_output)

        attn_out = self.attention(
            sequence_output[:, 1:, :], attention_mask[:, 1:].bool()
        )  # N, labels_num, hidden_size

        logits1 = self.linear(attn_out)
        logits2 = self.classifier(sequence_output)
        logits = logits1 + logits2

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AttentionRGCN(Network):
    def __init__(
        self,
        num_labels: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        linear_size: List[int],
        gcn_hidden_size: List[int],
        dropout: float,
        max_length: int,
        init_adj: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super(AttentionRGCN, self).__init__(emb_size, **kwargs)
        self.batch_m = nn.BatchNorm1d(max_length)
        self.batch_m2 = nn.BatchNorm1d(num_labels)
        self.lstm = LSTMEncoder(emb_size, hidden_size, num_layers, dropout)
        self.attention = MLAttention(num_labels, hidden_size * 2)
        self.gcl = GCNLayer(
            num_labels, [hidden_size * 2] + gcn_hidden_size, dropout, init_adj
        )
        self.linear = MLLinear([gcn_hidden_size[-1] + hidden_size * 2] + linear_size, 1)

    def forward(
        self,
        inputs,
        return_emb=False,
        pass_emb=False,
        return_hidden=False,
        pass_hidden=False,
        return_attn=False,
        pass_attn=False,
        rnn_training=False,
        **kwargs,
    ):
        if return_emb and pass_emb:
            raise ValueError("`return_emb` and `pass_emb` both cannot be True")

        if return_hidden and pass_hidden:
            raise ValueError("`return_hidden` and `pass_hidden` both cannot be True")

        if return_attn and pass_attn:
            raise ValueError("`return_attn` and `pass_attn` both cannot be True")

        return_kwargs = {
            "return_emb": return_emb,
            "return_hidden": return_hidden,
            "return_attn": return_attn,
        }

        pass_kwargs = {
            "pass_emb": pass_emb,
            "pass_hidden": pass_hidden,
            "pass_attn": pass_attn,
        }

        for kw1, kw2 in combinations(return_kwargs.keys(), 2):
            if return_kwargs[kw1] and return_kwargs[kw2]:
                raise ValueError(f"`{kw1}` and `{kw2}` both cannot be True")

        for kw1, kw2 in combinations(pass_kwargs.keys(), 2):
            if pass_kwargs[kw1] and pass_kwargs[kw2]:
                raise ValueError(f"`{kw1}` and `{kw2}` both cannot be True")

        if not pass_emb and not pass_hidden and not pass_attn:
            emb_out, lengths, masks = self.emb(inputs, **kwargs)
        elif pass_emb:
            emb_out, lengths, masks = inputs
        else:
            emb_out, lengths, masks = None, None, None

        if return_emb:
            return emb_out, lengths, masks

        if emb_out is not None:
            emb_out, masks = emb_out[:, : lengths.max()], masks[:, : lengths.max()]
            emb_out = self.batch_m(emb_out)

            rnn_out = self.lstm(
                emb_out, lengths, training=rnn_training
            )  # N, L, hidden_size * 2
        elif pass_hidden:
            rnn_out, lengths, masks = inputs
        else:
            rnn_out, lengths, masks = None, None, None

        if return_hidden:
            return rnn_out, lengths, masks

        if rnn_out is not None:
            attn_out = self.attention(rnn_out, masks)  # N, labels_num, hidden_size * 2
        elif pass_attn:
            attn_out = inputs
        else:
            attn_out = None

        if return_attn:
            return (attn_out,)

        attn_out = self.batch_m2(attn_out)
        gcn_out = self.gcl(attn_out)
        outputs = torch.cat(
            [attn_out, gcn_out.expand(attn_out.shape[0], *gcn_out.shape)], dim=-1
        )

        return self.linear(outputs)


class CornetAttentionRNN(AttentionRNN):
    def __init__(
        self,
        num_labels: int,
        cor_context_size: List[int],
        *args,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, *args, **kwargs)
        self.cornet = CorNet(num_labels, cor_context_size)

    def forward(self, *args, **kwargs):
        ret = super().forward(*args, **kwargs)
        if type(ret) == tuple:
            return ret

        return self.cornet(ret)


class CornetAttentionRNNv2(AttentionRNN):
    def __init__(
        self,
        num_labels: int,
        cor_context_size: List[int],
        n_cor_nets: int,
        *args,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, *args, **kwargs)
        self.cornet = nn.ModuleList(
            [CorNet(num_labels, cor_context_size) for _ in range(n_cor_nets)]
        )

    def forward(self, *args, return_raw: bool = False, **kwargs):
        ret = super().forward(*args, **kwargs)
        if type(ret) == tuple:
            return ret

        if return_raw:
            return ret

        outputs = torch.stack([layer(ret) for layer in self.cornet]).sum(dim=0)

        return outputs


class EaseAttentionRNN(AttentionRNN):
    def __init__(
        self,
        num_labels: int,
        dataset: Dataset,
        lamda: float = 50,
        random_init: bool = True,
        adj_trainable: bool = True,
        add_skip_connection: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, *args, **kwargs)
        self.B = nn.Linear(num_labels, num_labels, bias=False)
        self.add_skip_connection = add_skip_connection

        if random_init:
            nn.init.xavier_uniform_(self.B.weight)
        else:
            B = get_ease_weight(dataset, lamda)
            self.B.weight.data = torch.from_numpy(B).float()

        self.B.requires_grad_ = adj_trainable

    def forward(self, *args, **kwargs):
        ret = super().forward(*args, **kwargs)
        if type(ret) == tuple:
            return ret

        outputs = self.B(ret)

        if self.add_skip_connection:
            outputs = F.relu(outputs)
            outputs = outputs + ret

        return outputs


class LabelGCNAttentionRNN(AttentionRNN):
    """
    Last layer of GCN is readout
    """

    def __init__(
        self,
        num_labels: int,
        hidden_size: int,
        gcn_hidden_size: List[int],
        readout_linear_size: List[int],
        gcn_dropout: float,
        gcn_init_adj: Optional[torch.Tensor] = None,
        gcn_adj_trainable: bool = False,
        enable_gating: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels,
            hidden_size=hidden_size,
            *args,
            **kwargs,
        )

        self.gcl = GCNLayer(
            num_labels,
            [hidden_size * 2] + gcn_hidden_size,
            gcn_dropout,
            gcn_init_adj,
            gcn_adj_trainable,
        )

        self.readout = Readout(gcn_hidden_size[-1:] + readout_linear_size, num_labels)
        self.enable_gating = enable_gating
        if enable_gating:
            self.gate = GateAttention(num_labels, 2)

    def forward(self, *args, **kwargs):
        attn_out = super().forward(return_attn=True, *args, **kwargs)[0]
        outputs1 = self.readout(self.gcl(attn_out))
        outputs2 = self.linear(attn_out)

        if self.enable_gating:
            outputs = self.gate(torch.stack([outputs1, outputs2], dim=1)).sum(dim=1)
        else:
            outputs = outputs1 + outputs2

        return outputs


class LabelGCNAttentionRNNv2(AttentionRNN):
    """
    Last layer of GCN is MLLinear
    """

    def __init__(
        self,
        num_labels: int,
        hidden_size: int,
        gcn_hidden_size: List[int],
        gcl_linear_size: List[int],
        gcn_dropout: float,
        gcn_init_adj: Optional[torch.Tensor] = None,
        gcn_adj_trainable: bool = False,
        enable_gating: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels,
            hidden_size=hidden_size,
            *args,
            **kwargs,
        )

        self.gcl = GCNLayer(
            num_labels,
            [hidden_size * 2] + gcn_hidden_size,
            gcn_dropout,
            gcn_init_adj,
            gcn_adj_trainable,
        )

        self.gcl_linear = MLLinear(gcn_hidden_size[-1:] + gcl_linear_size, 1)
        self.enable_gating = enable_gating
        if enable_gating:
            self.gate = GateAttention(num_labels, 2)

    def forward(self, *args, **kwargs):
        attn_out = super().forward(return_attn=True, *args, **kwargs)[0]
        outputs1 = self.gcl_linear(self.gcl(attn_out))
        outputs2 = self.linear(attn_out)

        if self.enable_gating:
            outputs = self.gate(torch.stack([outputs1, outputs2], dim=1)).sum(dim=1)
        else:
            outputs = outputs1 + outputs2

        return outputs


class LabelGCNAttentionRNNv3(AttentionRNN):
    def __init__(
        self,
        num_labels: int,
        hidden_size: int,
        linear_size: List[int],
        gcn_hidden_size: List[int],
        gcn_dropout: float,
        gcn_init_adj: Optional[torch.Tensor] = None,
        gcn_adj_trainable: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels,
            hidden_size=hidden_size,
            linear_size=linear_size,
            *args,
            **kwargs,
        )

        self.gcl = GCNLayer(
            num_labels,
            [hidden_size * 2] + gcn_hidden_size,
            gcn_dropout,
            gcn_init_adj,
            gcn_adj_trainable,
        )

        self.linear = MLLinear(gcn_hidden_size[-1:] + linear_size, 1)

    def forward(self, *args, **kwargs):
        attn_out = super().forward(return_attn=True, *args, **kwargs)[0]
        gcn_out = self.gcl(attn_out)
        return self.linear(gcn_out)


class LabelGCNAttentionRNNv4(AttentionRNN):
    def __init__(
        self,
        num_labels: int,
        hidden_size: int,
        linear_size: List[int],
        label_emb_size: int,
        gcn_hidden_size: List[int],
        gcn_dropout: float,
        gcn_init_adj: Optional[torch.Tensor] = None,
        gcn_adj_trainable: bool = False,
        gcn_adj_dropout: Optional[float] = None,
        label_emb_init: Optional[np.ndarray] = None,
        use_gat: bool = False,
        gat_num_heads: int = 3,
        enable_gating: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            num_labels=num_labels,
            hidden_size=hidden_size,
            linear_size=linear_size,
            *args,
            **kwargs,
        )

        self.enable_gating = enable_gating

        self.init_label_emb(num_labels, label_emb_size, label_emb_init)

        if use_gat:
            assert gcn_init_adj is not None, "gcn_init_adj must be set when use GAT."
            gcl_hidden_size = [label_emb_size] + gcn_hidden_size
            self.g = dgl.from_scipy(sp.csr_matrix(gcn_init_adj.numpy()))
            # self.g = dgl.add_self_loop(g)
            self.gcl = nn.ModuleList(
                GATConv(
                    in_s,
                    out_s,
                    gat_num_heads,
                    feat_drop=gcn_dropout,
                    residual=True,
                    activation=F.relu,
                    allow_zero_in_degree=True,
                )
                for in_s, out_s in zip(gcl_hidden_size[:-1], gcl_hidden_size[1:])
            )
        else:
            self.gcl = GCNLayer(
                num_labels,
                [label_emb_size] + gcn_hidden_size,
                gcn_dropout,
                gcn_init_adj,
                gcn_adj_trainable,
                gcn_adj_dropout,
            )

        if enable_gating:
            assert (
                gcn_hidden_size[-1] == hidden_size * 2
            ), "Last hidden size of GCN and first hidden size of MLP must be same."
            self.gate = nn.Linear(hidden_size * 4, 2, bias=False)
            nn.init.xavier_normal_(self.gate.weight)
        else:
            self.register_parameter("gate", None)
            self.linear = MLLinear(
                [hidden_size * 2 + gcn_hidden_size[-1]] + linear_size, 1
            )

    def init_label_emb(self, num_labels, label_emb_size, label_emb_init):
        self.label_emb = nn.Parameter(torch.FloatTensor(num_labels, label_emb_size))

        if label_emb_init is not None:
            assert (
                label_emb_init.shape[1] == label_emb_size
            ), "Mismatching of dimension of label embedding"

            self.label_emb.data = torch.from_numpy(label_emb_init).float()
        else:
            nn.init.xavier_normal_(self.label_emb.data)

    def forward(self, *args, **kwargs):
        attn_out = super().forward(return_attn=True, *args, **kwargs)[0]

        if isinstance(self.gcl, nn.ModuleList):
            gcn_out = self.label_emb
            if attn_out.device != self.g:
                self.g = self.g.to(attn_out.device)

            for gcl in self.gcl:
                gcn_out = gcl(self.g, gcn_out).mean(dim=1)

            gcn_out = gcn_out.unsqueeze(0)
        else:
            gcn_out = self.gcl(self.label_emb).unsqueeze(0)

        repeat_vals = [attn_out.shape[0] // gcn_out.shape[0]] + [-1] * (
            len(gcn_out.shape) - 1
        )

        gcn_out = gcn_out.expand(*repeat_vals)

        if self.enable_gating:
            weights = F.softmax(
                self.gate(torch.cat([attn_out, gcn_out], dim=-1)), dim=-1
            )
            weights = weights.transpose(2, 1)
            outputs = (
                weights[:, 0, :].unsqueeze(-1) * attn_out
                + weights[:, 1, :].unsqueeze(-1) * gcn_out
            )
        else:
            outputs = torch.cat([attn_out, gcn_out], dim=-1)

        return self.linear(outputs)


class LabelGCNAttentionRNNv5(nn.Module):
    def __init__(
        self,
        num_labels: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        linear_size: List[int],
        dropout: float,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.attention_rnn = AttentionRNNEncoder(
            emb_size,
            hidden_size,
            num_layers,
            dropout,
            *args,
            **kwargs,
        )

        self.label_emb = Embedding(
            num_labels, emb_size=hidden_size * 2, padding_idx=None
        )

        self.linear = MLLinear(linear_size[-1], num_labels)

        linear_size = [hidden_size * 2] + linear_size + [hidden_size * 2]

        self.conv = nn.ModuleList(
            GINConv(nn.Linear(in_s, out_s), "sum", learn_eps=True)
            for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
        )

        self.residual = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                Residual(linear_size[-1], linear_size[-1], dropout),
            )
            for _ in range(len(linear_size) - 1)
        )

    def forward(
        self,
        graph: Union[dgl.DGLGraph, List[DGLBlock]],
        inputs: torch.Tensor,
        return_hidden: bool = False,
    ):
        hidden_list = []

        if isinstance(graph, list):
            blocks = graph
        else:
            blocks = [g for g in range(self.conv_list)]

        h = inputs

        for conv, residual, block in zip(self.conv, self.residual, blocks):
            h = conv(block, h)
            h = residual(h)
            hidden_list.append(h)

        stacked_hidden = stack_embed(
            blocks, blocks[-1].dstdata["_ID"], hidden_list
        )  # N x num_layer x hidden_size
        outputs = stacked_hidden.mean(dim=1)
        outputs = self.linear(outputs)

        ret = (outputs,)

        if return_hidden:
            ret += (stacked_hidden,)

        return ret


# class LabelGCNAttentionRNNv5(AttentionRNN):
#     """Do not use"""
#     def __init__(
#         self,
#         num_labels: int,
#         hidden_size: int,
#         linear_size: List[int],
#         label_emb_size: int,
#         gcn_hidden_size: List[int],
#         gcn_dropout: float,
#         gcn_init_adj: Optional[torch.Tensor] = None,
#         gcn_adj_trainable: bool = False,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(
#             num_labels=num_labels,
#             hidden_size=hidden_size,
#             linear_size=linear_size,
#             *args,
#             **kwargs,
#         )

#         self.label_emb = nn.Parameter(torch.FloatTensor(num_labels, label_emb_size))
#         nn.init.xavier_normal_(self.label_emb.data)

#         self.gcl = GCNLayer(
#             num_labels,
#             [label_emb_size] + gcn_hidden_size,
#             gcn_dropout,
#             gcn_init_adj,
#             gcn_adj_trainable,
#         )

#         self.linear2 = nn.Linear(num_labels, gcn_hidden_size[-1])

#     def forward(self, *args, **kwargs):
#         model_out = self.linear2(super().forward(*args, **kwargs))
#         gcn_out = self.gcl(self.label_emb)
#         return model_out @ gcn_out.transpose(1, 0)


class GraphXC(nn.Module):
    def __init__(
        self,
        num_labels: int,
        input_size: int,
        output_size: int,
        hidden_size: List[int],
        dropout: int = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        linear_size = [input_size] + hidden_size + [output_size]
        self.attention = MLAttention2(num_labels, output_size)
        self.linear = MLLinear([output_size], 1)

        self.conv_list = nn.ModuleList(
            GINConv(nn.Linear(in_s, out_s), "sum", learn_eps=True)
            for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
        )

        self.residual_list = nn.ModuleList(
            nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                Residual(output_size, output_size, dropout),
            )
            for _ in range(len(linear_size) - 1)
        )

    def forward(
        self,
        graph: Union[dgl.DGLGraph, List[DGLBlock]],
        inputs: torch.Tensor,
        return_hidden: bool = False,
    ):
        hidden_list = []

        if isinstance(graph, list):
            blocks = graph
        else:
            blocks = [g for g in range(self.conv_list)]

        h = inputs

        for conv, residual, block in zip(self.conv_list, self.residual_list, blocks):
            h = conv(block, h)
            h = residual(h)
            hidden_list.append(h)

        stacked_hidden = stack_embed(
            blocks, blocks[-1].dstdata["_ID"], hidden_list
        )  # N x num_layer x hidden_size
        outputs = self.attention(stacked_hidden)
        outputs = self.linear(outputs)

        ret = (outputs,)

        if return_hidden:
            ret += (stacked_hidden,)

        return ret


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: List[int]):
        super().__init__()

        self.encoder = MLLinear([input_size] + hidden_size[:-1], hidden_size[-1])
        self.decoder = MLLinear(hidden_size[::-1], input_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class LabelEncoder(nn.Module):
    def __init__(
        self,
        num_labels: int,
        linear_size: Optional[List[int]] = None,
        output_size: Optional[int] = None,
        emb_size: Optional[int] = None,
        emb_init: Optional[Union[np.ndarray, str]] = None,
        emb_trainable: bool = True,
        dropout: bool = 0.2,
        mp_enabled: bool = False,
        enable_layer_norm: bool = True,
        output_layer: Optional[nn.Module] = None,
    ):
        super().__init__()

        assert not (
            bool(linear_size) ^ bool(output_size)
        ), "linear_size and output_size are both must be set or unset"

        self.mp_enabled = mp_enabled

        self.emb = LabelEmbedding(
            num_labels, emb_size, emb_init, emb_trainable, dropout
        )
        self.layer_norm = nn.LayerNorm(emb_size) if enable_layer_norm else Identity()
        self.linear = (
            MLLinear([emb_size] + linear_size, output_size, enable_layer_norm)
            if linear_size
            else Identity()
        )
        self.output_layer = Identity() if output_layer is None else output_layer

    def forward(
        self, inputs: torch.Tensor, mp_enabled: Optional[bool] = None
    ) -> torch.Tensor:
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            outputs = self.output_layer(self.linear(self.layer_norm(self.emb(inputs))))
        return outputs


class LabelGINEncoder(nn.Module):
    def __init__(
        self,
        graph: dgl.DGLGraph,
        num_labels: int,
        output_size: int,
        hidden_size: List[int],
        emb_size: Optional[int] = None,
        emb_init: Union[np.ndarray, str] = None,
        emb_trainable: bool = True,
        dropout: bool = 0.2,
        gin_aggregate_type: str = "sum",
        enable_residual: bool = True,
        fanouts: List[int] = [4, 3, 2],
        mp_enabled: bool = False,
        use_stack: bool = True,
        version: int = 1,
    ):
        super().__init__()
        self.graph = graph
        self.mp_enabled = mp_enabled
        self.use_stack = use_stack

        self.sampler = MultiLayerNeighborSampler(fanouts)

        self.emb = LabelEmbedding(
            num_labels, emb_size, emb_init, emb_trainable, dropout
        )

        linear_size = [self.emb.emb.embedding_dim] + hidden_size + [output_size]

        if version == 1:
            assert len(linear_size) - 1 == len(
                fanouts
            ), "# of GIN layers and # of fanouts must be same."

        if use_stack:
            self.attention = MLAttention2(1, output_size)
        else:
            self.attention = self.register_parameter("attention", None)

        if version == 1:
            self.conv_list = nn.ModuleList(
                GINConv(nn.Linear(in_s, out_s), gin_aggregate_type, learn_eps=True)
                for in_s, out_s in zip(linear_size[:-1], linear_size[1:])
            )
        else:
            self.conv_list = nn.ModuleList(
                GINConv(
                    MLLinear(linear_size[:-1], linear_size[-1]),
                    gin_aggregate_type,
                    learn_eps=True,
                )
                for _ in range(len(fanouts))
            )

        if enable_residual:
            if version == 1:
                self.residual_list = nn.ModuleList(
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        Residual(output_size, output_size, dropout),
                    )
                    for _ in range(len(linear_size) - 1)
                )
            else:
                self.residual_list = nn.ModuleList(
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        Residual(output_size, output_size, dropout),
                    )
                    for _ in range(len(fanouts))
                )
        else:
            if version == 1:
                self.residual_list = nn.ModuleList(
                    Identity() for _ in range(len(linear_size) - 1)
                )
            else:
                self.residual_list = nn.ModuleList(
                    Identity() for _ in range(len(fanouts))
                )

    def forward(
        self, inputs: torch.Tensor, mp_enabled: Optional[bool] = None
    ) -> torch.Tensor:
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        device = inputs.device
        hidden_list = []

        if len(inputs.size()) == 2:
            input_size = inputs.size()
            inputs = inputs.flatten()
        else:
            input_size = None

        inputs = inputs.cpu()

        input_ids, counts = torch.unique_consecutive(
            inputs[inputs.argsort()], return_counts=True
        )

        blocks = self.sampler.sample_blocks(self.graph, input_ids)

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            h = self.emb(blocks[0].srcdata["_ID"].to(device))

            for conv, residual, block in zip(
                self.conv_list, self.residual_list, blocks
            ):
                h = conv(block.to(device), h)
                h = residual(h)
                hidden_list.append(h)

            if self.use_stack:
                stacked_hidden = stack_embed(
                    blocks, blocks[-1].dstdata["_ID"], hidden_list
                )  # N x num_layer x hidden_size

                outputs = self.attention(stacked_hidden).squeeze()
            else:
                outputs = hidden_list[-1]

            outputs = outputs[np.repeat(np.arange(counts.shape[0]), counts)][
                inputs.argsort()
            ]

            if input_size is not None:
                outputs = outputs.view(*input_size, -1)

        return outputs


class SBert(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: Optional[int] = None,
        linear_size: Optional[List[int]] = None,
        mp_enabled: bool = False,
        max_seq_length: int = 150,
        pooling_mode: Optional[str] = None,
        output_layer: Optional[nn.Module] = None,
    ):
        super().__init__()

        if pooling_mode is not None:
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ["mean", "max", "cls", "att"]

        assert not (
            bool(num_labels) ^ bool(linear_size)
        ), "num_labels and linear_size are both must be set or unset"

        self.mp_enabled = mp_enabled

        word_embedding_model = models.Transformer(
            model_name, max_seq_length=max_seq_length
        )

        embedding_size = word_embedding_model.get_word_embedding_dimension()

        if pooling_mode == "att":
            pooling_model = MLAttentionForSBert(1, embedding_size)
        else:
            pooling_model = models.Pooling(
                embedding_size,
                pooling_mode=pooling_mode,
            )

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.linear = (
            MLLinear([embedding_size] + linear_size, num_labels)
            if num_labels is not None
            else None
        )
        self.output_layer = Identity() if output_layer is None else output_layer

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_linear: bool = False,
        mp_enabled: Optional[bool] = None,
    ):
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            model_outputs = self.model(inputs)
            outputs = (
                self.output_layer(model_outputs["sentence_embedding"]),
                model_outputs,
            )

            if return_linear:
                outputs = outputs + (self.linear(model_outputs),)

        return outputs

    def tokenize(self, *args, **kwargs):
        return self.model.tokenize(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)


class Pooling(nn.Module):
    def __init__(
        self,
        pooling_mode: str = "mean",
        hidden_size: Optional[int] = None,
        linear_size: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        assert pooling_mode in ["max", "sum", "mean", "att"]
        self.pooling_mode = pooling_mode

        # if pooling_mode == 'max':

        if pooling_mode == "att":
            assert hidden_size is not None
            self.attn = MLAttention(1, hidden_size)
        else:
            self.attn = self.register_parameter("attn", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "max":
            outputs = torch.max(inputs, dim=0, keepdim=True)[0]
        elif self.pooling_mode == "sum":
            outputs = torch.sum(inputs, dim=0, keepdim=True)
        elif self.pooling_mode == "mean":
            outputs = torch.mean(inputs, dim=0, keepdim=True)
        elif self.pooling_mode == "att":
            outputs = self.attn(
                inputs.unsqueeze(0),
                torch.tensor([[True] * inputs.shape[0]]).to(inputs.device),
            ).squeeze(1)
        else:
            raise AssertionError

        return outputs


class LaCNN(Network):
    """Implementation of modified CNN Kim

    Appended Label-wise attention after CNN outputs.
    """

    def __init__(
        self,
        num_labels: int,
        emb_size: int,
        linear_size: List[int],
        num_filters: int,
        filter_sizes: List[int],
        seq_len: int,
        cnn_dropout: float = 0.2,
        stride: int = 1,
        pooling_type: str = "max",
        mp_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(emb_size, **kwargs)
        self.cnn = CNNLayer(
            num_filters,
            filter_sizes,
            seq_len,
            emb_size,
            stride,
            pooling_type,
            cnn_dropout,
        )
        self.attention = MLAttention(num_labels, num_filters)
        self.linear = MLLinear([num_filters] + linear_size, 1)
        self.num_conv = len(filter_sizes)
        self.mp_enabled = mp_enabled

    def forward(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        return_emb: bool = False,
        pass_emb: bool = False,
        return_attn: bool = False,
        pass_attn: bool = False,
        mp_enabled: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        if mp_enabled is None:
            mp_enabled = self.mp_enabled

        if return_emb and pass_emb:
            raise ValueError("`return_emb` and `pass_emb` both cannot be True")

        if return_attn and pass_attn:
            raise ValueError("`return_attn` and `pass_attn` both cannot be True")

        return_kwargs = {
            "return_emb": return_emb,
            "return_attn": return_attn,
        }

        pass_kwargs = {
            "pass_emb": pass_emb,
            "pass_attn": pass_attn,
        }

        for kw1, kw2 in combinations(return_kwargs.keys(), 2):
            if return_kwargs[kw1] and return_kwargs[kw2]:
                raise ValueError(f"`{kw1}` and `{kw2}` both cannot be True")

        for kw1, kw2 in combinations(pass_kwargs.keys(), 2):
            if pass_kwargs[kw1] and pass_kwargs[kw2]:
                raise ValueError(f"`{kw1}` and `{kw2}` both cannot be True")

        with torch.cuda.amp.autocast(enabled=mp_enabled):
            if not pass_emb and not pass_attn:
                emb_out, _, _ = self.emb(inputs, **kwargs)
            elif pass_emb:
                emb_out = inputs[0]
            else:
                emb_out = None

            if return_emb:
                return (emb_out,)

            if emb_out is not None:
                cnn_out = self.cnn(emb_out)
            else:
                cnn_out = None

            if cnn_out is not None:
                masks = torch.ones(cnn_out.size(0), self.num_conv, dtype=torch.bool).to(
                    cnn_out.device
                )
                attn_out = self.attention(cnn_out, masks)
            elif pass_attn:
                attn_out = inputs[0]
            else:
                attn_out = None

            if return_attn:
                return (attn_out,)

            return (self.linear(attn_out),)
