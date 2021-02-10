# Reference: https://github.com/yourh/AttentionXML/blob/master/deepxml/networks.py

"""
Created on 2020/12/31
@author Sangwoo Han
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaModel,
    RobertaPreTrainedModel,
)

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
        num_labels: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        linear_size: List[int],
        dropout: float,
        max_length: int,
        **kwargs
    ):
        super(AttentionRNN, self).__init__(emb_size, **kwargs)
        self.batch_m = nn.BatchNorm1d(max_length)
        self.lstm = LSTMEncoder(emb_size, hidden_size, num_layers, dropout)
        self.attention = MLAttention(num_labels, hidden_size * 2)
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

        emb_out = self.batch_m(emb_out)

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
    ):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.batch_m = nn.BatchNorm1d(config.max_length)
        self.attention = MLAttention(self.num_labels, config.hidden_size)
        self.linear = MLLinear([config.hidden_size] + linear_size, 1)

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
        # if return_emb:
        #     return (self.get_input_embeddings()(input_ids),)

        # if not pass_emb:
        #     outputs = self.roberta(
        #         input_ids,
        #         attention_mask=attention_mask,
        #         token_type_ids=token_type_ids,
        #         position_ids=position_ids,
        #         head_mask=head_mask,
        #         inputs_embeds=inputs_embeds,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        #     sequence_output = self.batch_m(outputs[0])
        # else:
        #     inputs_embeds, attention_mask = outputs[0], outputs[1]
        #     outputs = self.roberta(
        #         attention_mask=attention_mask,
        #         token_type_ids=token_type_ids,
        #         position_ids=position_ids,
        #         head_mask=head_mask,
        #         inputs_embeds=inputs_embeds,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        #     sequence_output = self.batch_m(outputs[0])

        attn_out = self.attention(
            sequence_output[:, 1:, :], attention_mask[:, 1:].bool()
        )  # N, labels_num, hidden_size

        logits = self.linear(attn_out)

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
