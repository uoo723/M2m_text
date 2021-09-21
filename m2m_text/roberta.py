"""
Created on 2021/09/20
@author Sangwoo Han
@reference https://github.com/clovaai/ssmix
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaLayer,
    RobertaPooler,
    RobertaPreTrainedModel,
)


# Copied from transformers.model.roberta.modeling_roberta.RobertaModel
class RobertaModel4Mix(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder4Mix(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        inputs=None,
        inputs2=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        trace_grad=False,
        mix_lambda=None,
        mix_layer=None,
        mix_embedding=False,
    ):
        """Copied from https://github.com/clovaai/ssmix/blob/master/classification_model.py"""
        if inputs.get("token_type_ids") is None:
            inputs["token_type_ids"] = None

        if inputs2 is not None and inputs2.get("token_type_ids") is None:
            inputs2["token_type_ids"] = None

        input_ids, attention_mask, token_type_ids = (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs2 is not None:
            input_ids2, attention_mask2, token_type_ids2 = (
                inputs2["input_ids"],
                inputs2["attention_mask"],
                inputs2["token_type_ids"],
            )
            input_shape2 = input_ids2.size()
            if attention_mask2 is None:
                attention_mask2 = torch.ones(input_shape2, device=device)
            if token_type_ids2 is None:
                token_type_ids = torch.zeros(
                    input_shape2, dtype=torch.long, device=device
                )
            embedding_output2 = self.embeddings(
                input_ids=input_ids2,
                position_ids=position_ids,
                token_type_ids=token_type_ids2,
                inputs_embeds=inputs_embeds,
            )
            extended_attention_mask2: torch.Tensor = self.get_extended_attention_mask(
                attention_mask2, input_shape2, device
            )
        else:
            embedding_output2, extended_attention_mask2 = None, None

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if trace_grad:
            embedding_output = embedding_output.detach().requires_grad_(True)

        if mix_embedding:
            assert mix_layer is None
            embedding_output = (
                mix_lambda * embedding_output + (1 - mix_lambda) * embedding_output2
            )
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask | attention_mask2, input_shape, device
            )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            hidden_states2=embedding_output2,
            attention_mask2=extended_attention_mask2,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mix_lambda=mix_lambda,
            mix_layer=mix_layer,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (
                (sequence_output, pooled_output)
                + encoder_outputs[1:]
                + (embedding_output,)
            )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class RobertaEncoder4Mix(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [RobertaLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        hidden_states2=None,
        attention_mask2=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mix_lambda=None,
        mix_layer=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            ############################# Gernal step ##################################
            # Copied from transforms.models.roberta.modeling_roberta.RobertaEncoder
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            ############################################################################

            ################################## Mixup Step ##############################
            # Copied from https://github.com/clovaai/ssmix/blob/master/classification_model.py
            if mix_layer is not None:
                assert mix_layer >= 0
                assert mix_lambda > 0
                assert hidden_states2 is not None
                assert attention_mask2 is not None

                if i <= mix_layer:
                    if getattr(self.config, "gradient_checkpointing", False):

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(
                                    *inputs, past_key_value, output_attentions
                                )

                            return custom_forward

                        layer_outputs2 = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer_module),
                            hidden_states2,
                            attention_mask2,
                            layer_head_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                        )
                    else:
                        layer_outputs2 = layer_module(
                            hidden_states2,
                            attention_mask2,
                            layer_head_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            past_key_value,
                            output_attentions,
                        )

                if i == mix_layer:
                    hidden_states = (
                        mix_lambda * hidden_states + (1 - mix_lambda) * hidden_states2
                    )
            ############################################################################

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
