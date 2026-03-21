import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_cobra import CobraConfig
from .model import CobraSSM

logger = logging.get_logger(__name__)

class CobraPreTrainedModel(PreTrainedModel):
    config_class = CobraConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False # Initial version

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class CobraForCausalLM(CobraPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"cobra.embedding.weight": True, "cobra.lm_head.weight": True}

    def __init__(self, config: CobraConfig):
        super().__init__(config)
        self.cobra = CobraSSM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.num_hidden_layers,
            d_state=config.d_state,
            num_scales=config.num_scales,
            num_slots=config.num_slots
        )
        # Inform transformers about the tied weights
        self.tie_weights()
        self.post_init()


    def get_input_embeddings(self):
        return self.cobra.embedding

    def set_input_embeddings(self, value):
        self.cobra.embedding = value

    def get_output_embeddings(self):
        return self.cobra.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.cobra.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.cobra = decoder

    def get_decoder(self):
        return self.cobra

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        ssm_states=None,
        mem_states=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits, out_ssm_states, out_mem_states = self.cobra(
            input_ids, 
            ssm_states=ssm_states, 
            mem_states=mem_states
        )

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + (out_ssm_states, out_mem_states)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(out_ssm_states, out_mem_states), # Wrapping recurrent states
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Handle DynamicCache or tuple-based past_key_values
        ssm_states = None
        mem_states = None
        
        if past_key_values is not None:
            if hasattr(past_key_values, "__getitem__") and not isinstance(past_key_values, str):
                ssm_states = past_key_values[0]
                mem_states = past_key_values[1]
            elif hasattr(past_key_values, "get_seq_length"): # likely a Cache object
                # For now, we don't have a custom CobraCache, so we fallback
                # This part might need further refinement for true recurrent generation
                pass

        return {
            "input_ids": input_ids,
            "ssm_states": ssm_states,
            "mem_states": mem_states,
        }

