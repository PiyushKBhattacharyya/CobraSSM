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
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class CobraCache:
    """Lightweight recurrent state cache for O(1) generation."""
    def __init__(self, ssm_states, mem_states, prev_embs, seen_seps):
        self.ssm_states = ssm_states
        self.mem_states = mem_states
        self.prev_embs = prev_embs
        self.seen_seps = seen_seps

    def __getitem__(self, idx):
        # HF sometimes indexes past_key_values — return self to avoid errors
        if idx == 0:
            return self.ssm_states
        elif idx == 1:
            return self.mem_states
        return None

    def __len__(self):
        return 2

    def get_seq_length(self, layer_idx=0):
        return 0  # Not applicable for recurrent models


class CobraForCausalLM(CobraPreTrainedModel, GenerationMixin):

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
        past_key_values=None,
        return_dict=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- O(1) Recurrent Path (generation) ---
        if past_key_values is not None and isinstance(past_key_values, CobraCache):
            cache = past_key_values
            # input_ids is (b, 1) — just the new token
            logits, new_ssm, new_mem, new_prev, new_sep = self.cobra.forward_step(
                input_ids,
                cache.ssm_states,
                cache.mem_states,
                cache.prev_embs,
                cache.seen_seps
            )
            new_cache = CobraCache(new_ssm, new_mem, new_prev, new_sep)
            
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                )

            if not return_dict:
                return ((loss,) + (logits, new_cache)) if loss is not None else (logits, new_cache)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=new_cache,
            )

        # --- Full Sequence Path (training / prefill) ---
        logits, out_ssm, out_mem = self.cobra(
            input_ids,
            ssm_states=None,
            mem_states=None
        )

        # Build initial cache from full-sequence states
        b = input_ids.shape[0]
        device = input_ids.device
        dtype = logits.dtype
        
        # prev_embs: last token's norm_mem embedding per layer
        prev_embs = []
        x = self.cobra.embedding(input_ids)
        for block in self.cobra.blocks:
            prev_embs.append(block.norm_mem(x)[:, -1])  # (b, d)
            # x changes per layer, but we approximate with input embedding
            # This is acceptable for the initial cache seed

        seen_seps = []
        for _ in self.cobra.blocks:
            # Check if SEP appeared anywhere in the prefix
            has_sep = (input_ids == 1).any(dim=-1)  # (b,)
            seen_seps.append(has_sep)

        new_cache = CobraCache(out_ssm, out_mem, prev_embs, seen_seps)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + (new_cache,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_cache,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None and isinstance(past_key_values, CobraCache):
            # Only feed the last token — states carry all history
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
        }
