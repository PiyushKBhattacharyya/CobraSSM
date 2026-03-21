from transformers import PretrainedConfig

class CobraConfig(PretrainedConfig):
    model_type = "cobra"

    def __init__(
        self,
        vocab_size=100,
        d_model=128,
        num_hidden_layers=1,
        d_state=16,
        num_scales=4,
        num_slots=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sep_id=1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.d_state = d_state
        self.num_scales = num_scales
        self.num_slots = num_slots
        self.sep_id = sep_id
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
