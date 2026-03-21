import torch
from cobrassm import CobraConfig, CobraForCausalLM

def test_hf_integration():
    config = CobraConfig(
        vocab_size=100,
        d_model=64,
        n_layers=1,
        d_state=8,
        num_scales=2,
        num_slots=8
    )
    
    model = CobraForCausalLM(config)
    print("Model initialized successfully.")
    print(f"Parameter count: {model.num_parameters() / 1e3:.2f} K")
    
    # Test forward
    input_ids = torch.randint(0, 100, (1, 10))
    outputs = model(input_ids, labels=input_ids)
    print(f"Forward pass completed. Loss: {outputs.loss.item():.4f}")
    
    # Test save/load
    model.save_pretrained("./test_model")
    model2 = CobraForCausalLM.from_pretrained("./test_model")
    print("Model saved and reloaded successfully.")
    
    # Test generation
    gen_ids = model2.generate(input_ids, max_new_tokens=5, do_sample=True, top_k=50)
    print(f"Generation successful. Output shape: {gen_ids.shape}")

if __name__ == "__main__":
    test_hf_integration()
