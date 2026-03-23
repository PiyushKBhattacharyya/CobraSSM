import torch
import torch.nn as nn
from cobrassm import CobraConfig, CobraForCausalLM

def debug_backward():
    print("Testing Forward/Backward on CPU...")
    config = CobraConfig(vocab_size=100, d_model=64, num_hidden_layers=2, d_state=8, num_scales=2)
    device = torch.device('cpu')
    model = CobraForCausalLM(config).to(device)
    
    input_ids = torch.randint(0, 100, (1, 128)).to(device)
    labels = input_ids.clone()
    
    print("Forward Pass...")
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    print(f"Loss: {loss.item():.4f}")
    
    print("Backward Pass...")
    loss.backward()
    print("Backward Success!")

    # Now test DirectML if available
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"\nTesting Forward/Backward on DirectML ({dml_device})...")
        model_dml = CobraForCausalLM(config).to(dml_device)
        input_dml = input_ids.to(dml_device)
        labels_dml = labels.to(dml_device)
        
        print("Forward Pass (DML)...")
        outputs_dml = model_dml(input_dml, labels=labels_dml)
        loss_dml = outputs_dml.loss
        print(f"Loss: {loss_dml.item():.4f}")
        
        print("Backward Pass (DML)...")
        loss_dml.backward()
        print("Backward Success (DML)!")
    except Exception as e:
        print(f"DirectML Test Failed: {e}")

if __name__ == "__main__":
    debug_backward()
