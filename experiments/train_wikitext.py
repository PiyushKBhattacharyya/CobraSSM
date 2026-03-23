import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from cobrassm import CobraConfig, CobraForCausalLM
import time

def train_wikitext():
    device = torch.device('cpu')
    print("Using CPU for stable training baseline.")


    # 1. Setup Tokenizer and Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    config = CobraConfig(
        vocab_size=len(tokenizer),
        d_model=128, # Reduced from 256 to 128 (approx 8.5M params)
        num_hidden_layers=6,
        d_state=16,
        num_scales=4
    )
    model = CobraForCausalLM(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 2. Load Dataset (WikiText-2)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")
    
    train_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    model.train()
    print("Starting training on WikiText-2...")
    
    for epoch in range(1):
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 1 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | Perplexity: {torch.exp(loss).item():.2f}")

                
            if step > 500: # Temporary limit for validation
                break

if __name__ == "__main__":
    train_wikitext()
