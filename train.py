import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.cuda.amp import autocast, GradScaler
from google.colab import drive

# 1. MOUNT DRIVE & SETUP PATHS
drive.mount('/content/drive')
CHECKPOINT_DIR = "/content/drive/MyDrive/SparseMind_Checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "sparsemind_300m_latest.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

# 2. INITIALIZE MODEL & OPTIMIZER
model = SparseMindLM(hidden_dim=512, num_heads=8, num_layers=8, num_slots=65536).to(device)
optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
scaler = GradScaler()
loss_fn = nn.CrossEntropyLoss()

# 3. CHECKPOINT RECOVERY SYSTEM
start_step = 1
if os.path.exists(CHECKPOINT_PATH):
    print(f"🔄 Found existing checkpoint at {CHECKPOINT_PATH}. Resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_step = checkpoint['step'] + 1
    print(f"✅ Resuming from Step {start_step}")
else:
    print("🚀 No checkpoint found. Starting fresh pre-training!")

# 4. DATA PIPELINE (FineWeb Streaming)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
data_iterator = iter(dataset)

BLOCK_SIZE = 256
MICRO_BATCH = 8       
GRAD_ACCUM_STEPS = 4  

def get_stream_batch():
    text_chunk = ""
    while len(tokenizer.encode(text_chunk)) < (BLOCK_SIZE * MICRO_BATCH) + 1:
        text_chunk += " " + next(data_iterator)['text']
    tokens = tokenizer.encode(text_chunk, truncation=True, max_length=(BLOCK_SIZE * MICRO_BATCH) + 1)
    tokens = torch.tensor(tokens, dtype=torch.long)
    x = tokens[:-1].view(MICRO_BATCH, BLOCK_SIZE)
    y = tokens[1:].view(MICRO_BATCH, BLOCK_SIZE)
    return x.to(device), y.to(device)

# 5. THE TRAINING LOOP
STEPS = 100000 # Set high for 6B tokens
SAVE_EVERY = 1000 # Saves to Google Drive every 1000 steps

model.train()
for step in range(start_step, STEPS + 1):
    for _ in range(GRAD_ACCUM_STEPS):
        x, y = get_stream_batch()
        
        with autocast(dtype=torch.float16):
            logits, entropy_penalty = model(x)
            main_loss = loss_fn(logits.view(-1, 50257), y.view(-1))
            loss = (main_loss + (0.01 * entropy_penalty)) / GRAD_ACCUM_STEPS
        
        scaler.scale(loss).backward()
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    if step % 50 == 0:
        true_loss = main_loss.item()
        perplexity = math.exp(min(true_loss, 15)) # Cap to prevent math overflow display
        print(f"Step {step:5d} | True Loss: {true_loss:.4f} | Perplexity: {perplexity:.2f}")

    # RESILIENT SAVING
    if step % SAVE_EVERY == 0:
        print(f"💾 Saving Checkpoint at Step {step}...")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': true_loss,
        }, CHECKPOINT_PATH)
        print("✅ Checkpoint Saved!")

print("\n🎉 Training Complete!")
