from src.tokenizer import CharTokenizer
from src.dataset import TextDataset
from src.model import TransformerLM
import torch
from torch.utils.data import DataLoader
import os
import joblib

SEQ_LENGTH = 8
BATCH_SIZE = 2
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
FFN_FACTOR = 4
NUM_EPOCHS = 10

if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")


device = torch.device("cpu")

with open("data/example.txt", "r") as f:
    raw_texts = f.read()


tokenizer = CharTokenizer(texts=raw_texts)
token_ids = tokenizer.encode(raw_texts)

dataset = TextDataset(token_ids=token_ids, seq_len=SEQ_LENGTH, pad_id=tokenizer.pad_id)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)


model = TransformerLM(tokenizer=tokenizer, d_model=D_MODEL, d_ff=D_MODEL, seq_length=SEQ_LENGTH, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_factor=FFN_FACTOR, dropout=0.1).to(device)


get_causal_mask = lambda T: torch.tril(torch.ones((1, 1, T, T), dtype=torch.bool))
mask = get_causal_mask(SEQ_LENGTH).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb, attn_mask=mask).to(device)
        loss = criterion(logits.view(-1, tokenizer.vocab_size), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch{epoch}.pt")

joblib.dump(tokenizer, "checkpoints/tokenizer.joblib")
