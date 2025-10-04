import torch
import torch.nn.functional as F

SEQ_LENGTH = 32
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
FFN_FACTOR = 4

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device=torch.device("cpu")):
    """
    model: trained TransformerLM
    tokenizer: CharTokenizer
    prompt: string
    max_new_tokens: number of tokens to generate
    temperature: controls randomness (higher = more random)
    """
    model.eval()
    model.to(device)

    # Encode prompt to token IDs
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # If input is longer than model seq_len, truncate
        if generated.size(1) > model.seq_len:
            x = generated[:, -model.seq_len:]
        else:
            x = generated

        # Forward pass
        logits = model(x)  # (1, T, vocab_size)

        # Take logits of last token
        next_token_logits = logits[:, -1, :] / temperature

        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)  # (1,1)

        # Append to sequence
        generated = torch.cat([generated, next_token], dim=1)

    # Decode generated tokens to string
    output_text = tokenizer.decode(generated[0].tolist())
    return output_text

if __name__ == "__main__":
    from src.model import TransformerLM
    import joblib

    device = torch.device('cpu')
    tokenizer = joblib.load("checkpoints/tokenizer.joblib")

    model = TransformerLM(tokenizer=tokenizer, d_model=D_MODEL, d_ff=D_MODEL, seq_length=SEQ_LENGTH, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_factor=FFN_FACTOR, dropout=0.1).to(device)
    model.load_state_dict(torch.load("checkpoints/checkpoint_epoch9.pt", map_location=device))
    model.eval()

    prompt = "What is coding?"
    generated_text = generate(model, tokenizer, prompt, max_new_tokens=1024, temperature=1, device=device)
    print("".join(generated_text))