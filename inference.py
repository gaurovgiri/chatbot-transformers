import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device="cpu"):
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