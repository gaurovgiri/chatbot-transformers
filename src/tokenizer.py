'''
Transformers don't work on raw strings — they work on sequences of integers (token IDs).

A tokenizer maps text → token IDs (encode), and later we map token IDs → text (decode).
'''

class CharTokenizer:
    def __init__(self, texts):
        self.vocab = ["<pad>", "<unk>"] + sorted(set(texts))
        self.pad_id = 0
        self.unk_id = 1

        # Our lookup table for Token IDs and Corresponding Character
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text:str):
        return [self.stoi.get(ch, "<unk>") for ch in text]
    
    def decode(self, ids: list[int]):
        return [self.itos.get(i, self.stoi["<unk>"]) for i in ids]
    

    @property
    def vocab_size(self):
        return len(self.vocab)
    
if __name__ == "__main__":
    tokenizer = CharTokenizer("hello my name is gaurav")
    enc = tokenizer.encode('hi')
    print(enc)
    dec = tokenizer.decode(enc)
    print(dec)
    