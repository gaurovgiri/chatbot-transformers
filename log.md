1. Build CharTokenizer
2. Build Dataset class
3. Build Multi-Head Self Attention
    - instantiate two Linear Projections with two dropouts. One will be `qkv` that will project d_model to 3*d_model (Q, K, V) and another is `out` that will project d_model to d_model
    - now this multi head self attention will pass the input to qkv
    - break down `qkv` into `q`, `k` and `v` which are currently of `(B, T, d_model)`
    - break down d_model of each into `n_heads` and `head_dim`, and rearrange the dimension such that we can calculate attention for each head separately i.e. `(B, n_head, T, head_dim)`
    - start caculating attention_score by dot product `query` with transpose of `key` such that we calculate query against each key to find the most suitable one
    - apply attention_masking using lower triangular mask
    - to find out the most suitable`key` for a `query`, we will apply softmax against the last dimension (key dimension)
    - apply dropout to the score to randomly mask attention for better performance
    - now apply dot product with the `value` to get the final attention score
    - now revert the dimensions to original state by transposing 1 and 2 dimensions and merging n_heads and head_dim with view to get again C or d_model
    - pass the output through linear projection `out` and apply dropout
4. Build a FeedForward Network
    - Simple Non-Linear Network build using two linear layers and ReLU or GELU activation in middle
    - input and output dim is d_model but the dim in between is d_ff (greater than d_model), acting as 2 convolution layer with size 1 kernel
    - add a dropout and output the value
5. Build Transformer Block
    - This is the collection of Multi head self attention, feed forward network that are passed layer normalization
    - The residues are also passed along!
6. Building the whole model
    - Now the entire Transformer Based Language Model is written.
    - The language model contains two embeddings:
        - Token Embedding => This is the relationship between tokens (vocab_size, d_model)
        - Positional Embedding => This is embedding is used to know the position of the token in the sentence (seq_len, d_model)
    - pass them both to dropout after adding both the embedding
    - Now the transformer blocks can be multiplied as many layers as required
    - Finally the output of the transformer block is passed through layer normalization
    - A linear layer is required to map embeddings to vocab_size or our logits by which we predict our next token.
7. Write a training loop
8. Write a inference code