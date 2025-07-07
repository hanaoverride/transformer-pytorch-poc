import torch
from .embedding import TokenEmbedding, PositionalEncoding
from .encoder import EncoderBlock

def main():
    # Parameters
    vocab_size = 1000
    d_model = 512
    n_heads = 8
    d_ff = 2048
    seq_len = 10
    batch_size = 2

    # Create dummy input
    src = torch.randint(0, vocab_size, (batch_size, seq_len))
    src_mask = None # In a real scenario, you would create a mask.

    # Instantiate modules
    token_embedding = TokenEmbedding(vocab_size, d_model)
    pos_embedding = PositionalEncoding(d_model, max_len=seq_len)
    encoder_block = EncoderBlock(d_model, n_heads, d_ff)

    # Forward pass
    embedded = token_embedding(src)
    positioned = pos_embedding(embedded.transpose(0, 1)).transpose(0, 1)
    
    print("Shape after embedding and positional encoding:", positioned.shape)

    output = encoder_block(positioned, src_mask)

    print("Output shape from Encoder Block:", output.shape)
    print("Output value sample from Encoder Block:\n", output)

if __name__ == '__main__':
    main()
