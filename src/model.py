import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Helper class that adds positional encoding to the token embedding to introduce a notion of word order
    """
    def __init__(self, emb_size: int, dropout: float, max_length: int = 512):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)

        pos_embedding = torch.zeros((max_length, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0).transpose(0, 1)

        self._dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self._dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    """
    Helper class to convert tensor of input indices into corresponding tensor of token embeddings
    """
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        self._emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self._emb_size)


class Seq2SeqTransformer(nn.Module):
    """
    Seq2Seq Network
    """
    def __init__(self, n_encoder_layers: int, n_decoder_layers: int, emb_size: int, n_heads: int, src_vocab_size: int,
                 tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.3):
        super().__init__()
        self.transformers = nn.Transformer(d_model=emb_size, nhead=n_heads, num_encoder_layers=n_encoder_layers,
                                           num_decoder_layers=n_decoder_layers, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True)
        self.generator = nn.Linear(in_features=emb_size, out_features=tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(vocab_size=src_vocab_size, emb_size=emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size=tgt_vocab_size, emb_size=emb_size)
        self.positional_encoding = PositionalEncoding(emb_size=emb_size, dropout=dropout)

        self._init_weights()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformers(src=src_emb, tgt=tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask,
                                 src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformers.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformers.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf')
        Unmasked positions are filled with float(0.0)
        :param sz: Size of the square mask
        :return: torch.Tensor
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
