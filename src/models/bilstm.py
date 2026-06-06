# filename: src/models/bilstm.py
# purpose:  BiLSTM ticket-type classifier — GloVe 100d → BiLSTM → Linear head
# version:  1.0

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM ticket-type classifier with GloVe embeddings.

    Architecture: Embedding(GloVe-100d) → Dropout → BiLSTM → Dropout → Linear

    Special token contract (must match build_vocab in 08a_bilstm.py):
        <PAD> = index 0  (padding_idx — embedding returns zero vector)
        <UNK> = index 1  (OOV tokens)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_classes: int = 5,
        dropout: float = 0.3,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.bidirectional = bidirectional   # stored explicitly — nn.LSTM has no .bidirectional attr

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_out_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor (batch_size, seq_len) — token indices
        Returns:
            logits: FloatTensor (batch_size, n_classes)
        """
        embedded = self.dropout(self.embedding(x))          # (B, L, emb_dim)
        _, (hidden, _) = self.lstm(embedded)                # hidden: (n_layers*dirs, B, H)

        if self.bidirectional:
            # hidden[-2]: last layer forward pass
            # hidden[-1]: last layer backward pass
            hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)   # (B, 2H)
        else:
            hidden_cat = hidden[-1]                                    # (B, H)

        return self.fc(self.dropout(hidden_cat))             # (B, n_classes)

    def load_pretrained_embeddings(
        self,
        embedding_matrix: torch.Tensor,
        freeze: bool = False,
    ) -> None:
        """
        Copy pretrained GloVe vectors into the embedding layer.

        Args:
            embedding_matrix: FloatTensor (vocab_size, embedding_dim)
            freeze: if True, embedding weights are not updated during backprop

        Raises:
            ValueError: if embedding_matrix shape doesn't match (vocab_size, embedding_dim)
            TypeError:  if embedding_matrix is not a torch.Tensor
        """
        expected = (self.vocab_size, self.embedding_dim)
        if embedding_matrix.shape != expected:
            raise ValueError(
                f"Embedding matrix shape mismatch: expected {expected}, "
                f"got {tuple(embedding_matrix.shape)}"
            )
        if not isinstance(embedding_matrix, torch.Tensor):
            raise TypeError(
                f"embedding_matrix must be torch.Tensor, got {type(embedding_matrix).__name__}"
            )
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = not freeze
