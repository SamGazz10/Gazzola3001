"""The overtake-probability network."""
import torch
from torch import nn

from .features import BINARY, CATEGORICAL, NUMERIC, VOCABS

EMBED_DIMS = {"condition": 2, "segment": 3, "circuit": 6, "tyre": 4}


class OvertakeNet(nn.Module):
    """Embeddings for the categorical inputs, then a multilayer perceptron.

    The output is a single logit; apply a sigmoid to get P(overtake).
    """

    def __init__(self, hidden=(128, 128, 64), dropout=0.1):
        super().__init__()
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.embeddings = nn.ModuleDict(
            {vocab: nn.Embedding(len(values), EMBED_DIMS[vocab]) for vocab, values in VOCABS.items()}
        )
        width = sum(EMBED_DIMS[vocab] for vocab in CATEGORICAL.values()) + len(NUMERIC) + len(BINARY)
        layers = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SiLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, num):
        embedded = [self.embeddings[vocab](cat[:, i]) for i, vocab in enumerate(CATEGORICAL.values())]
        return self.mlp(torch.cat(embedded + [num], dim=1)).squeeze(1)
