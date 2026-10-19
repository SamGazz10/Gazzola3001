"""The overtake-probability network.

Embedding sizes are small because the vocabularies are small (3-25 values).
"""
import torch
from torch import nn

from .features import BINARY, CATEGORICAL, NUMERIC, VOCABS

EMBED_DIMS = {"condition": 2, "segment": 3, "circuit": 6, "tyre": 4}


class OvertakeNet(nn.Module):
    """Embeddings for the categorical inputs, then a multilayer perceptron.

    Each categorical input is mapped to a small learned vector (both tyre
    columns share one tyre embedding). These are concatenated with the
    standardised numeric inputs and the 0/1 flags and passed through fully
    connected layers with SiLU activations and dropout. The output is a single
    logit; a sigmoid turns it into P(overtake).

    Parameters
    ----------
    hidden : tuple of int, optional
        Width of each hidden layer.
    dropout : float, optional
        Dropout probability after each hidden layer (training only).
    n_inputs : int, optional
        Number of numeric and flag inputs (`overtake.features.Preprocessor.n_inputs`).
    """

    def __init__(self, hidden=(64, 64), dropout=0.1, n_inputs=len(NUMERIC) + len(BINARY)):
        super().__init__()
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.n_inputs = n_inputs
        self.embeddings = nn.ModuleDict(
            {vocab: nn.Embedding(len(values), EMBED_DIMS[vocab]) for vocab, values in VOCABS.items()}
        )
        width = sum(EMBED_DIMS[vocab] for vocab in CATEGORICAL.values()) + n_inputs
        layers = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SiLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, num):
        """Compute logits.

        Parameters
        ----------
        cat : torch.LongTensor, shape (n, 5)
        num : torch.FloatTensor, shape (n, n_inputs)
            As produced by `overtake.features.Preprocessor.transform`.

        Returns
        -------
        torch.Tensor, shape (n,)
            Log-odds of an overtake.
        """
        embedded = [self.embeddings[vocab](cat[:, i]) for i, vocab in enumerate(CATEGORICAL.values())]
        return self.mlp(torch.cat(embedded + [num], dim=1)).squeeze(1)
