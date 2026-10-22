"""The overtake-probability network.

Embedding sizes are small because the vocabularies are small (3-25 values).
The network outputs a logit; `overtake.predict.OvertakePredictor` applies the
sigmoid and averages an ensemble of these networks.
"""
import torch
from torch import nn

from .features import BINARY, CATEGORICAL, NUMERIC, VOCABS

# Embedding width per vocabulary (judgement): fewer numbers than values, more
# for larger vocabularies (3 conditions, 4 segments, 25 circuits, 5 tyres).
EMBED_DIMS = {"condition": 2, "segment": 3, "circuit": 6, "tyre": 4}


class OvertakeNet(nn.Module):
    """Embeddings for the categorical inputs, then a multilayer perceptron.

    Each categorical input is mapped to a small learned vector (both tyre
    columns share one tyre embedding). These are concatenated with the
    standardised numeric inputs and the 0/1 flags and passed through fully
    connected layers with SiLU activations and dropout. The output is a single
    logit; a sigmoid turns it into P(overtake).

    With the default arguments the first layer takes 33 numbers (19 from the
    embeddings, counting the tyre embedding once per tyre column, plus 14
    inputs) and the network has 6,589 trainable parameters.

    Parameters
    ----------
    hidden : tuple of int, optional
        Width of each hidden layer.
    dropout : float, optional
        Dropout probability after each hidden layer (training only).
    n_inputs : int, optional
        Number of numeric and flag inputs (`overtake.features.Preprocessor.n_inputs`).
        The default, 14, matches a preprocessor without physics features.

    Attributes
    ----------
    hidden, dropout, n_inputs
        The arguments, kept so `overtake.predict.OvertakePredictor.save` can
        store the architecture and rebuild the network from a checkpoint.
    embeddings : torch.nn.ModuleDict
        One `torch.nn.Embedding` per vocabulary, keyed by vocabulary name.
    mlp : torch.nn.Sequential
        The fully connected layers, ending in a single output.

    Examples
    --------
    >>> import pandas as pd
    >>> from overtake.features import DEFAULTS, Preprocessor
    >>> df = pd.DataFrame([DEFAULTS, {**DEFAULTS, "gap_s": 0.5}])
    >>> pre = Preprocessor().fit(df)
    >>> net = OvertakeNet(n_inputs=pre.n_inputs).eval()
    >>> torch.sigmoid(net(*pre.transform(df))).shape  # untrained, so the values mean nothing
    torch.Size([2])
    """

    def __init__(self, hidden=(64, 64), dropout=0.1, n_inputs=len(NUMERIC) + len(BINARY)):
        """Build the embedding tables and layers; see the class docstring."""
        super().__init__()
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.n_inputs = n_inputs
        self.embeddings = nn.ModuleDict(
            {vocab: nn.Embedding(len(values), EMBED_DIMS[vocab]) for vocab, values in VOCABS.items()}
        )
        # Summed over the categorical columns, not the vocabularies, so the
        # shared tyre embedding counts twice (attacker and defender).
        width = sum(EMBED_DIMS[vocab] for vocab in CATEGORICAL.values()) + n_inputs
        layers = []
        for size in hidden:
            layers += [nn.Linear(width, size), nn.SiLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cat, num):
        """Compute logits from the tensors `overtake.features.Preprocessor.transform` produces.

        Parameters
        ----------
        cat : torch.LongTensor, shape (n, 5)
            Vocabulary indices, columns in `overtake.features.CATEGORICAL` order.
        num : torch.FloatTensor, shape (n, n_inputs)
            Standardised numeric inputs followed by the 0/1 flags.

        Returns
        -------
        torch.Tensor, shape (n,)
            Log-odds of an overtake.
        """
        # Column i of cat is looked up in its own vocabulary's table, so both
        # tyre columns read the shared tyre embedding.
        embedded = [self.embeddings[vocab](cat[:, i]) for i, vocab in enumerate(CATEGORICAL.values())]
        return self.mlp(torch.cat(embedded + [num], dim=1)).squeeze(1)
