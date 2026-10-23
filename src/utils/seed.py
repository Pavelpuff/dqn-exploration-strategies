import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Sets all random seeds for deterministic and reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(model: nn.Module) -> None:
    """Initializes linear layer weights orthogonally and biases to zero."""
    if isinstance(model, nn.Linear):
        nn.init.orthogonal_(model.weight.data)
        if model.bias is not None:
            model.bias.data.fill_(0.0)
