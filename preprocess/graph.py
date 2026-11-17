"""Patient similarity graph over the prior-weighted protein profiles.

The encoder only uses the support of the matrix: ``Adj == 0`` becomes a forbidden
attention pair. Both the threshold and the training graph are estimated on the
training fold, and a test patient is attached to its training neighbours only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def estimate_threshold(edge_per_node: int, data: torch.Tensor) -> float:
    """Distance at order statistic ``edge_per_node * N``, i.e. about that many neighbours per node."""
    dist = cosine_distance(data, data)
    index = edge_per_node * data.shape[0]
    return torch.sort(dist.reshape(-1)).values[index].item()


def _symmetrise(adj: torch.Tensor) -> torch.Tensor:
    adj_t = adj.transpose(0, 1)
    return adj + adj_t * (adj_t > adj).float() - adj * (adj_t > adj).float()


def _add_self_loops_and_normalise(adj: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(adj.shape[0], device=adj.device)
    return F.normalize(adj + identity, p=1)


def build_train_adjacency(data_tr: torch.Tensor, threshold: float) -> torch.Tensor:
    """Graph of the training fold."""
    dist = cosine_distance(data_tr, data_tr)
    keep = (dist <= threshold).float()
    diagonal = torch.arange(keep.shape[0], device=keep.device)
    keep[diagonal, diagonal] = 0

    adj = (1 - dist) * keep
    return _add_self_loops_and_normalise(_symmetrise(adj))


def build_train_test_adjacency(data_all: torch.Tensor, num_train: int, threshold: float) -> torch.Tensor:
    """Training graph plus the train--test cross edges; the test--test block stays zero."""
    num_all = data_all.shape[0]
    adj = torch.zeros((num_all, num_all), device=data_all.device)

    dist_tr2te = cosine_distance(data_all[:num_train], data_all[num_train:])
    adj[:num_train, num_train:] = (1 - dist_tr2te) * (dist_tr2te <= threshold).float()

    dist_te2tr = cosine_distance(data_all[num_train:], data_all[:num_train])
    adj[num_train:, :num_train] = (1 - dist_te2tr) * (dist_te2tr <= threshold).float()

    return _add_self_loops_and_normalise(_symmetrise(adj))


def attention_mask(adj: torch.Tensor) -> torch.Tensor:
    """True where the pair is forbidden, as expected by nn.MultiheadAttention."""
    return adj == 0
