"""Small synthetic graphs used by correctness tests."""

import torch

from .data import K_NEIGHBORS
from .model import ATOM_VOCAB


def synthetic_drug(n_nodes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    graph = {
        "pos": torch.randn(n_nodes, 3, generator=generator),
        # Eight columns: the seven atom features plus the chirality tag.
        "feat": torch.zeros(n_nodes, len(ATOM_VOCAB) + 1, dtype=torch.long),
    }
    graph["edge"] = (~torch.eye(n_nodes, dtype=torch.bool)).nonzero().T
    graph["eattr"] = torch.zeros(graph["edge"].shape[1], dtype=torch.long)
    return graph


def synthetic_protein(n_nodes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    position = torch.randn(n_nodes, 3, generator=generator)
    k = min(K_NEIGHBORS, n_nodes - 1)
    nearest = torch.cdist(position, position).topk(k + 1, largest=False).indices[:, 1:]
    return {
        "pos": position,
        "feat": torch.zeros(n_nodes, dtype=torch.long),
        "plddt": torch.ones(n_nodes),
        "edge": torch.stack(
            [nearest.reshape(-1), torch.arange(n_nodes).repeat_interleave(k)]
        ),
        "eattr": torch.zeros(n_nodes * k, dtype=torch.long),
    }


def transform_graph(graph, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    translation = torch.randn(3, generator=generator) * 10
    return {**graph, "pos": graph["pos"] @ rotation + translation}


def reflect_graph(graph):
    """Mirror a graph through the yz-plane.

    Reflection preserves every interatomic distance while inverting chirality,
    so it is the sharpest probe of what a distance-only encoder can see: a
    model whose messages depend only on distances cannot tell the result from
    the original, which is precisely the enantiomer blindness being tested.
    """
    mirrored = graph["pos"].clone()
    mirrored[:, 0] = -mirrored[:, 0]
    return {**graph, "pos": mirrored}
