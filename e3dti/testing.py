"""Small synthetic graphs used by correctness tests."""

import torch

from .data import K_NEIGHBORS


def synthetic_drug(n_nodes, seed=0):
    generator = torch.Generator().manual_seed(seed)
    graph = {
        "pos": torch.randn(n_nodes, 3, generator=generator),
        "feat": torch.zeros(n_nodes, 7, dtype=torch.long),
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
