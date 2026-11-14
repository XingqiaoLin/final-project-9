"""E3DTI model with padding-safe bilinear attention and geometry ablations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


ATOM_VOCAB = [13, 7, 5, 2, 6, 5, 2]
N_BOND = 5
SEP_BINS = [1, 2, 3, 4, 8, 16, 32]


class RBF(nn.Module):
    def __init__(self, n: int = 32, cutoff: float = 20.0):
        super().__init__()
        self.register_buffer("mu", torch.linspace(0, cutoff, n))
        self.gamma = (n / cutoff) ** 2

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distance - self.mu) ** 2)


class EGNNLayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int, n_rbf: int = 32):
        super().__init__()
        self.rbf = RBF(n_rbf)
        self.msg = nn.Sequential(
            nn.Linear(2 * dim + n_rbf + edge_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.coord = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1, bias=False)
        )
        nn.init.xavier_uniform_(self.coord[-1].weight, gain=1e-3)
        self.upd = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, x, edge, edge_features, use_geometry: bool = True):
        src, dst = edge
        rel = x[dst] - x[src]
        distance = rel.norm(dim=-1, keepdim=True)
        radial = self.rbf(distance) if use_geometry else torch.zeros_like(self.rbf(distance))
        messages = self.msg(
            torch.cat([h[dst], h[src], radial, edge_features], dim=-1)
        )
        messages = messages * self.gate(messages)
        degree = (
            torch.zeros(len(h), 1, device=h.device)
            .index_add_(0, dst, torch.ones_like(distance))
            .clamp(min=1)
        )
        if use_geometry:
            x = x + (
                torch.zeros_like(x).index_add_(
                    0,
                    dst,
                    rel / (distance + 1) * torch.tanh(self.coord(messages)),
                )
                / degree
            )
        aggregated = torch.zeros_like(h).index_add_(0, dst, messages) / degree
        h = self.norm(h + self.upd(torch.cat([h, aggregated], dim=-1)))
        return h, x


class EGNN(nn.Module):
    def __init__(self, dim: int, edge_dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [EGNNLayer(dim, edge_dim) for _ in range(n_layers)]
        )

    def forward(self, h, x, edge, edge_features, use_geometry: bool = True):
        for layer in self.layers:
            h, x = layer(h, x, edge, edge_features, use_geometry=use_geometry)
        return h


class DrugEncoder(nn.Module):
    def __init__(self, dim: int, n_layers: int, use_geometry: bool = True):
        super().__init__()
        self.use_geometry = use_geometry
        self.embeddings = nn.ModuleList([nn.Embedding(v, dim) for v in ATOM_VOCAB])
        self.bond = nn.Embedding(N_BOND, 16)
        self.egnn = EGNN(dim, 16, n_layers)

    def forward(self, graph):
        h = sum(
            embedding(graph["feat"][:, i])
            for i, embedding in enumerate(self.embeddings)
        )
        return self.egnn(
            h,
            graph["pos"],
            graph["edge"],
            self.bond(graph["eattr"]),
            use_geometry=self.use_geometry,
        )


class ProteinEncoder(nn.Module):
    def __init__(self, dim: int, n_layers: int, use_geometry: bool = True):
        super().__init__()
        self.use_geometry = use_geometry
        self.aa = nn.Embedding(21, dim)
        self.plddt = nn.Linear(1, dim)
        self.separation = nn.Embedding(len(SEP_BINS) + 1, 16)
        self.egnn = EGNN(dim, 16, n_layers)

    def forward(self, graph):
        h = self.aa(graph["feat"]) + self.plddt(graph["plddt"][:, None])
        return self.egnn(
            h,
            graph["pos"],
            graph["edge"],
            self.separation(graph["eattr"]),
            use_geometry=self.use_geometry,
        )


def to_dense(h, batch, index, n_graphs):
    """Pad concatenated nodes and return both features and a validity mask."""
    max_nodes = int(index.max()) + 1
    out = h.new_zeros(n_graphs, max_nodes, h.size(-1))
    mask = torch.zeros(n_graphs, max_nodes, dtype=torch.bool, device=h.device)
    out[batch, index] = h
    mask[batch, index] = True
    return out, mask


class FCNet(nn.Module):
    def __init__(self, dims, activation="ReLU", dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(dims[i], dims[i + 1]), dim=None))
            if activation:
                layers.append(getattr(nn, activation)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if activation:
            layers.append(getattr(nn, activation)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MaskedBANLayer(nn.Module):
    """DrugBAN BAN layer with explicit drug/protein padding masks."""

    def __init__(
        self,
        v_dim,
        q_dim,
        h_dim,
        h_out,
        activation="ReLU",
        dropout=0.2,
        k=3,
    ):
        super().__init__()
        self.c = 32
        self.k = k
        self.h_dim = h_dim
        self.h_out = h_out
        self.v_net = FCNet([v_dim, h_dim * k], activation, dropout)
        self.q_net = FCNet([q_dim, h_dim * k], activation, dropout)
        if k > 1:
            self.p_net = nn.AvgPool1d(k, stride=k)
        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.empty(1, h_out, 1, h_dim * k).normal_())
            self.h_bias = nn.Parameter(torch.empty(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * k, h_out), dim=None)
        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, attention):
        logits = torch.einsum("bvk,bvq,bqk->bk", v, attention, q)
        if self.k > 1:
            logits = self.p_net(logits.unsqueeze(1)).squeeze(1) * self.k
        return logits

    def forward(self, v, q, v_mask, q_mask, softmax=False):
        v_projected = self.v_net(v)
        q_projected = self.q_net(q)
        if self.h_out <= self.c:
            attention = (
                torch.einsum(
                    "xhyk,bvk,bqk->bhvq", self.h_mat, v_projected, q_projected
                )
                + self.h_bias
            )
        else:
            v_outer = v_projected.transpose(1, 2).unsqueeze(3)
            q_outer = q_projected.transpose(1, 2).unsqueeze(2)
            product = torch.matmul(v_outer, q_outer)
            attention = self.h_net(product.transpose(1, 2).transpose(2, 3))
            attention = attention.transpose(2, 3).transpose(1, 2)

        valid = v_mask[:, None, :, None] & q_mask[:, None, None, :]
        if softmax:
            attention = attention.masked_fill(~valid, torch.finfo(attention.dtype).min)
            attention = F.softmax(
                attention.flatten(2), dim=-1
            ).view_as(attention)
            attention = attention.masked_fill(~valid, 0)
        else:
            attention = attention.masked_fill(~valid, 0)

        # Zero projected padding too, so FCNet biases cannot leak through pooling.
        v_projected = v_projected * v_mask.unsqueeze(-1)
        q_projected = q_projected * q_mask.unsqueeze(-1)
        logits = sum(
            self.attention_pooling(v_projected, q_projected, attention[:, head])
            for head in range(self.h_out)
        )
        return self.bn(logits), attention


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return self.fc4(x)


class E3DTI(nn.Module):
    ABLATIONS = {
        "full": (True, True),
        "no_drug_3d": (False, True),
        "no_protein_3d": (True, False),
        "no_3d": (False, False),
    }

    def __init__(
        self,
        dim=128,
        drug_layers=4,
        protein_layers=4,
        ban_dim=256,
        heads=2,
        ablation="full",
    ):
        super().__init__()
        if ablation not in self.ABLATIONS:
            raise ValueError(f"Unknown ablation {ablation!r}; choose {sorted(self.ABLATIONS)}")
        drug_geometry, protein_geometry = self.ABLATIONS[ablation]
        self.drug = DrugEncoder(dim, drug_layers, drug_geometry)
        self.protein = ProteinEncoder(dim, protein_layers, protein_geometry)
        self.bcn = weight_norm(
            MaskedBANLayer(dim, dim, ban_dim, heads), name="h_mat", dim=None
        )
        self.classifier = MLPDecoder(ban_dim, 512, 128)

    def forward(self, drug_graph, protein_graph):
        n_graphs = int(drug_graph["n_graphs"])
        drug, drug_mask = to_dense(
            self.drug(drug_graph),
            drug_graph["batch"],
            drug_graph["idx"],
            n_graphs,
        )
        protein, protein_mask = to_dense(
            self.protein(protein_graph),
            protein_graph["batch"],
            protein_graph["idx"],
            n_graphs,
        )
        fused, attention = self.bcn(
            drug, protein, drug_mask, protein_mask, softmax=False
        )
        return self.classifier(fused).squeeze(-1), attention


def collate_graphs(graphs):
    node_counts = torch.tensor([len(graph["pos"]) for graph in graphs])
    offsets = torch.cumsum(node_counts, 0) - node_counts
    out = {
        key: torch.cat([graph[key] for graph in graphs])
        for key in graphs[0]
        if key != "edge"
    }
    out["edge"] = torch.cat(
        [graph["edge"] + offset for graph, offset in zip(graphs, offsets)], dim=1
    )
    out["batch"] = torch.arange(len(graphs)).repeat_interleave(node_counts)
    out["idx"] = torch.cat([torch.arange(n) for n in node_counts.tolist()])
    out["n_graphs"] = torch.tensor(len(graphs))
    return out
