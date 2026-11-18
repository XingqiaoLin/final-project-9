"""E3DTI model with padding-safe bilinear attention and geometry ablations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


ATOM_VOCAB = [13, 7, 5, 2, 6, 5, 2]
# Tetrahedral chirality tag, appended by drug_graph as an eighth atom column:
# unspecified, clockwise, counter-clockwise, other.
#
# It sits outside ATOM_VOCAB on purpose. Both drug encoders iterate over
# ATOM_VOCAB, so the extra column is inert unless a model asks for it, and
# every run made before this existed stays bit-for-bit reproducible.
CHIRALITY_VOCAB = 4
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


def shuffle_positions_within_graphs(pos, batch):
    """Permute coordinates among the nodes of each graph, independently per graph.

    This is the capacity-matched geometry control. Turning geometry off
    (`no_drug_3d` and friends) removes an input pathway: the RBF features go to
    zero and the coordinate-update MLP stops contributing, so a drop in score
    could be read as the model having less to work with rather than the geometry
    having carried signal. Shuffling instead keeps every tensor shape, every
    parameter and every code path identical, and keeps each molecule's own set
    of coordinates intact -- it only destroys which atom sits where. A model
    that scores the same with shuffled coordinates was not using the geometry.

    A fresh permutation is drawn on every forward pass, like dropout noise, so
    the coordinates cannot be memorised. That makes evaluation of this ablation
    stochastic; it is seeded and restored along with the rest of the RNG state.
    """
    keys = torch.rand(len(batch), device=pos.device)
    # batch is ascending and keys lie in [0, 1), so sorting batch + key keeps
    # every node inside its own graph's block and randomises order within it.
    order = torch.argsort(batch.to(keys.dtype) + keys)
    return pos[order]


class DrugEncoder(nn.Module):
    """E(3)-equivariant drug encoder.

    Messages depend only on interatomic distances, which are preserved by
    reflection as well as rotation. The branch is therefore blind to chirality
    by construction: a molecule and its mirror image produce identical
    embeddings. Roughly 45% of the drugs in every benchmark split here carry a
    defined stereocentre, so that blindness is not a corner case.

    `use_chirality` adds the tetrahedral tag as a learned embedding. The tag
    comes from the molecular graph, not the coordinates, so it distinguishes
    enantiomers without disturbing invariance to rotation or translation.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        use_geometry: bool = True,
        shuffle_coords: bool = False,
        use_chirality: bool = False,
    ):
        super().__init__()
        self.use_geometry = use_geometry
        self.shuffle_coords = shuffle_coords
        self.use_chirality = use_chirality
        self.embeddings = nn.ModuleList([nn.Embedding(v, dim) for v in ATOM_VOCAB])
        self.chirality = (
            nn.Embedding(CHIRALITY_VOCAB, dim) if use_chirality else None
        )
        self.bond = nn.Embedding(N_BOND, 16)
        self.egnn = EGNN(dim, 16, n_layers)

    def forward(self, graph):
        h = sum(
            embedding(graph["feat"][:, i])
            for i, embedding in enumerate(self.embeddings)
        )
        if self.chirality is not None:
            h = h + self.chirality(graph["feat"][:, len(ATOM_VOCAB)])
        pos = graph["pos"]
        if self.shuffle_coords:
            pos = shuffle_positions_within_graphs(pos, graph["batch"])
        return self.egnn(
            h,
            pos,
            graph["edge"],
            self.bond(graph["eattr"]),
            use_geometry=self.use_geometry,
        )


class ProteinEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        use_geometry: bool = True,
        shuffle_coords: bool = False,
    ):
        super().__init__()
        self.use_geometry = use_geometry
        self.shuffle_coords = shuffle_coords
        self.aa = nn.Embedding(21, dim)
        self.plddt = nn.Linear(1, dim)
        self.separation = nn.Embedding(len(SEP_BINS) + 1, 16)
        self.egnn = EGNN(dim, 16, n_layers)

    def forward(self, graph):
        h = self.aa(graph["feat"]) + self.plddt(graph["plddt"][:, None])
        pos = graph["pos"]
        if self.shuffle_coords:
            pos = shuffle_positions_within_graphs(pos, graph["batch"])
        return self.egnn(
            h,
            pos,
            graph["edge"],
            self.separation(graph["eattr"]),
            use_geometry=self.use_geometry,
        )


class DrugBANDrugEncoder(nn.Module):
    """DrugBAN's drug branch: a GCN over the covalent bond graph.

    The contrast with DrugEncoder is the point of the whole project, so the two
    are kept as alike as possible apart from the thing under test. Atom
    features, embedding dimension and the fusion layers are identical; what
    differs is that this sees only *which atoms are bonded*, with no distances
    and no coordinates.

    drug_graph ships a fully connected edge list with `eattr` holding the bond
    type and 0 meaning "no bond", so the covalent graph is recovered by masking
    rather than by building a second graph. That keeps both branches reading
    byte-identical inputs.
    """

    def __init__(self, dim: int, n_layers: int = 3, use_chirality: bool = False):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, dim) for v in ATOM_VOCAB])
        self.chirality = (
            nn.Embedding(CHIRALITY_VOCAB, dim) if use_chirality else None
        )
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])

    def forward(self, graph):
        h = sum(
            embedding(graph["feat"][:, i])
            for i, embedding in enumerate(self.embeddings)
        )
        if self.chirality is not None:
            h = h + self.chirality(graph["feat"][:, len(ATOM_VOCAB)])
        src, dst = graph["edge"]
        bonded = graph["eattr"] > 0
        src, dst = src[bonded], dst[bonded]
        degree = (
            torch.zeros(len(h), 1, device=h.device)
            .index_add_(0, dst, torch.ones(len(dst), 1, device=h.device))
            .clamp(min=1)
        )
        for layer, norm in zip(self.layers, self.norms):
            neighbours = torch.zeros_like(h).index_add_(0, dst, h[src]) / degree
            h = norm(F.relu(layer(h + neighbours)))
        return h


class DrugBANProteinEncoder(nn.Module):
    """DrugBAN's protein branch: a 1-D CNN over the amino-acid sequence.

    Kernel widths 3/6/9 follow the published architecture. Residues arrive
    flattened across the batch, so they are densified to (batch, length) for the
    convolutions and scattered back, leaving the fusion layer reading exactly
    the layout it reads from ProteinEncoder.

    Padding positions are zeroed after every layer. Without that, convolutions
    would smear real residues into padding and back, and a prediction would
    depend on the longest protein in the batch -- the same defect the masked BAN
    layer exists to prevent.
    """

    def __init__(self, dim: int, kernels=(3, 6, 9)):
        super().__init__()
        self.aa = nn.Embedding(21, dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2)
                for k in kernels
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(dim) for _ in kernels])

    def forward(self, graph):
        batch, index = graph["batch"], graph["idx"]
        n_graphs = int(batch.max()) + 1
        length = int(index.max()) + 1
        h = self.aa(graph["feat"])

        dense = h.new_zeros(n_graphs, length, h.size(-1))
        dense[batch, index] = h
        mask = torch.zeros(n_graphs, length, dtype=torch.bool, device=h.device)
        mask[batch, index] = True

        x = dense.transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x)[..., :length]))
            x = x * mask[:, None, :]
        return x.transpose(1, 2)[batch, index]


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
    # drug geometry, protein geometry, shuffle drug coords, shuffle protein coords.
    #
    # The no_*_3d family switches the geometry pathway off. The shuffled_*
    # family leaves architecture and capacity untouched and randomises which
    # node holds which coordinate instead; see
    # shuffle_positions_within_graphs for why that is the stronger control.
    ABLATIONS = {
        "full": (True, True, False, False),
        "no_drug_3d": (False, True, False, False),
        "no_protein_3d": (True, False, False, False),
        "no_3d": (False, False, False, False),
        "shuffled_coords": (True, True, True, True),
        "shuffled_drug_coords": (True, True, True, False),
        "shuffled_protein_coords": (True, True, False, True),
    }

    def __init__(
        self,
        dim=128,
        drug_layers=4,
        protein_layers=4,
        ban_dim=256,
        heads=2,
        ablation="full",
        drug_chirality=False,
    ):
        super().__init__()
        if ablation not in self.ABLATIONS:
            raise ValueError(f"Unknown ablation {ablation!r}; choose {sorted(self.ABLATIONS)}")
        (
            drug_geometry,
            protein_geometry,
            shuffle_drug,
            shuffle_protein,
        ) = self.ABLATIONS[ablation]
        self.drug = DrugEncoder(
            dim, drug_layers, drug_geometry, shuffle_drug, drug_chirality
        )
        self.protein = ProteinEncoder(
            dim, protein_layers, protein_geometry, shuffle_protein
        )
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


class DrugBAN2D(nn.Module):
    """The 2D baseline: DrugBAN's encoders feeding E3DTI's fusion unchanged.

    This is a reimplementation inside this pipeline, not the official
    repository, and that is a deliberate trade. docs/experiment_protocol.md
    lists six things that must match before a comparison means anything --
    split membership, seeds, early-stopping rule, decision threshold, metric
    implementation and duplicate handling. Running through the same loader,
    trainer and metrics makes all six identical by construction rather than by
    hand. The official runner additionally selects on the test set in its
    cross-domain path, so its unmodified output could not be compared anyway.

    What is given up is comparability with the published numbers. Those remain
    a separate exercise; see docs/experiment_protocol.md. Do not put the two in
    one table.

    `dim` is the lever for the capacity control: the check-in feedback asked for
    a baseline matched on parameter count, so that a win for E3DTI cannot be
    explained by it simply being the bigger model.
    """

    def __init__(
        self, dim=128, drug_layers=3, ban_dim=256, heads=2, drug_chirality=False
    ):
        super().__init__()
        self.drug = DrugBANDrugEncoder(dim, drug_layers, drug_chirality)
        self.protein = DrugBANProteinEncoder(dim)
        self.bcn = weight_norm(
            MaskedBANLayer(dim, dim, ban_dim, heads), name="h_mat", dim=None
        )
        self.classifier = MLPDecoder(ban_dim, 512, 128)

    forward = E3DTI.forward


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
