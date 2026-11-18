"""Dataset loading, graph construction, and split audits for E3DTI."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split

from .model import ATOM_VOCAB, CHIRALITY_VOCAB, N_BOND, SEP_BINS, collate_graphs


ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
BONDS = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
CHIRAL_TAGS = {
    Chem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
K_NEIGHBORS = 16


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def load_drug_conformers(path):
    with gzip.open(path) as handle:
        return {
            molecule.GetProp("_Name"): molecule
            for molecule in Chem.ForwardSDMolSupplier(handle, removeHs=True)
            if molecule is not None
        }


def load_protein_structures(path):
    arrays = np.load(path)
    aa = arrays["aa"]
    xyz = arrays["ca_xyz"]
    plddt = arrays["plddt"]
    offsets = arrays["offset"]
    return {
        protein_id: ("".join(aa[start:end]), xyz[start:end], plddt[start:end])
        for protein_id, start, end in zip(
            arrays["pid"], offsets[:-1], offsets[1:]
        )
    }


def drug_graph(molecule):
    position = torch.tensor(
        molecule.GetConformer().GetPositions(), dtype=torch.float32
    )
    features = [
        [
            ELEMENTS.index(atom.GetSymbol())
            if atom.GetSymbol() in ELEMENTS
            else len(ELEMENTS),
            min(atom.GetDegree(), 6),
            min(max(atom.GetFormalCharge(), -2), 2) + 2,
            int(atom.GetIsAromatic()),
            HYBRIDIZATIONS.index(atom.GetHybridization())
            if atom.GetHybridization() in HYBRIDIZATIONS
            else len(HYBRIDIZATIONS),
            min(atom.GetTotalNumHs(), 4),
            int(atom.IsInRing()),
            # Eighth column: the tetrahedral chirality tag. Read from the
            # molecular graph rather than the coordinates, because the geometry
            # cannot carry it -- the encoders see only interatomic distances,
            # which a reflection leaves unchanged. Encoders ignore this column
            # unless asked for it, so adding it changed no existing result.
            CHIRAL_TAGS.get(atom.GetChiralTag(), 3),
        ]
        for atom in molecule.GetAtoms()
    ]
    n_atoms = len(features)
    bond = torch.zeros(n_atoms, n_atoms, dtype=torch.long)
    for item in molecule.GetBonds():
        src = item.GetBeginAtomIdx()
        dst = item.GetEndAtomIdx()
        bond_type = BONDS.get(item.GetBondType(), 1)
        bond[src, dst] = bond[dst, src] = bond_type
    src, dst = (~torch.eye(n_atoms, dtype=torch.bool)).nonzero(as_tuple=True)
    return {
        "pos": position - position.mean(0),
        "feat": torch.tensor(features, dtype=torch.long),
        "edge": torch.stack([src, dst]),
        "eattr": bond[src, dst],
    }


def protein_graph(sequence, xyz, plddt, geometry_neighbors=True):
    position = torch.tensor(xyz, dtype=torch.float32)
    position = position - position.mean(0)
    n_nodes = len(position)
    neighbors = min(K_NEIGHBORS, n_nodes - 1)
    if geometry_neighbors:
        pairwise = torch.cdist(position, position)
    else:
        sequence_index = torch.arange(n_nodes)
        pairwise = (sequence_index[:, None] - sequence_index[None, :]).abs().float()
    nearest = pairwise.topk(neighbors + 1, largest=False).indices[:, 1:]
    dst = torch.arange(n_nodes).repeat_interleave(neighbors)
    src = nearest.reshape(-1)
    separation = torch.bucketize(
        (src - dst).abs(), torch.tensor(SEP_BINS)
    )
    features = torch.tensor(
        [AMINO_ACIDS.index(aa) if aa in AMINO_ACIDS else 20 for aa in sequence]
    )
    return {
        "pos": position,
        "feat": features,
        "plddt": torch.tensor(plddt, dtype=torch.float32) / 100,
        "edge": torch.stack([src, dst]),
        "eattr": separation,
    }


# The cluster splits ship no validation file, so one is carved out of
# source_train.csv.  That carve-out must NOT depend on the training seed:
# docs/experiment_protocol.md requires train/validation/test membership to be
# identical across seeds, so that seed spread measures model variance alone.
# Keying it to the training seed gave seeds 1/2/3 near-disjoint validation sets
# (only 0.9% of biosnap/cluster validation pairs were common to all three),
# which silently mixed split-resampling variance into every reported standard
# deviation on exactly the splits carrying the out-of-domain claim.
VALIDATION_SPLIT_SEED = 0


def read_splits(data_root, dataset, split):
    directory = Path(data_root) / "main" / dataset / split
    if split == "cluster":
        source = pd.read_csv(directory / "source_train.csv")
        train, validation = train_test_split(
            source,
            test_size=0.1,
            random_state=VALIDATION_SPLIT_SEED,
            stratify=source["Y"],
        )
        frames = {
            "train": train.reset_index(drop=True),
            "val": validation.reset_index(drop=True),
            "test": pd.read_csv(directory / "target_test.csv"),
        }
    else:
        frames = {
            name: pd.read_csv(directory / f"{name}.csv")
            for name in ("train", "val", "test")
        }
    return frames


def pair_set(frame):
    return set(zip(frame["SMILES"], frame["Protein"]))


def clean_splits(frames):
    """Remove duplicate rows and cross-split pair leakage. Opt-in only.

    The shipped DrugBAN files carry three defects that this undoes, in order:

    1. the same (SMILES, Protein) pair repeated inside a frame, which silently
       upweights those pairs in the loss and in any metric computed over them;
    2. training pairs that reappear in validation or test, which is the leakage
       DrugBAN's own published numbers also carry;
    3. validation pairs that reappear in test, which is the worst of the three
       for us specifically -- validation selects both the checkpoint and the
       decision threshold, so those pairs steer the model and are then scored
       as if they had been held out.

    `human/cold` is the split this exists for: 394 duplicated training rows
    (11.5%) and 12 pairs in both validation and test (3.9% of a 311-row test
    set), on a split the check-in names as carrying the out-of-domain claim.

    Overlaps are resolved by dropping the row from the *earlier* frame -- train
    loses pairs shared with validation or test, validation loses pairs shared
    with test -- so cross-split cleaning never costs the test set a row. Test is
    where the reported numbers come from and human/cold's is only 311 rows;
    shrinking it would cost statistical power on exactly the comparison being
    made.

    Deduplication is the one step that does touch test, because an exact repeat
    is counted twice in every metric computed over it. On human/cold this takes
    test from 311 rows to 304 distinct pairs, and validation from 155 to 150
    before the 12 shared pairs are dropped.

    Cleaning makes results non-comparable with published DrugBAN numbers, which
    is why it is a flag and not the default. See docs/data_integrity_audit.md.
    """
    cleaned = {
        name: frame.drop_duplicates(["SMILES", "Protein"]).reset_index(drop=True)
        for name, frame in frames.items()
    }
    report = {
        f"{name}_duplicates_removed": len(frames[name]) - len(cleaned[name])
        for name in frames
    }

    def drop_pairs(frame, forbidden):
        if not forbidden:
            return frame, 0
        keep = ~pd.Series(
            list(zip(frame["SMILES"], frame["Protein"])), index=frame.index
        ).isin(forbidden)
        return frame[keep].reset_index(drop=True), int((~keep).sum())

    test_pairs = pair_set(cleaned["test"])
    cleaned["val"], report["val_rows_shared_with_test_removed"] = drop_pairs(
        cleaned["val"], test_pairs
    )
    cleaned["train"], report["train_rows_shared_with_val_or_test_removed"] = (
        drop_pairs(cleaned["train"], pair_set(cleaned["val"]) | test_pairs)
    )

    report["rows_before"] = {name: len(frame) for name, frame in frames.items()}
    report["rows_after"] = {name: len(frame) for name, frame in cleaned.items()}
    return cleaned, report


def audit_splits(frames):
    pairs = {name: pair_set(frame) for name, frame in frames.items()}
    return {
        "within_train_duplicates": int(
            frames["train"].duplicated(["SMILES", "Protein"]).sum()
        ),
        "train_val_overlap": len(pairs["train"] & pairs["val"]),
        "train_test_overlap": len(pairs["train"] & pairs["test"]),
        "val_test_overlap": len(pairs["val"] & pairs["test"]),
    }


class PairDataset(torch.utils.data.Dataset):
    """Pair lookup for notebook-safe workers=0 loading.

    Large graph dictionaries are intentionally shared by reference in the main
    process. On spawn-based systems, use workers=0 to avoid copying them.
    """

    def __init__(self, frame, drug_graphs, protein_graphs):
        self.drug_ids = frame["did"].tolist()
        self.protein_ids = frame["pid"].tolist()
        self.labels = frame["Y"].astype(np.float32).tolist()
        self.drug_graphs = drug_graphs
        self.protein_graphs = protein_graphs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.drug_graphs[self.drug_ids[index]],
            self.protein_graphs[self.protein_ids[index]],
            self.labels[index],
        )


def collate_pairs(batch):
    drugs, proteins, labels = zip(*batch)
    return (
        collate_graphs(drugs),
        collate_graphs(proteins),
        torch.tensor(labels),
    )
