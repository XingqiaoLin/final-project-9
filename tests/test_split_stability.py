"""The cluster validation carve-out must not depend on the training seed.

docs/experiment_protocol.md: "Use the same fixed CSV split for every seed.
Seeds change initialization, dropout, and batch order, not train/validation/test
membership."

Before this was enforced, read_splits() passed the training seed as the
random_state of the source_train.csv carve-out, so seeds 1/2/3 produced
near-disjoint validation sets (9 of 977 biosnap/cluster validation pairs were
common to all three).  Seed spread on cluster splits then measured model
variance *and* split-resampling variance together, on exactly the splits that
carry the project's out-of-domain claim.
"""

import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from e3dti.data import pair_set, read_splits


def write_cluster_split(root, n_source=200, n_target=40):
    directory = Path(root) / "main" / "demo" / "cluster"
    directory.mkdir(parents=True)

    # Every (SMILES, Protein) pair is distinct, so a pair appearing in two
    # frames means the split leaked rather than that the fixture repeated one.
    source = pd.DataFrame(
        {
            "SMILES": [f"C{i}" for i in range(n_source)],
            "Protein": [f"PROT{i % 23}" for i in range(n_source)],
            "Y": [i % 2 for i in range(n_source)],
        }
    )
    target = pd.DataFrame(
        {
            "SMILES": [f"N{i}" for i in range(n_target)],
            "Protein": [f"TGT{i % 11}" for i in range(n_target)],
            "Y": [i % 2 for i in range(n_target)],
        }
    )
    source.to_csv(directory / "source_train.csv", index=False)
    target.to_csv(directory / "target_test.csv", index=False)
    return root


class ClusterSplitStabilityTests(unittest.TestCase):
    def test_read_splits_takes_no_seed(self):
        """A seed parameter is how the bug was reintroduced last time."""
        parameters = list(inspect.signature(read_splits).parameters)
        self.assertNotIn(
            "seed",
            parameters,
            "read_splits must not accept a seed: the validation carve-out has "
            "to be identical for every training seed.",
        )

    def test_cluster_validation_membership_is_stable(self):
        with tempfile.TemporaryDirectory() as root:
            write_cluster_split(root)
            first = read_splits(root, "demo", "cluster")
            second = read_splits(root, "demo", "cluster")

            self.assertEqual(pair_set(first["val"]), pair_set(second["val"]))
            self.assertEqual(pair_set(first["train"]), pair_set(second["train"]))

    def test_cluster_train_and_validation_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            write_cluster_split(root)
            frames = read_splits(root, "demo", "cluster")

            self.assertEqual(
                pair_set(frames["train"]) & pair_set(frames["val"]),
                set(),
                "the carve-out must partition source_train, not sample from it",
            )
            self.assertEqual(
                len(frames["train"]) + len(frames["val"]),
                200,
                "every source_train row must land in exactly one of train/val",
            )


if __name__ == "__main__":
    unittest.main()
