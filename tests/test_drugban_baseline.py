"""Guards for the 2D baseline.

The project's claim is comparative, so the baseline has to be defensible in
three specific ways: it must be matched on capacity, it must not be reading
geometry by accident, and it must be as padding-safe as the model it is
compared against.
"""

import argparse
import unittest

import torch

from e3dti.model import E3DTI, DrugBAN2D, collate_graphs
from e3dti.testing import synthetic_drug, synthetic_protein, transform_graph
from train_e3dti import MATCHED_BASELINE_DIM, build_model


class MatchedCapacityTests(unittest.TestCase):
    def test_the_matched_baseline_is_within_two_percent_of_e3dti(self):
        """The check-in feedback asked for a control with capacity held fixed.
        If this drifts, a score gap stops being evidence about geometry and
        becomes evidence about model size."""
        e3dti = sum(
            p.numel()
            for p in E3DTI(dim=128, drug_layers=4, protein_layers=4).parameters()
        )
        baseline = sum(
            p.numel() for p in DrugBAN2D(dim=MATCHED_BASELINE_DIM).parameters()
        )
        gap = abs(baseline - e3dti) / e3dti
        self.assertLess(
            gap,
            0.02,
            f"E3DTI has {e3dti:,} parameters, matched baseline {baseline:,}, "
            f"{gap:.1%} apart",
        )

    def test_build_model_refuses_a_geometry_ablation_on_the_baseline(self):
        """`--arch drugban --ablation shuffled_coords` is meaningless and would
        silently produce a run that looks like a control but is not one."""
        args = argparse.Namespace(
            arch="drugban",
            ablation="shuffled_coords",
            dim=128,
            drug_layers=4,
            protein_layers=4,
        )
        with self.assertRaises(ValueError):
            build_model(args)


class BaselineBehaviourTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DrugBAN2D(dim=32).eval()
        self.drug = synthetic_drug(6, seed=1)
        self.protein = synthetic_protein(20, seed=2)

    def predict(self, drugs, proteins):
        with torch.no_grad():
            return self.model(collate_graphs(drugs), collate_graphs(proteins))[0]

    def test_prediction_ignores_coordinates_entirely(self):
        """The baseline's whole purpose is to be the model without geometry.
        Rotating and reflecting the input must change nothing at all -- not
        approximately, as for an equivariant model, but exactly, because the
        coordinates are never read."""
        original = self.predict([self.drug], [self.protein])
        moved = self.predict(
            [transform_graph(self.drug, seed=5)],
            [transform_graph(self.protein, seed=6)],
        )
        torch.testing.assert_close(original, moved, atol=0, rtol=0)

    def test_prediction_is_independent_of_padding_in_other_batch_items(self):
        """A convolution smears across padding unless it is masked, which would
        make a prediction depend on the longest protein batched alongside it."""
        solo = self.predict([self.drug], [self.protein])[0]
        mixed = self.predict(
            [self.drug, synthetic_drug(14, seed=3)],
            [self.protein, synthetic_protein(61, seed=4)],
        )[0]
        torch.testing.assert_close(solo, mixed, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
