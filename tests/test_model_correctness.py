import unittest

import torch

from e3dti.model import E3DTI, collate_graphs
from e3dti.testing import synthetic_drug, synthetic_protein, transform_graph


class ModelCorrectnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = E3DTI(
            dim=32, drug_layers=1, protein_layers=1
        ).eval()
        self.drug = synthetic_drug(5, seed=1)
        self.protein = synthetic_protein(20, seed=2)

    def predict(self, drugs, proteins):
        with torch.no_grad():
            return self.model(
                collate_graphs(drugs), collate_graphs(proteins)
            )[0]

    def test_prediction_is_independent_of_padding_in_other_batch_items(self):
        solo = self.predict([self.drug], [self.protein])[0]
        mixed = self.predict(
            [self.drug, synthetic_drug(12, seed=3)],
            [self.protein, synthetic_protein(40, seed=4)],
        )[0]
        torch.testing.assert_close(solo, mixed, atol=1e-5, rtol=1e-5)

    def test_prediction_is_e3_invariant(self):
        original = self.predict([self.drug], [self.protein])
        transformed = self.predict(
            [transform_graph(self.drug, seed=5)],
            [transform_graph(self.protein, seed=6)],
        )
        torch.testing.assert_close(original, transformed, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
