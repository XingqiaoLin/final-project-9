"""Guards for the capacity-matched geometry control.

The check-in promised a shuffled-coordinate ablation with "architecture and
capacity unchanged". Three things have to hold for that claim to be honest, and
each is cheap to break by accident later:

1. the shuffle keeps every coordinate inside its own graph,
2. the ablated model has exactly the same parameters as the full model,
3. the shuffle actually changes the prediction, i.e. it is wired in at all.
"""

import unittest

import torch

from e3dti.model import E3DTI, collate_graphs, shuffle_positions_within_graphs
from e3dti.testing import synthetic_drug, synthetic_protein


class ShufflePermutationTests(unittest.TestCase):
    def test_coordinates_stay_inside_their_own_graph(self):
        """A coordinate borrowed from another molecule would put the atom
        hundreds of angstroms away and fall outside the RBF cutoff, which is a
        different and much weaker control than the one we claim to run."""
        graphs = [synthetic_drug(5, seed=1), synthetic_drug(9, seed=2)]
        batched = collate_graphs(graphs)
        torch.manual_seed(0)

        shuffled = shuffle_positions_within_graphs(
            batched["pos"], batched["batch"]
        )

        for graph_id in batched["batch"].unique():
            selected = batched["batch"] == graph_id
            before = batched["pos"][selected]
            after = shuffled[selected]
            # Same multiset of coordinates, possibly reordered.
            torch.testing.assert_close(
                before.sum(dim=0), after.sum(dim=0), atol=1e-5, rtol=1e-5
            )
            for row in after:
                self.assertTrue(
                    bool((before == row).all(dim=1).any()),
                    "a coordinate came from a different graph",
                )

    def test_a_single_node_graph_is_left_alone(self):
        pos = torch.tensor([[1.0, 2.0, 3.0]])
        batch = torch.tensor([0])
        torch.manual_seed(0)
        torch.testing.assert_close(
            shuffle_positions_within_graphs(pos, batch), pos
        )


class ShuffledAblationTests(unittest.TestCase):
    def setUp(self):
        self.drug = synthetic_drug(6, seed=1)
        self.protein = synthetic_protein(20, seed=2)

    def build(self, ablation):
        torch.manual_seed(7)
        return E3DTI(
            dim=32, drug_layers=1, protein_layers=1, ablation=ablation
        ).eval()

    def encode(self, model, seed):
        """Return both branch embeddings for one forward pass.

        These assertions deliberately look at the encoder outputs rather than
        the final logit. On an untrained model the geometry pathway moves the
        logit by around 1e-6 -- the RBF features enter through a freshly
        initialised MLP and the residual stream dominates -- so a
        prediction-level assertion would be measuring initialisation scale and
        would pass whether or not the shuffle were connected. The embeddings
        move by ~5e-4, four orders of magnitude clear of the tolerance.
        """
        torch.manual_seed(seed)
        with torch.no_grad():
            return (
                model.drug(collate_graphs([self.drug])),
                model.protein(collate_graphs([self.protein])),
            )

    def test_capacity_is_identical_to_the_full_model(self):
        """This is the whole point of the control: if the parameter counts or
        shapes differ, a score gap is confounded with model size and the
        ablation cannot support the claim we want to make from it."""
        full = dict(self.build("full").named_parameters())
        shuffled = dict(self.build("shuffled_coords").named_parameters())

        self.assertEqual(full.keys(), shuffled.keys())
        for name, parameter in full.items():
            self.assertEqual(
                parameter.shape,
                shuffled[name].shape,
                f"{name} changed shape under the ablation",
            )
            torch.testing.assert_close(parameter, shuffled[name])

    def test_shuffling_changes_both_branch_embeddings(self):
        """Without this, a silently disconnected shuffle would look like the
        strongest possible result -- geometry provably irrelevant -- while
        actually measuring nothing."""
        model = self.build("shuffled_coords")
        drug_first, protein_first = self.encode(model, seed=1)
        drug_second, protein_second = self.encode(model, seed=2)

        self.assertFalse(
            torch.allclose(drug_first, drug_second, atol=1e-6),
            "the shuffle is not reaching the drug encoder",
        )
        self.assertFalse(
            torch.allclose(protein_first, protein_second, atol=1e-6),
            "the shuffle is not reaching the protein encoder",
        )

    def test_full_model_is_unaffected_by_the_new_code_path(self):
        """The shuffle must be inert for every other ablation, or it would
        quietly contaminate the main results."""
        model = self.build("full")
        drug_first, protein_first = self.encode(model, seed=1)
        drug_second, protein_second = self.encode(model, seed=2)

        torch.testing.assert_close(drug_first, drug_second)
        torch.testing.assert_close(protein_first, protein_second)

    def test_each_branch_shuffles_without_disturbing_the_other(self):
        cases = {
            "shuffled_drug_coords": (True, False),
            "shuffled_protein_coords": (False, True),
        }
        for ablation, (drug_moves, protein_moves) in cases.items():
            with self.subTest(ablation=ablation):
                model = self.build(ablation)
                drug_first, protein_first = self.encode(model, seed=1)
                drug_second, protein_second = self.encode(model, seed=2)

                for name, before, after, should_move in (
                    ("drug", drug_first, drug_second, drug_moves),
                    ("protein", protein_first, protein_second, protein_moves),
                ):
                    unchanged = torch.allclose(before, after, atol=1e-6)
                    self.assertEqual(
                        not unchanged,
                        should_move,
                        f"{name} branch moved={not unchanged}, "
                        f"expected {should_move} under {ablation}",
                    )


if __name__ == "__main__":
    unittest.main()
