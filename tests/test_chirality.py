"""The drug branch's blindness to chirality, and the fix for it.

E(3) includes reflections. The drug encoder's messages are built from
interatomic distances, and a reflection preserves every distance, so the branch
cannot distinguish a molecule from its mirror image. That is a property of the
architecture, not a bug in it -- but it matters here, because roughly 45% of the
drugs in every benchmark split carry a defined stereocentre, and enantiomers
routinely differ in activity.

These tests pin both halves: that the default model really is blind, and that
`drug_chirality=True` really does cure it.
"""

import unittest

import torch

from e3dti.model import E3DTI, collate_graphs
from e3dti.testing import reflect_graph, synthetic_drug, synthetic_protein


class ReflectionBlindnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.drug = synthetic_drug(8, seed=1)
        self.protein = synthetic_protein(20, seed=2)

    def encode(self, model, drug):
        with torch.no_grad():
            return model.drug(collate_graphs([drug]))

    def test_the_default_drug_branch_cannot_see_a_reflection(self):
        """Documented limitation, asserted rather than claimed. `atol=0`: the
        embeddings are not merely close, they are the same numbers, because
        the reflected molecule produces an identical distance matrix."""
        model = E3DTI(dim=32, drug_layers=2, protein_layers=1).eval()
        original = self.encode(model, self.drug)
        mirrored = self.encode(model, reflect_graph(self.drug))
        torch.testing.assert_close(original, mirrored, atol=0, rtol=0)

    def test_the_chirality_tag_breaks_the_tie(self):
        """With the tag, two molecules identical in every way except their
        stereocentre must embed differently, or the feature is doing nothing."""
        model = E3DTI(
            dim=32, drug_layers=2, protein_layers=1, drug_chirality=True
        ).eval()

        left = self.drug
        right = {**self.drug, "feat": self.drug["feat"].clone()}
        # Flip one atom from clockwise to counter-clockwise.
        right["feat"][0, -1] = 2
        left = {**left, "feat": left["feat"].clone()}
        left["feat"][0, -1] = 1

        self.assertFalse(
            torch.allclose(
                self.encode(model, left), self.encode(model, right), atol=1e-6
            ),
            "the chirality tag is not reaching the drug encoder",
        )

    def test_the_chirality_tag_keeps_rotation_and_translation_invariance(self):
        """The tag must fix reflection blindness without costing the property
        the architecture was chosen for. It comes from the molecular graph, not
        the coordinates, so moving the molecule still changes nothing."""
        model = E3DTI(
            dim=32, drug_layers=2, protein_layers=1, drug_chirality=True
        ).eval()
        generator = torch.Generator().manual_seed(3)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
        if torch.det(rotation) < 0:  # keep it a pure rotation, not a reflection
            rotation[:, 0] = -rotation[:, 0]
        moved = {
            **self.drug,
            "pos": self.drug["pos"] @ rotation + torch.tensor([5.0, -2.0, 1.0]),
        }
        torch.testing.assert_close(
            self.encode(model, self.drug),
            self.encode(model, moved),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_the_tag_is_inert_when_switched_off(self):
        """Every result produced before the eighth column existed must remain
        reproducible, so the column must change nothing unless asked for."""
        model = E3DTI(dim=32, drug_layers=2, protein_layers=1).eval()
        tagged = {**self.drug, "feat": self.drug["feat"].clone()}
        tagged["feat"][:, -1] = 2
        torch.testing.assert_close(
            self.encode(model, self.drug),
            self.encode(model, tagged),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
