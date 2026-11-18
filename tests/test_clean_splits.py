"""Guards for the opt-in split cleaning.

The cleaned split is what human/cold results would be reported on, so the two
things that make it defensible have to hold: no leakage survives, and the test
set is not quietly shrunk to achieve that.
"""

import unittest

import pandas as pd

from e3dti.data import audit_splits, clean_splits, pair_set


def frame(pairs):
    return pd.DataFrame(
        {
            "SMILES": [smiles for smiles, _ in pairs],
            "Protein": [protein for _, protein in pairs],
            "Y": [i % 2 for i in range(len(pairs))],
        }
    )


class CleanSplitsTests(unittest.TestCase):
    def setUp(self):
        # train repeats ("C1", "P1") and also shares it with val;
        # ("C3", "P3") sits in both val and test, the human/cold defect.
        self.frames = {
            "train": frame(
                [("C1", "P1"), ("C1", "P1"), ("C2", "P2"), ("C5", "P5")]
            ),
            "val": frame([("C1", "P1"), ("C3", "P3"), ("C4", "P4")]),
            "test": frame([("C3", "P3"), ("C6", "P6"), ("C7", "P7")]),
        }

    def test_no_leakage_survives(self):
        cleaned, _ = clean_splits(self.frames)
        audit = audit_splits(cleaned)
        self.assertEqual(
            audit,
            {
                "within_train_duplicates": 0,
                "train_val_overlap": 0,
                "train_test_overlap": 0,
                "val_test_overlap": 0,
            },
        )

    def test_cross_split_cleaning_never_costs_the_test_set_a_row(self):
        """Overlaps are resolved on the earlier frame. Test is where the
        reported numbers come from and human/cold's is only 311 rows, so
        dropping from it would cost power on the very comparison being made."""
        cleaned, _ = clean_splits(self.frames)
        pd.testing.assert_frame_equal(cleaned["test"], self.frames["test"])

    def test_exact_repeats_inside_test_are_still_removed(self):
        """The one step that does touch test. A repeated pair is counted twice
        in every metric computed over it, which is its own distortion -- on the
        real human/cold split this is 7 rows."""
        frames = dict(self.frames)
        frames["test"] = frame(
            [("C6", "P6"), ("C6", "P6"), ("C7", "P7")]
        )
        cleaned, report = clean_splits(frames)
        self.assertEqual(len(cleaned["test"]), 2)
        self.assertEqual(report["test_duplicates_removed"], 1)

    def test_the_shared_pair_leaves_validation_not_test(self):
        cleaned, _ = clean_splits(self.frames)
        self.assertNotIn(("C3", "P3"), pair_set(cleaned["val"]))
        self.assertIn(("C3", "P3"), pair_set(cleaned["test"]))

    def test_rows_that_were_never_a_problem_are_kept(self):
        """Cleaning must not quietly cost unrelated data."""
        cleaned, _ = clean_splits(self.frames)
        self.assertIn(("C2", "P2"), pair_set(cleaned["train"]))
        self.assertIn(("C5", "P5"), pair_set(cleaned["train"]))
        self.assertIn(("C4", "P4"), pair_set(cleaned["val"]))

    def test_the_report_accounts_for_every_dropped_row(self):
        cleaned, report = clean_splits(self.frames)
        for name in self.frames:
            self.assertEqual(report["rows_before"][name], len(self.frames[name]))
            self.assertEqual(report["rows_after"][name], len(cleaned[name]))
        self.assertEqual(report["train_duplicates_removed"], 1)
        self.assertEqual(report["val_rows_shared_with_test_removed"], 1)
        self.assertEqual(report["train_rows_shared_with_val_or_test_removed"], 1)

    def test_an_already_clean_split_is_unchanged(self):
        clean = {
            "train": frame([("C1", "P1"), ("C2", "P2")]),
            "val": frame([("C3", "P3")]),
            "test": frame([("C4", "P4")]),
        }
        cleaned, report = clean_splits(clean)
        for name, original in clean.items():
            pd.testing.assert_frame_equal(cleaned[name], original)
        self.assertEqual(report["rows_before"], report["rows_after"])


if __name__ == "__main__":
    unittest.main()
