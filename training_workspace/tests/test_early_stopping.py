import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from meshGraphNet_self.early_stopping import (
    reference_from_checkpoint,
    update_early_stopping,
)
from meshGraphNet_self.training import checkpoint_state, restore_checkpoint


class EarlyStoppingTests(unittest.TestCase):
    def test_tiny_exact_improvement_does_not_reset_patience(self):
        result = update_early_stopping(0.999, 1.0, 1.0, 4, 0.002)

        self.assertTrue(result.exact_improvement)
        self.assertFalse(result.meaningful_improvement)
        self.assertEqual(result.epochs_without_improvement, 5)
        self.assertEqual(result.reference_loss, 1.0)
        self.assertEqual(result.best_valid_loss, 0.999)

    def test_threshold_improvement_resets_patience(self):
        result = update_early_stopping(0.998, 1.0, 1.0, 4, 0.002)

        self.assertTrue(result.meaningful_improvement)
        self.assertEqual(result.epochs_without_improvement, 0)
        self.assertEqual(result.reference_loss, 0.998)

    def test_first_validation_initializes_both_best_values(self):
        result = update_early_stopping(0.5, math.inf, math.inf, 0, 0.002)

        self.assertTrue(result.exact_improvement)
        self.assertTrue(result.meaningful_improvement)
        self.assertEqual(result.best_valid_loss, 0.5)
        self.assertEqual(result.reference_loss, 0.5)
        self.assertEqual(result.relative_improvement, 1.0)

    def test_legacy_checkpoint_uses_exact_best_as_reference(self):
        self.assertEqual(reference_from_checkpoint({}, 0.25), 0.25)

    def test_transolver_curve_stops_at_epoch_88_but_keeps_exact_best(self):
        values = [
            (79, 0.023280902321968954),
            (80, 0.023304390547467375),
            (81, 0.02385531343366908),
            (82, 0.023906205935697446),
            (83, 0.023265274478978124),
            (84, 0.02371963376812551),
            (85, 0.02304793785763883),
            (86, 0.023142618491128943),
            (87, 0.022995012351726664),
            (88, 0.0228934765019362),
        ]
        best = reference = 0.02290656498695242
        wait = 0
        exact_best_epoch = 78
        stop_epoch = None

        for epoch, value in values:
            update = update_early_stopping(value, best, reference, wait, 0.002)
            best, reference, wait = (
                update.best_valid_loss,
                update.reference_loss,
                update.epochs_without_improvement,
            )
            if update.exact_improvement:
                exact_best_epoch = epoch
            if wait >= 10:
                stop_epoch = epoch
                break

        self.assertEqual(stop_epoch, 88)
        self.assertEqual(exact_best_epoch, 88)
        self.assertAlmostEqual(reference, 0.02290656498695242)

    def test_checkpoint_roundtrip_preserves_early_stopping_reference(self):
        model = torch.nn.Linear(1, 1)
        optimizer = Adam(model.parameters(), lr=1.0e-4)
        scheduler = ReduceLROnPlateau(optimizer)
        state = checkpoint_state(
            3,
            30,
            model,
            optimizer,
            scheduler,
            0.2,
            {},
            epochs_without_improvement=4,
            early_stopping_reference_loss=0.25,
        )
        self.assertEqual(state["early_stopping_reference_loss"], 0.25)

        with patch("meshGraphNet_self.training.torch.load", return_value=state):
            restored_model = torch.nn.Linear(1, 1)
            restored_optimizer = Adam(restored_model.parameters(), lr=1.0e-4)
            restored_scheduler = ReduceLROnPlateau(restored_optimizer)
            restored = restore_checkpoint(
                Path("checkpoint.pt"),
                restored_model,
                restored_optimizer,
                restored_scheduler,
                torch.device("cpu"),
            )

        self.assertEqual(restored[3], 4)
        self.assertEqual(restored[4], 0.25)


if __name__ == "__main__":
    unittest.main()
