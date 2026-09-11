from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from milho_experiment.gan_losses import (
    discriminator_features,
    feature_matching_loss,
    gradient_loss,
    masked_l1,
    multiscale_texture_loss,
    parcel_mask,
    vegetation_index_loss,
)


SPEC = spec_from_file_location(
    "stage13_loss_ablation",
    Path(__file__).parents[1] / "code/pipeline/stage13_temporal_gan_loss_ablation.py")
stage13 = module_from_spec(SPEC)
sys.modules[SPEC.name] = stage13
SPEC.loader.exec_module(stage13)


class GanLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.target = torch.rand(2, 3, 32, 32) * 0.8 + 0.1
        self.target[:, :, :4, :] = 0
        self.mask = parcel_mask(self.target)

    def test_losses_are_zero_for_identical_images(self):
        self.assertAlmostEqual(float(masked_l1(self.target, self.target, self.mask)), 0.0)
        self.assertAlmostEqual(float(vegetation_index_loss(
            self.target, self.target, self.mask)), 0.0)
        self.assertAlmostEqual(float(multiscale_texture_loss(
            self.target, self.target, self.mask, windows=(5, 11))), 0.0)
        self.assertAlmostEqual(float(gradient_loss(self.target, self.target, self.mask)), 0.0)

    def test_mask_ignores_background(self):
        fake = self.target.clone()
        fake[:, :, :4, :] = 1
        self.assertAlmostEqual(float(masked_l1(fake, self.target, self.mask)), 0.0)
        fake[:, :, 10, 10] = 1
        self.assertGreater(float(masked_l1(fake, self.target, self.mask)), 0)

    def test_texture_detects_different_spatial_pattern(self):
        target = torch.full((1, 3, 32, 32), 0.5)
        fake = target.clone()
        checker = (torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2
        fake[:, 2] = checker.float()
        mask = torch.ones((1, 1, 32, 32))
        self.assertGreater(float(multiscale_texture_loss(
            fake, target, mask, windows=(5, 11))), 0.01)

    def test_all_reconstruction_losses_propagate_to_fake(self):
        fake = self.target.clone().requires_grad_(True)
        loss = (masked_l1(fake, self.target * .9, self.mask)
                + vegetation_index_loss(fake, self.target * .9, self.mask)
                + multiscale_texture_loss(fake, self.target * .9, self.mask, windows=(5,))
                + gradient_loss(fake, self.target * .9, self.mask))
        loss.backward()
        self.assertIsNotNone(fake.grad)
        self.assertTrue(torch.isfinite(fake.grad).all())
        self.assertGreater(float(fake.grad.abs().sum()), 0)

    def test_feature_matching_detaches_real_activations(self):
        discriminator = torch.nn.Module()
        discriminator.model = torch.nn.Sequential(
            torch.nn.Conv2d(5, 4, 3, padding=1),
            torch.nn.LeakyReLU(.2),
            torch.nn.Conv2d(4, 1, 3, padding=1),
        )
        fake = torch.rand(1, 5, 8, 8, requires_grad=True)
        real = torch.rand(1, 5, 8, 8, requires_grad=True)
        _, fake_features = discriminator_features(discriminator, fake)
        _, real_features = discriminator_features(discriminator, real)
        feature_matching_loss(fake_features, real_features).backward()
        self.assertIsNotNone(fake.grad)
        self.assertIsNone(real.grad)

    def test_loss_specs_encode_exact_five_arms(self):
        specs = [stage13.LossSpec(name) for name in stage13.ARM_NAMES]
        self.assertEqual(len(specs), 5)
        self.assertFalse(specs[0].uses_gan)
        self.assertTrue(stage13.LossSpec("l1_gan").uses_gan)
        self.assertEqual(stage13.LossSpec("full").auxiliary_components,
                         ("indices", "texture", "gradient"))
        with self.assertRaises(ValueError):
            stage13.LossSpec("unknown")

    def test_holm_adjustment_is_monotone_in_sorted_order(self):
        adjusted = stage13._holm_adjust([.04, .01])
        self.assertEqual(adjusted, [.04, .02])

    def test_rotating_policy_loads_only_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "model.pth"
            resume = root / "model.resume.pth"
            final.touch()
            self.assertIsNone(stage13._checkpoint_to_load("rotating", final, resume))
            resume.touch()
            self.assertEqual(stage13._checkpoint_to_load("rotating", final, resume), resume)
            self.assertEqual(stage13._checkpoint_to_load("all", final, resume), final)
            self.assertIsNone(stage13._checkpoint_to_load("none", final, resume))

    def test_rotating_policy_removes_final_and_resume_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "model.pth"
            resume = root / "model.resume.pth"
            final.touch()
            resume.touch()
            saved = []
            stage13._finish_checkpoint("rotating", final, resume, saved.append)
            self.assertEqual(saved, [])
            self.assertFalse(final.exists())
            self.assertFalse(resume.exists())

    def test_failed_checkpoint_write_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.resume.pth"
            temporary = path.with_suffix(path.suffix + ".tmp")

            def fail_after_partial_write(_payload, destination):
                Path(destination).write_bytes(b"partial")
                raise OSError("disk full")

            stateful = SimpleNamespace(state_dict=lambda: {})
            dataset = SimpleNamespace(generator=torch.Generator().manual_seed(1))
            with patch("torch.save", side_effect=fail_after_partial_write):
                with self.assertRaises(OSError):
                    stage13._save_checkpoint(
                        path, generator=stateful, discriminator=None, opt_g=stateful,
                        opt_d=None, epoch=10, epochs=20, fingerprint="fingerprint",
                        spec=stage13.LossSpec("l1"), scales={}, history=[],
                        dataset=dataset, loader_generator=torch.Generator().manual_seed(1))
            self.assertFalse(path.exists())
            self.assertFalse(temporary.exists())

    def test_summary_writes_complete_ablation_reports(self):
        rows = []
        for target in stage13.PRIMARY_SCENARIOS:
            for arm_position, arm in enumerate(stage13.ARM_NAMES):
                for scenario_position, scenario in enumerate(stage13.SCENARIOS):
                    for seed in (7, 11, 23):
                        for fid in range(24):
                            y = float(fid + 1)
                            rows.append({
                                "target": target, "rs_mode": "rs_puro", "arm": arm,
                                "scenario": scenario, "seed": seed, "fid": fid,
                                "outer_block": fid // 6 + 1, "y": y,
                                "prediction": y + .01 * (scenario_position - arm_position),
                            })
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            stage13.summarize(rows, out, bootstrap=20)
            expected = {
                "contrasts.csv", "loss_contrasts.csv", "metrics.csv",
                "metrics_per_seed.csv", "metrics_seed_summary.csv",
                "oof_predictions_ensemble.csv", "primary_contrasts.csv",
            }
            self.assertEqual({path.name for path in out.iterdir()}, expected)
            primary = pd.read_csv(out / "primary_contrasts.csv")
            self.assertEqual(set(primary.target), set(stage13.PRIMARY_SCENARIOS))
            self.assertIn("all_seeds_positive", primary)
            self.assertIn("p_holm", primary)


if __name__ == "__main__":
    unittest.main()
