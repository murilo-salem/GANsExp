import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from milho_experiment.gan_attributes import (
    ALL_RRENIR_CANDIDATES,
    GLOBAL_GLCM_CHANNELS,
    LOCAL_GLCM_CHANNELS,
    LEGACY_RRENIR_CANDIDATES,
    RRENIR_INDEX_NAMES,
    rrenir_attribute_channels,
)
from milho_experiment.indices import attribute_channels, rrenir_indices
from milho_experiment.pipeline.stage_03_attributes.hyperspectral_features import (
    GLCM_NAMES, HARALICK_NAMES, glcm_features, glcm_window_maps,
)
from milho_experiment.temporal import AUX_CANDIDATES, temporal_input
from milho_experiment.pipeline.stage_06_datasets.stage6_features_ortho import parcel_features
from milho_experiment.pipeline.stage_08_scenarios import stage12_temporal_gan_value as stage12


class CompleteGanCandidatesTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.refl = rng.uniform(.05, .95, (16, 16, 3)).astype(np.float32)

    def test_catalog_covers_every_available_index_and_glcm(self):
        self.assertEqual(set(RRENIR_INDEX_NAMES), set(rrenir_indices(self.refl)))
        self.assertEqual(set(GLOBAL_GLCM_CHANNELS),
                         {f"glcm_global_{n}" for n in HARALICK_NAMES})
        self.assertEqual(set(LOCAL_GLCM_CHANNELS),
                         {f"glcmw{w}_{n}" for w in (3, 5, 7) for n in GLCM_NAMES})
        self.assertEqual(len(AUX_CANDIDATES), 38)
        self.assertEqual(len(set(AUX_CANDIDATES)), len(AUX_CANDIDATES))
        self.assertEqual(AUX_CANDIDATES, ALL_RRENIR_CANDIDATES)
        extracted = parcel_features(self.refl)
        self.assertEqual(set(GLOBAL_GLCM_CHANNELS),
                         {name.replace("glcm_", "glcm_global_", 1)
                          for name in extracted if name.startswith("glcm_")})
        self.assertEqual(set(LOCAL_GLCM_CHANNELS),
                         {name for name in extracted if name.startswith("glcmw")})
        channels = rrenir_attribute_channels(self.refl, AUX_CANDIDATES)
        self.assertEqual(channels.shape, (16, 16, 38))
        self.assertTrue(np.isfinite(channels).all())
        self.assertTrue(((channels >= 0) & (channels <= 1)).all())

    def test_global_planes_match_existing_descriptors_and_keep_their_value(self):
        expected = glcm_features(self.refl[..., 2], np.ones((16, 16), dtype=bool))
        raw = rrenir_attribute_channels(self.refl, GLOBAL_GLCM_CHANNELS, normalize="none")
        encoded = rrenir_attribute_channels(self.refl, GLOBAL_GLCM_CHANNELS)
        for i, name in enumerate(HARALICK_NAMES):
            value = expected[f"glcm_{name}"]
            np.testing.assert_allclose(raw[..., i], value, rtol=1e-6, atol=1e-8)
            code = .5 + np.arctan(np.sign(value) * np.log1p(abs(value))) / np.pi
            np.testing.assert_allclose(encoded[..., i], code, rtol=1e-6)
        constant = np.full_like(self.refl, .5)
        other = rrenir_attribute_channels(constant, ("glcm_global_contrast",))
        contrast = encoded[..., GLOBAL_GLCM_CHANNELS.index("glcm_global_contrast")]
        self.assertGreater(float(contrast[0, 0]), float(other[0, 0, 0]))

    def test_local_channels_use_existing_maps_and_preserve_requested_order(self):
        names = ("glcmw7_correlation", "EVI2", "glcmw3_entropy")
        channels = rrenir_attribute_channels(self.refl, names, normalize="none")
        for pos, window, metric in ((0, 7, "correlation"), (2, 3, "entropy")):
            maps, _ = glcm_window_maps(self.refl[..., 2], np.ones((16, 16), bool), window)
            np.testing.assert_allclose(channels[..., pos], np.nan_to_num(maps[metric]), rtol=1e-6)
        np.testing.assert_allclose(channels[..., 1], rrenir_indices(self.refl)["EVI2"])

    def test_legacy_checkpoints_receive_identical_channels(self):
        expected = attribute_channels(self.refl, LEGACY_RRENIR_CANDIDATES)
        actual = rrenir_attribute_channels(self.refl, LEGACY_RRENIR_CANDIDATES)
        np.testing.assert_array_equal(actual, expected)
        history = (self.refl, self.refl * .9)
        packed = temporal_input(history, ("glcm_global_entropy", "glcmw7_correlation"))
        self.assertEqual(packed.shape, (16, 16, 10))
        np.testing.assert_array_equal(packed[..., :3], history[0])
        np.testing.assert_array_equal(packed[..., 5:8], history[1])

    def test_missing_bands_invalid_names_and_small_constant_images(self):
        for name in ("GNDVI", "EVI", "VARI", "TGI", "glcmw9_entropy"):
            with self.assertRaises(ValueError):
                rrenir_attribute_channels(self.refl, (name,))
        channels = rrenir_attribute_channels(np.zeros((4, 4, 3)), AUX_CANDIDATES)
        self.assertTrue(np.isfinite(channels).all())
        with self.assertRaises(ValueError):
            rrenir_attribute_channels(self.refl, ("glcm_global_entropy",), normalize="global")
        channel = rrenir_attribute_channels(self.refl, ("glcm_global_entropy",), normalize="global",
                                            global_bounds={"glcm_global_entropy": (0., 10.)})
        self.assertTrue(((channel >= 0) & (channel <= 1)).all())

    def test_forward_selection_scores_all_candidates_internally(self):
        winner = "glcmw7_homogeneity"
        calls = []

        def score(records, target, attrs, **kwargs):
            calls.append(tuple(attrs))
            return 1. if winner in attrs else 0.

        with patch.object(stage12, "screen_score", side_effect=score):
            selected, _ = stage12.progressive_select(
                [], "Produtividade", horizon="R5", outer_block=1, max_aux=2,
                epochs=50, seed=7, batch_size=4, device="cpu", cache=Path("unused"), n_jobs=1)
        self.assertEqual(selected, (winner,))
        self.assertEqual({attrs[0] for attrs in calls if len(attrs) == 1}, set(AUX_CANDIDATES))
        with patch.object(stage12, "screen_score", side_effect=score):
            selected, _ = stage12.progressive_select(
                [], "Produtividade", horizon="R5", outer_block=1, max_aux=4,
                epochs=50, seed=7, batch_size=4, device="cpu", cache=Path("unused"), n_jobs=1,
                candidates=(winner,))
        self.assertEqual(selected, (winner,))


if __name__ == "__main__":
    unittest.main()
