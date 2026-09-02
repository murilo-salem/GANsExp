import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "pipeline"))
import build_5band as features


class Build5BandTests(unittest.TestCase):
    def test_discovers_new_2324_bgr_pairs_with_explicit_channel_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Ortho_BGRv6_2023_2024.tif").touch()
            (root / "RRENIR_v6_2023_2024.tif").touch()
            pairs = features.find_pairs(root)
            self.assertEqual(pairs["V6"][0].name, "Ortho_BGRv6_2023_2024.tif")
            self.assertEqual(pairs["V6"][1], (2, 1, 0))

    def test_stack5_reorders_bgr_to_blue_green_red(self):
        bgr = np.array([[[10, 20, 30]]], dtype=np.uint16)
        rrenir = np.array([[[40, 50, 60]]], dtype=np.uint16)
        stack = features.stack5(bgr, (2, 1, 0), rrenir, size=1)
        self.assertGreater(stack[0, 0, 0], 0)  # azul
        self.assertGreater(stack[0, 0, 1], stack[0, 0, 0])  # verde > azul
        self.assertGreater(stack[0, 0, 2], 0)  # vermelho RRENIR

    def test_features_add_rgb_texture_namespaces(self):
        rng = np.random.default_rng(42)
        refl5 = rng.uniform(0.1, 0.9, size=(32, 32, 5)).astype(np.float32)
        refl5[..., 4] = 0.8
        refl5[..., 2] = 0.2
        result = features.parcel_features_5b(refl5)
        self.assertIn("rgb_glcm_entropy", result)
        self.assertIn("rgb_glcmw7_homogeneity", result)
