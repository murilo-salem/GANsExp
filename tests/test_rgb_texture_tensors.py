import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "pipeline"))
import build_rgb_texture_tensors as tensors


class RgbTextureTensorTests(unittest.TestCase):
    def test_rgb_indices_have_stable_channel_order(self):
        rgb = np.array([[[0.2, 0.6, 0.1]]], dtype=np.float32)
        got = tensors.rgb_indices(rgb)[0, 0]
        self.assertEqual(tensors.CHANNEL_NAMES[:7], ("R", "G", "B", "VARI", "ExG", "GLI", "TGI"))
        self.assertAlmostEqual(got[1], 0.9, places=6)  # ExG

    def test_discovery_covers_and_excludes_expected_stages(self):
        sources = tensors.discover_sources(Path("/data"))
        included = {(s.season, s.stage) for s in sources if not s.exclusion}
        excluded = {(s.season, s.stage) for s in sources if s.exclusion}
        self.assertEqual(len(included), 7)
        self.assertIn(("2023_2024", "R5"), excluded)
        self.assertIn(("2025_2026", "V10"), excluded)

    def test_small_tiff_writes_22_channel_zarr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[2:14, 2:14] = [40, 100, 20]
            src = root / "rgb.tif"
            tifffile.imwrite(src, image)
            source = tensors.Source("test", "V1", "2026-01-01", src, (0, 1, 2))
            row = tensors.write_source(source, root / "out", chunk=10)
            group = tensors.zarr.open_group(root / "out" / row["zarr"], mode="r")
            self.assertEqual(group["tensor"].shape, (16, 16, 22))
            self.assertEqual(tuple(group.attrs["channel_names"]), tensors.CHANNEL_NAMES)
            self.assertTrue(group["valid_mask"][5, 5])
            self.assertTrue(np.isnan(group["tensor"][0, 0]).all())
            self.assertIn("BloscCodec", str(group["tensor"].compressors))

    def test_glcm_tile_halo_matches_full_map(self):
        q = (np.arange(225, dtype=np.int16).reshape(15, 15) % 8)
        mask = np.ones_like(q, dtype=bool)
        full, _ = tensors.glcm_window_maps_quantized(q, mask, w=7)
        # Núcleo 5:10 com halo de 4 pixels: fonte 1:14 e núcleo local 4:9.
        tile, _ = tensors.glcm_window_maps_quantized(q[1:14, 1:14], mask[1:14, 1:14], w=7)
        for name in tensors.GLCM_NAMES:
            np.testing.assert_allclose(full[name][5:10, 5:10], tile[name][4:9, 4:9])
