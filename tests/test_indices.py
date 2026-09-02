import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pytorch-CycleGAN-and-pix2pix"))

from milho_experiment.indices import (
    GAN_ATTRIBUTE_NAMES,
    attribute_channels,
    attribute_channels_5band,
    bands5_indices,
    rrenir_indices,
)


class IndicesCanonicalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.refl = rng.uniform(0.0, 1.0, size=(16, 16, 3)).astype(np.float32)
        self.refl5 = rng.uniform(0.0, 1.0, size=(16, 16, 5)).astype(np.float32)

    def test_rrenir_indices_shapes_and_ranges(self):
        idx = rrenir_indices(self.refl)
        for name in ["NDVI", "NDRE", "CIrededge", "SAVI"]:
            self.assertIn(name, idx)
            self.assertEqual(idx[name].shape, (16, 16))

    def test_attribute_channels_defaults_to_legacy_four(self):
        ch = attribute_channels(self.refl)
        self.assertEqual(ch.shape, (16, 16, 4))
        self.assertTrue(np.all((ch >= 0) & (ch <= 1)))

    def test_attribute_channels_respects_names(self):
        for names, expected in [
            (("chlorophyll",), 1),
            (("chlorophyll", "NDVI"), 2),
            (("chlorophyll", "NDVI", "NDRE", "SAVI", "EVI2"), 5),
        ]:
            ch = attribute_channels(self.refl, names=names)
            self.assertEqual(ch.shape[-1], expected)

    def test_attribute_channels_rejects_unavailable_5band_index(self):
        # GNDVI exige banda verde, ausente no RRENIR de 3 bandas.
        with self.assertRaises(ValueError):
            attribute_channels(self.refl, names=("GNDVI",))

    def test_attribute_channels_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            attribute_channels(self.refl, names=("invalid",))

    def test_attribute_channels_consistency_with_geo(self):
        """Índices usados pela GAN devem coincidir com os da pipeline agronômica."""
        import geo  # type: ignore

        # geo.py reexporta as funções canônicas; verifica identidade e valores
        from milho_experiment.indices import chlorophyll_map
        self.assertIs(geo.rrenir_indices, rrenir_indices)
        self.assertIs(geo.chlorophyll_map, chlorophyll_map)

        idx_geo = geo.rrenir_indices(self.refl)
        idx_canonical = rrenir_indices(self.refl)
        for k in idx_geo:
            np.testing.assert_allclose(idx_geo[k], idx_canonical[k], rtol=1e-6)

    def test_5band_includes_extra_indices(self):
        idx = bands5_indices(self.refl5)
        for name in ["NDVI", "GNDVI", "EVI", "EVI2", "VARI", "TGI"]:
            self.assertIn(name, idx)

    def test_attribute_channels_5band_matches_rrenir_for_common_indices(self):
        ch5 = attribute_channels_5band(self.refl5, names=("chlorophyll", "NDVI", "NDRE", "SAVI"))
        # Os índices comuns devem ser numericamente iguais aos do RRENIR puro
        # quando as bandas RRENIR correspondem às posições 2,3,4 do stack 5-bandas.
        refl3 = self.refl5[..., [2, 3, 4]]
        ch3 = attribute_channels(refl3, names=("chlorophyll", "NDVI", "NDRE", "SAVI"))
        np.testing.assert_allclose(ch5, ch3, rtol=1e-6)


class PhenologyDatasetIntegrationTests(unittest.TestCase):
    def test_dataset_accepts_variable_channels(self):
        import tempfile

        from data.phenology_dataset import PhenologyDataset  # type: ignore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            droot = root / "ds"
            for sub in ("train/input", "train/target"):
                (droot / sub).mkdir(parents=True, exist_ok=True)
            np.save(droot / "train/input/pair.npy", np.random.rand(8, 8, 3).astype(np.float32))
            np.save(droot / "train/target/pair.npy", np.random.rand(8, 8, 3).astype(np.float32))

            class Opt:
                dataroot = str(droot)
                phase = "train"
                crop_size = 8
                load_size = 8
                isTrain = True
                gan_attr_names = "chlorophyll,NDVI"
                gan_attr_normalize = "image"

            ds = PhenologyDataset(Opt())
            self.assertEqual(ds.attr_names, ("chlorophyll", "NDVI"))
            self.assertEqual(ds.n_cond, 2)
            item = ds[0]
            self.assertEqual(item["A"].shape[0], 3 + 2)  # 3 bandas + 2 condicionais


if __name__ == "__main__":
    unittest.main()
