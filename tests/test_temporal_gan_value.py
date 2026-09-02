from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location
import tempfile
import unittest

import numpy as np
import pandas as pd

from milho_experiment.temporal import (
    HORIZONS,
    TemporalRecord,
    build_records,
    expected_input_channels,
    paired_stratified_bootstrap_delta,
    temporal_input,
)

STAGE12_SPEC = spec_from_file_location(
    "stage12_temporal", Path(__file__).parents[1] / "code/pipeline/stage12_temporal_gan_value.py")
stage12 = module_from_spec(STAGE12_SPEC)
STAGE12_SPEC.loader.exec_module(stage12)


class TemporalDatasetTests(unittest.TestCase):
    def _fixture(self, root: Path):
        stage4 = root / "stage4"
        (stage4 / "npy").mkdir(parents=True)
        rows = []
        field = []
        stages = ("V6", "V8", "V13", "R2", "R5")
        for fid, (block, dose) in enumerate(((1, 0), (2, 50)), start=1):
            for position, stage in enumerate(stages):
                name = f"{stage}_p{fid}.npy"
                image = np.zeros((8, 8, 3), np.float32)
                image[..., 0] = 0.1 + position * 0.01
                image[..., 1] = 0.2 + fid * 0.01
                image[..., 2] = 0.7
                np.save(stage4 / "npy" / name, image)
                rows.append({"fid": fid, "stage": stage, "bloco": block,
                             "dose_n": dose, "npy": f"npy/{name}"})
                field.append({"Estagio": stage, "Bloco": block, "Dose_N": dose,
                              "Biomassa": 100 * position + fid,
                              "Produtividade": 1000 + fid})
        pd.DataFrame(rows).to_csv(stage4 / "manifest.csv", index=False)
        table = root / "field.xlsx"
        pd.DataFrame(field).to_excel(table, index=False)
        return stage4, table

    def test_records_use_target_stage_biomass(self):
        with tempfile.TemporaryDirectory() as directory:
            stage4, table = self._fixture(Path(directory))
            v13 = build_records(stage4, table, "V13")
            r5 = build_records(stage4, table, "R5")
            self.assertEqual(v13[0].history_stages, HORIZONS["V13"][0])
            self.assertEqual(r5[0].history_stages, HORIZONS["R5"][0])
            self.assertEqual(v13[0].biomass, 201.0)
            self.assertEqual(r5[0].biomass, 401.0)

    def test_temporal_channel_order_and_count(self):
        first = np.zeros((4, 4, 3), np.float32)
        second = np.ones((4, 4, 3), np.float32)
        got = temporal_input((first, second), ())
        self.assertEqual(got.shape, (4, 4, 6))
        np.testing.assert_array_equal(got[..., :3], first)
        np.testing.assert_array_equal(got[..., 3:], second)
        with_aux = temporal_input((first + 0.1, second), ("NDVI", "glcm_asm"))
        self.assertEqual(with_aux.shape[-1], 10)
        self.assertEqual(expected_input_channels(("V6", "V8"), ("NDVI", "glcm_asm")), 10)

    def test_paired_block_bootstrap_detects_clear_gain(self):
        y = np.arange(24, dtype=float)
        blocks = np.repeat(np.arange(1, 5), 6)
        baseline = np.full(24, y.mean())
        candidate = y.copy()
        lo, hi = paired_stratified_bootstrap_delta(
            y, baseline, candidate, blocks, n_bootstrap=500, seed=3)
        self.assertGreater(lo, 0)
        self.assertGreater(hi, lo)

    def test_hybrid_rows_keep_real_synthetic_twins_in_same_group(self):
        records = [
            TemporalRecord(1, 1, 0, ("V6", "V8"), "V13", (), Path("x"), 10, 100),
            TemporalRecord(2, 2, 50, ("V6", "V8"), "V13", (), Path("y"), 20, 200),
        ]
        bundle = {
            1: {"H": np.array([1, 2]), "T": np.array([3]), "S": np.array([4])},
            2: {"H": np.array([5, 6]), "T": np.array([7]), "S": np.array([8])},
        }
        x, y, groups, test = stage12.scenario_matrices(
            "augment_hybrid", records, records, bundle, bundle, "Biomassa", False)
        self.assertEqual(x.shape, (4, 1))
        np.testing.assert_array_equal(y, [10, 20, 10, 20])
        np.testing.assert_array_equal(groups, [1, 2, 1, 2])
        np.testing.assert_array_equal(test.ravel(), [3, 7])

        x, _, groups, test = stage12.scenario_matrices(
            "combined", records, records, bundle, bundle, "Produtividade", False)
        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(test.shape, (2, 3))
        np.testing.assert_array_equal(groups, [1, 2, 1, 2])


if __name__ == "__main__":
    unittest.main()
