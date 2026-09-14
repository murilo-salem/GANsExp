"""Contratos do estágio R2->R5 que impedem vazamento temporal."""

import numpy as np
import pandas as pd


from milho_experiment.pipeline.stage_05_synthesis import stage9_rfinal_forecast as MODULE


def test_affine_normalizer_maps_target_median_to_source():
    source = [np.full((4, 4, 3), .6, np.float32)]
    target = [np.full((4, 4, 3), .3, np.float32)]
    normalizer = MODULE.AffineNormalizer.fit(source, target)
    assert np.allclose(normalizer.apply(target[0]), source[0])


def test_conditioning_preserves_expected_channel_count():
    tensor = MODULE.to_cond_tensor(np.full((16, 16, 3), .5, np.float32))
    assert tensor.shape == (7, 16, 16)


def test_prepare_keeps_target_r5_out_of_training_manifest(tmp_path):
    def stage4(name):
        root = tmp_path / name; (root / "npy").mkdir(parents=True)
        rows = []
        for stage in ("R2", "R5"):
            file = root / "npy" / f"{stage}.npy"; np.save(file, np.zeros((4, 4, 3), np.float32))
            rows.append({"fid": 1, "stage": stage, "npy": str(file.relative_to(root)), "dose_n": 0, "bloco": 1})
        pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
        return root
    out = tmp_path / "dataset"
    MODULE.command_prepare(type("Args", (), {"source_stage4": stage4("source"), "target_stage4": stage4("target"), "out": out})())
    assert "path_r5" in pd.read_csv(out / "source_pairs_r2_r5.csv")
    assert "path" in pd.read_csv(out / "target_r2_unlabeled.csv")
    assert "R5" not in pd.read_csv(out / "target_r2_unlabeled.csv").stage.tolist()


def test_group_calibration_oof_excludes_heldout_block():
    rows = []
    for block in range(1, 5):
        for x in (block, block + .5):
            rows.append({"bloco": block, "biomassa_predita": x, "biomassa_real": 100 + 10*x})
    data = pd.DataFrame(rows)
    predicted, folds = MODULE.group_calibration_oof(data, "biomassa_predita")
    assert np.allclose(predicted, data.biomassa_real)
    assert all(fold["n_calibration"] == 6 and fold["n_test"] == 2 for fold in folds)
