from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import pandas as pd

from milho_experiment.architecture_models import make_discriminators, make_generator


SPEC = spec_from_file_location(
    "stage14_architecture", Path(__file__).parents[1] / "code/pipeline/stage14_architecture_search.py")
stage14 = module_from_spec(SPEC)
sys.modules[SPEC.name] = stage14
SPEC.loader.exec_module(stage14)


@pytest.mark.parametrize("name,channels", [
    ("unet_l1", 3),
    ("unet_light", 3),
    ("resnet9", 3),
    ("attention_unet", 3),
    ("multiscale", 3),
    ("wgangp", 3),
    ("convlstm_lite", 6),
    ("simvp_lite", 6),
])
def test_generators_preserve_spatial_shape(name, channels):
    model = make_generator(name, channels, width=8).eval()
    with torch.no_grad():
        result = model(torch.zeros(1, channels, 64, 64))
    assert result.shape == (1, 3, 64, 64)
    assert torch.isfinite(result).all()


def test_multiscale_is_the_only_two_discriminator_model():
    assert len(make_discriminators("unet_light", 3, width=8)) == 1
    assert len(make_discriminators("multiscale", 3, width=8)) == 2
    assert "wgangp" not in stage14.DEFAULT_TRACK_MODELS["A"]


def test_affine_mapping_recovers_per_channel_relation(tmp_path):
    records = []
    for fid in range(2):
        source = np.linspace(0, 0.4, 8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)
        target = np.clip(source * np.array([1.2, 0.8, 1.1]) +
                         np.array([0.02, 0.03, 0.01]), 0, 1).astype(np.float32)
        mask = np.ones((8, 8), np.float32)
        source_path, target_path, mask_path = (tmp_path / f"{kind}{fid}.npy"
                                                for kind in ("s", "t", "m"))
        np.save(source_path, source)
        np.save(target_path, target)
        np.save(mask_path, mask)
        records.append(stage14.SearchRecord(fid, fid + 1, 0, (source_path,), target_path,
                                             mask_path, 1, 1, 1))
    model = stage14.AffineGenerator.fit(records, seed=7)
    np.testing.assert_allclose(model.predict([source]), target, atol=1e-5)


def test_fixed_extra_trees_contract():
    pipeline = stage14.extra_trees(seed=11, n_features=12)
    model = pipeline.named_steps["model"]
    assert pipeline.named_steps["select"].k == 8
    assert model.n_estimators == 500
    assert model.min_samples_leaf == 2
    assert model.max_features == 1.0


def test_holm_adjustment_and_weight_guard(tmp_path):
    adjusted = stage14.holm_adjust([0.01, 0.04, 0.2])
    assert adjusted == pytest.approx([0.03, 0.08, 0.2])
    stage14.assert_no_weights(tmp_path)
    forbidden = tmp_path / "accidental.pth"
    forbidden.write_bytes(b"weights")
    with pytest.raises(AssertionError, match="proibidos"):
        stage14.assert_no_weights(tmp_path)


def test_registry_has_separate_tracks_and_oracle_cannot_be_neural():
    assert stage14.BASELINES == {"A": "identity", "B": "history"}
    assert "oracle_real" not in stage14.NEURAL_MODELS
    assert set(stage14.TRACK_MODELS["A"]).isdisjoint({"convlstm_lite", "simvp_lite"})


def test_summary_accepts_a_baseline_only_smoke_run():
    predictions = pd.DataFrame(
        [{"track": "A", "target": "Biomassa", "scenario": "rs_only",
          "model": "identity", "seed": 7, "unit_id": f"u{unit}",
          "block": 1, "y": float(unit), "prediction": float(unit)}
         for unit in range(3)] +
        [{"track": "A", "target": "Biomassa", "scenario": "dose_only",
          "model": "dose_only", "seed": 7, "unit_id": f"u{unit}",
          "block": 1, "y": float(unit), "prediction": float(unit)}
         for unit in range(3)])
    images = pd.DataFrame(columns=["track", "model", "psnr", "ssim", "sam"])
    assert stage14.summarize(predictions, images, bootstrap=10).empty


def test_generated_images_are_masked_before_features(tmp_path):
    source = np.ones((8, 8, 3), np.float32) * 0.5
    target = source.copy()
    mask = np.zeros((8, 8), np.float32)
    mask[2:6, 2:6] = 1
    paths = [tmp_path / name for name in ("source.npy", "target.npy", "mask.npy")]
    for path, value in zip(paths, (source, target, mask)):
        np.save(path, value)
    record = stage14.SearchRecord(1, 1, 0, (paths[0],), paths[1], paths[2], 1, 10, 100)
    result = stage14.generate("identity", [record], torch.device("cpu"))[1]
    assert np.all(result[mask == 0] == 0)
    assert np.all(result[mask == 1] == 0.5)


def test_evaluation_aggregates_recortes_by_block_and_dose(tmp_path, monkeypatch):
    records = [
        stage14.SearchRecord(1, 1, 50, (), Path("t1"), None, 1, 10, 100),
        stage14.SearchRecord(2, 1, 50, (), Path("t2"), None, 1, 10, 100),
    ]
    monkeypatch.setattr(stage14, "_representation_features",
                        lambda track, record, synthetic: np.array([record.fid, 2.0]))
    matrix, target, units = stage14.evaluation_units(
        "A", records, None, "Biomassa", "rs_plus_dose")
    np.testing.assert_array_equal(matrix, [[1.5, 2.0, 50.0]])
    np.testing.assert_array_equal(target, [10.0])
    assert units == [{"unit_id": "b1_d50", "block": 1, "dose_n": 50.0, "n_fids": 2}]
