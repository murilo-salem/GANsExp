from pathlib import Path

import numpy as np
import pandas as pd

from milho_experiment.pipeline.stage_10_analysis.spectral_real_vs_synthetic import (
    discover_synthetic,
    paired_metrics,
    paired_profiles,
    summarize,
)


def _fixture(root: Path):
    real_root = root / "real"
    synthetic_root = root / "synthetic"
    (real_root / "npy").mkdir(parents=True)
    synthetic_root.mkdir()
    rows = []
    for fid in (1, 2):
        real = np.full((4, 4, 3), [0.1, 0.2, 0.4], dtype=np.float32)
        real[0, 0] = 0
        synthetic = np.full((4, 4, 3), [0.12, 0.18, 0.44], dtype=np.float32)
        real_name = f"npy/R2__p{fid:02d}.npy"
        np.save(real_root / real_name, real)
        np.save(synthetic_root / f"p{fid:03d}_V13toR2_d0_b1_fake.npy", synthetic)
        rows.append({"safra": "2023_2024", "stage": "R2", "fid": fid, "npy": real_name})
    pd.DataFrame(rows).to_csv(real_root / "manifest.csv", index=False)
    return real_root / "manifest.csv", synthetic_root


def test_profiles_pair_same_season_stage_and_parcel(tmp_path):
    manifest, synthetic = _fixture(tmp_path)
    profiles = paired_profiles(manifest, synthetic, "R2", "V13", "2023_2024")

    assert len(profiles) == 2 * 2 * 3
    assert set(profiles["fid"]) == {1, 2}
    real_red = profiles[(profiles.origem == "Real") & (profiles.banda == "Vermelho")]
    np.testing.assert_allclose(real_red.reflectancia_media, 0.1, atol=1e-7)

    summary = summarize(profiles)
    assert set(summary["n"]) == {2}
    assert len(summary) == 6

    metrics = paired_metrics(profiles)
    red = metrics[(metrics.escala == "banda") & (metrics.banda == "Vermelho")].iloc[0]
    assert np.isclose(red.vies, 0.02)
    assert len(metrics[metrics.escala == "perfil"]) == 1


def test_discovery_requires_source_when_fid_has_two_predictions(tmp_path):
    _, synthetic = _fixture(tmp_path)
    np.save(synthetic / "p001_V8toR2_d0_b1_fake.npy", np.zeros((4, 4, 3), np.float32))

    try:
        discover_synthetic(synthetic, "R2")
    except ValueError as error:
        assert "--source-stage" in str(error)
    else:
        raise AssertionError("deveria rejeitar duas previsões da mesma parcela")
