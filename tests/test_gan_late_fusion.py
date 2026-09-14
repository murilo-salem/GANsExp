import numpy as np
import pandas as pd
import pytest


from milho_experiment.pipeline.stage_08_scenarios import stage15_gan_late_fusion as stage15


def test_simplex_weights_are_complete_and_allow_zero_gan():
    weights = stage15.simplex_weights(0.05)
    assert len(weights) == 231
    assert (1.0, 0.0, 0.0) in weights
    assert (0.0, 0.0, 1.0) in weights
    assert all(sum(item) == pytest.approx(1.0) for item in weights)
    assert all(value >= 0 for item in weights for value in item)


def test_tree_grid_contract_and_k_is_capped():
    config = stage15.TreeConfig(12, 3, 0.5)
    pipeline = stage15.make_tree(config, seed=11, n_features=8)
    assert pipeline.named_steps["select"].k == 8
    model = pipeline.named_steps["model"]
    assert model.n_estimators == 500
    assert model.min_samples_leaf == 3
    assert model.max_features == 0.5


def test_fusion_selects_complementary_real_and_synthetic_predictions():
    rows = []
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    predictions = {
        ("real", stage15.TREE_CONFIGS[0].name): np.array([0.0, 0.8, 1.8, 2.6]),
        ("synthetic", stage15.TREE_CONFIGS[0].name): np.array([0.4, 1.2, 2.2, 3.0]),
        ("dose", "fixed"): np.repeat(1.5, 4),
    }
    for (branch, config), values in predictions.items():
        for index, (y, value) in enumerate(zip(truth, values), start=1):
            rows.append({"branch": branch, "config": config, "unit_id": f"u{index}",
                         "block": index, "dose_n": index * 50, "n_fids": 2,
                         "y": y, "prediction": value})
    result = stage15.select_fusion(pd.DataFrame(rows), 0.05)
    assert result["weight_real"] > 0
    assert result["weight_synthetic"] > 0
    assert result["weight_dose"] == 0
    assert sum(result[key] for key in (
        "weight_real", "weight_synthetic", "weight_dose")) == pytest.approx(1)


def test_nested_partitions_never_expose_outer_block():
    records = [stage15.SearchRecord(
        fid=block * 10 + dose, block=block, dose_n=dose,
        history_paths=(), target_path=Path("unused"), mask_path=None,
        weight=1.0, biomass=1.0, productivity=1.0)
        for block in range(1, 5) for dose in range(2)]
    partitions = list(stage15.nested_partitions(records, outer_block=4))
    assert [part[0] for part in partitions] == [1, 2, 3]
    for _, train, validation in partitions:
        assert all(record.block != 4 for record in train + validation)
        assert {record.fid for record in train}.isdisjoint(
            record.fid for record in validation)


def test_summary_requires_gain_over_nested_and_historical_synthetic(monkeypatch):
    frame = pd.DataFrame({
        "block": np.repeat(np.arange(1, 5), 2),
        "y": np.arange(8, dtype=float),
        "prediction_real": np.arange(8, dtype=float) + 1,
        "prediction_synthetic": np.arange(8, dtype=float) + 0.5,
        "prediction_dose": np.repeat(3.5, 8),
        "prediction_fused": np.arange(8, dtype=float),
    })
    monkeypatch.setattr(stage15, "paired_stratified_bootstrap_delta",
                        lambda *args, **kwargs: (0.01, 0.05))
    _, decision = stage15._summarize(frame, bootstrap=10)
    assert decision["improves_point_estimate"]
    assert decision["confirmatory_supported"]
