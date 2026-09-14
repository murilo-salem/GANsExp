"""Primitivas de validação que evitam vazamento entre estágios da mesma parcela."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def grouped_folds(groups: np.ndarray, n_splits: int = 4):
    """Gera folds e falha se uma parcela aparecer nos dois lados do split."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("são necessárias ao menos duas parcelas independentes")
    splitter = GroupKFold(n_splits=min(n_splits, len(unique)))
    for train, test in splitter.split(np.zeros(len(groups)), groups=groups):
        overlap = set(groups[train]) & set(groups[test])
        if overlap:
            raise AssertionError(f"vazamento de parcela entre treino e teste: {sorted(overlap)}")
        yield train, test


def assert_target_not_in_features(feature_names: list[str], target: str, hybrid: bool = False) -> None:
    """Bloqueia alvo e co-medida homônima em cenários de RS puro."""
    forbidden = {target, f"phys_{target}"}
    found = forbidden & set(feature_names)
    if found and not hybrid:
        raise ValueError(f"vazamento de alvo em X: {sorted(found)}")

