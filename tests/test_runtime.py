from pathlib import Path
import tempfile
import unittest

from milho_experiment.runtime import ProjectPaths, load_config
from milho_experiment.validation import assert_target_not_in_features, grouped_folds


class RuntimeTests(unittest.TestCase):
    def test_grouped_folds_never_mix_a_parcel(self):
        groups = [1, 1, 2, 2, 3, 3, 4, 4]
        folds = list(grouped_folds(groups, n_splits=4))
        self.assertEqual(len(folds), 4)
        for train, test in folds:
            self.assertFalse(set(groups[i] for i in train) & set(groups[i] for i in test))

    def test_target_guard(self):
        with self.assertRaises(ValueError):
            assert_target_not_in_features(["NDVI", "Produtividade"], "Produtividade")
        assert_target_not_in_features(["NDVI", "phys_Produtividade"], "Produtividade", hybrid=True)

    def test_config_requires_execution_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text("[experiment]\nrun_id='x'\n[execution]\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_paths_expand_environment_independent_markers(self):
        paths = ProjectPaths.discover(Path.cwd())
        self.assertEqual(paths.expand("{root}/docs").parent, paths.root)

