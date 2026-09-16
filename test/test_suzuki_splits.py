"""
Tests for the two-stage Suzuki multi-split pipeline (`src/suzuki_splits.py`).

Requires rdkit (featurization). Training itself is mocked in the barrier tests;
the real training loop is exercised by running the `--stage train` command.
"""

import os
import sys
import argparse
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

try:
    from rdkit import Chem  # noqa: F401
except ImportError as e:  # pragma: no cover - environment dependent
    raise unittest.SkipTest("rdkit is not importable: %s" % e)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import suzuki_splits as ss  # noqa: E402
from data import GraphDataset  # noqa: E402
from reaction_data import get_graph_data  # noqa: E402


def make_args(data_folder, log_dir, **overrides):
    """Namespace with the defaults of `main_finetune.py` relevant to the pipeline."""
    args = argparse.Namespace(
        seed=42,
        test_ratio=0.3,
        valid_ratio=0.1,
        split_strategy=None,
        stage=None,
        split_ids=list(range(10)),
        raw_split_dir="raw/suzuki",
        split_file_pattern="random_split_{split_id}.tsv",
        processed_npz_dir="processed/suzuki/npz",
        log_dir=log_dir,
        ratio_tolerance=0.01,
        drop_invalid_rows=False,
        overwrite=False,
        overwrite_results=False,
        skip_aggregate=False,
        Data_folder=data_folder,
        monitor_folder=os.path.join(log_dir, "monitor") + os.sep,
        image_folder=os.path.join(log_dir, "images") + os.sep,
        model_path=os.path.join(log_dir, "model") + os.sep,
        model_name="model.pt",
        npz_folder="unused",
        y_column="y",
        reaction_column="rxn",
        epochs=1,
        patience=0,
        track_test_each_epoch=False,
        num_workers=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def synthetic_reactions(n=100):
    """Small, unique, parseable reactions with 2-4 reactants."""
    rxns = []
    for k in range(1, 26):
        for r in ("C", "CC", "CCC", "c1ccccc1"):
            extra = ".[Na+].[OH-]" if k % 2 == 0 else ""
            chain = "C" * k
            rxns.append("%sO.OC(=O)%s%s>>%sOC(=O)%s" % (chain, r, extra, chain, r))
    return rxns[:n]


def write_synthetic_splits(data_folder, split_ids, n=100):
    rxns = synthetic_reactions(n)
    y = np.random.default_rng(0).random(n)
    full = pd.DataFrame({"rxn": rxns, "y": y})
    raw_dir = os.path.join(data_folder, "raw", "suzuki")
    os.makedirs(raw_dir, exist_ok=True)
    for split_id in split_ids:
        order = np.random.default_rng(100 + split_id).permutation(n)
        full.iloc[order].to_csv(
            os.path.join(raw_dir, "random_split_%d.tsv" % split_id), sep="\t"
        )


class TestSplitDerivation(unittest.TestCase):
    """Split logic on the real Suzuki raw files (no featurization)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.args = make_args(os.path.join(ROOT_DIR, "Data"), cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_ratios_disjointness_and_order(self):
        data = ss.load_raw_split(self.args, 0)
        self.assertEqual(len(data), 5760)
        subsets = ss.derive_split(self.args, data, 0)
        ratios = ss.check_split_integrity(self.args, data, subsets)
        self.assertEqual([len(subsets[s]) for s in ss.SUBSETS], [3628, 404, 1728])
        self.assertAlmostEqual(ratios["train"], 0.63, delta=0.01)
        self.assertAlmostEqual(ratios["valid"], 0.07, delta=0.01)
        self.assertAlmostEqual(ratios["test"], 0.30, delta=0.01)
        # "ordered": the test set is the last 30% of the file, preserving split identity
        self.assertTrue(subsets["test"].index.equals(data.index[-1728:]))
        self.assertEqual(subsets["test"][ss.SOURCE_ROW_COLUMN].min(), 5760 - 1728)

    def test_deterministic_and_splits_differ(self):
        data0 = ss.load_raw_split(self.args, 0)
        first = ss.derive_split(self.args, data0, 0)
        second = ss.derive_split(self.args, ss.load_raw_split(self.args, 0), 0)
        for name in ss.SUBSETS:
            self.assertTrue(first[name].index.equals(second[name].index))
        other = ss.derive_split(self.args, ss.load_raw_split(self.args, 1), 1)
        self.assertFalse(set(first["test"].index) == set(other["test"].index))
        self.assertEqual(ss.split_seed(self.args, 0), 42)
        self.assertEqual(ss.split_seed(self.args, 9), 51)

    def test_integrity_detects_overlap_and_bad_ratios(self):
        data = ss.load_raw_split(self.args, 0)
        subsets = ss.derive_split(self.args, data, 0)
        leaked = dict(subsets)
        leaked["valid"] = pd.concat([subsets["valid"], subsets["test"].iloc[:1]])
        with self.assertRaisesRegex(ValueError, "overlap"):
            ss.check_split_integrity(self.args, data, leaked)
        strict = make_args(self.args.Data_folder, self.tmp.name, test_ratio=0.2)
        with self.assertRaisesRegex(ValueError, "deviates"):
            ss.check_split_integrity(strict, data, subsets)

    def test_missing_file_and_bad_target(self):
        missing = make_args(
            self.args.Data_folder, self.tmp.name, raw_split_dir="raw/nope"
        )
        with self.assertRaises(FileNotFoundError):
            ss.load_raw_split(missing, 0)
        wrong_target = make_args(self.args.Data_folder, self.tmp.name, y_column="yield")
        with self.assertRaisesRegex(ValueError, "not found"):
            ss.load_raw_split(wrong_target, 0)

        with tempfile.TemporaryDirectory() as data_folder:
            write_synthetic_splits(data_folder, [0], n=10)
            path = os.path.join(data_folder, "raw", "suzuki", "random_split_0.tsv")
            df = pd.read_csv(path, sep="\t", index_col=0)
            df.iloc[3, df.columns.get_loc("y")] = np.nan
            df.to_csv(path, sep="\t")
            args = make_args(data_folder, self.tmp.name)
            with self.assertRaisesRegex(ValueError, "missing/invalid"):
                ss.load_raw_split(args, 0)
            args.drop_invalid_rows = True
            self.assertEqual(len(ss.load_raw_split(args, 0)), 9)


class TestFeatureSubsetting(unittest.TestCase):
    def test_subset_equals_direct_featurization(self):
        rxns = synthetic_reactions(12)
        y = np.linspace(0, 1, 12)
        rmol, pmol, reaction = get_graph_data(rxns, 4, 1, None, None, y)
        positions = [1, 6, 7, 11]
        s_rmol, s_pmol, s_reaction = ss.subset_graph_data(
            rmol, pmol, reaction, positions
        )
        d_rmol, d_pmol, d_reaction = get_graph_data(
            [rxns[p] for p in positions], 4, 1, None, None, y[positions]
        )
        for sub, direct in zip(s_rmol + s_pmol, d_rmol + d_pmol):
            for key in ss.MOL_KEYS:
                np.testing.assert_array_equal(
                    np.asarray(sub[key]), np.asarray(direct[key])
                )
                self.assertEqual(
                    np.asarray(sub[key]).dtype, np.asarray(direct[key]).dtype
                )
        np.testing.assert_array_equal(s_reaction["y"], d_reaction["y"])
        self.assertEqual(s_reaction["rsmi"], d_reaction["rsmi"])


class TestPreparationBarrier(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_folder = os.path.join(self.tmp.name, "Data")
        self.log_dir = os.path.join(self.tmp.name, "logs")
        write_synthetic_splits(self.data_folder, [0, 1, 2])
        self.args = make_args(self.data_folder, self.log_dir, split_ids=[0, 1, 2])

    def tearDown(self):
        for handler in ss.logging.getLogger().handlers[:]:
            handler.close()
            ss.logging.getLogger().removeHandler(handler)
        for logger_name in list(ss.logging.Logger.manager.loggerDict):
            if logger_name.startswith("suzuki_regression."):
                for handler in ss.logging.getLogger(logger_name).handlers[:]:
                    handler.close()
                    ss.logging.getLogger(logger_name).removeHandler(handler)
        self.tmp.cleanup()

    def _run(self, stage, **overrides):
        args = make_args(
            self.data_folder, self.log_dir, split_ids=[0, 1, 2], stage=stage
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return ss.run_stage(args)

    def _fake_finetune(self, split_args, save_attention=True):
        path = split_args.Data_folder + split_args.npz_folder + "/test.npz"
        y = np.load(path, allow_pickle=True)["reaction"].item()["y"]
        metrics = {"mae": 0.1, "rmse": 0.2, "r2": 0.5, "pearson": 0.7}
        return {
            "best_epoch": 0,
            "val_loss": 0.01,
            "val_metrics": metrics,
            "test_metrics": metrics,
            "test_labels": list(y),
            "test_preds": list(y + 0.1),
            "n_train": 63,
            "n_valid": 7,
            "n_test": len(y),
            "train_runtime_sec": 1.0,
            "eval_runtime_sec": 0.5,
        }

    def test_prepare_validate_resume_and_barrier(self):
        self.assertEqual(self._run("prepare"), 0)
        for split_id in [0, 1, 2]:
            folder = ss.split_dir(self.args, split_id)
            for name in ("train.npz", "valid.npz", "test.npz", ss.METADATA_FILE):
                self.assertTrue(os.path.isfile(os.path.join(folder, name)))
            self.assertEqual(len(GraphDataset(os.path.join(folder, "test.npz"))), 30)
            self.assertEqual(ss.validate_split(self.args, split_id), [])
        self.assertEqual(self._run("validate"), 0)

        # resume: valid splits are skipped, not rewritten
        test_npz = os.path.join(ss.split_dir(self.args, 1), "test.npz")
        mtime = os.path.getmtime(test_npz)
        with mock.patch.object(ss, "featurize_reactions") as featurize:
            self.assertEqual(self._run("prepare"), 0)
            featurize.assert_not_called()
        self.assertEqual(os.path.getmtime(test_npz), mtime)

        # a corrupted file blocks training for every split
        with open(test_npz, "wb") as f:
            f.write(b"not an npz file")
        self.assertEqual(self._run("validate"), 1)
        with mock.patch.object(ss, "finetune") as finetune:
            self.assertEqual(self._run("train"), 1)
            finetune.assert_not_called()

        # a missing file also blocks training
        os.remove(os.path.join(ss.split_dir(self.args, 2), "valid.npz"))
        with mock.patch.object(ss, "finetune") as finetune:
            self.assertEqual(
                self._run("all", split_ids=[0, 1, 2], raw_split_dir="raw/missing"), 1
            )
            finetune.assert_not_called()

        # re-preparation repairs only the invalid splits, then training runs
        with mock.patch.object(
            ss, "finetune", side_effect=self._fake_finetune
        ) as finetune:
            self.assertEqual(self._run("all"), 0)
            self.assertEqual(finetune.call_count, 3)

        paths = ss.result_paths(self.args)
        results = pd.read_csv(paths["results"])
        self.assertEqual(list(results["split_id"]), [0, 1, 2])
        self.assertTrue((results["status"] == "success").all())
        self.assertEqual(list(results["n_test"]), [30, 30, 30])
        predictions = pd.read_csv(paths["predictions"])
        self.assertEqual(
            list(predictions.columns),
            [
                "split_id",
                "sample_id",
                "original_index",
                "true_yield",
                "predicted_yield",
            ],
        )
        self.assertEqual(len(predictions), 90)
        summary = pd.read_csv(paths["summary"]).set_index("metric")
        self.assertAlmostEqual(summary.loc["test_mae", "mean"], 0.1)
        self.assertEqual(summary.loc["test_mae", "n_successful_runs"], 3)

        # results are not silently overwritten
        self.assertEqual(self._run("train"), 1)

    def test_failed_training_run_is_logged_and_excluded(self):
        self.assertEqual(self._run("prepare"), 0)
        calls = {"n": 0}

        def flaky(split_args, save_attention=True):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return self._fake_finetune(split_args, save_attention)

        with mock.patch.object(ss, "finetune", side_effect=flaky):
            self.assertEqual(self._run("train"), 1)
        paths = ss.result_paths(self.args)
        results = pd.read_csv(paths["results"])
        self.assertEqual(list(results["status"]), ["success", "failed", "success"])
        self.assertIn("boom", results.loc[1, "error"])
        summary = pd.read_csv(paths["summary"]).set_index("metric")
        self.assertEqual(summary.loc["test_rmse", "n_successful_runs"], 2)
        self.assertEqual(summary.loc["test_rmse", "n_failed_runs"], 1)

    def test_skip_aggregate_writes_no_summary(self):
        self.assertEqual(self._run("prepare"), 0)
        with mock.patch.object(ss, "finetune", side_effect=self._fake_finetune):
            self.assertEqual(self._run("train", skip_aggregate=True), 0)
        paths = ss.result_paths(self.args)
        self.assertFalse(os.path.exists(paths["summary"]))
        self.assertEqual(len(pd.read_csv(paths["results"])), 3)
        log_path = os.path.join(self.log_dir, "suzuki_splits_0_to_2.log")
        with open(log_path) as f:
            log = f.read()
        self.assertEqual(log.count("TEST RESULT"), 3)
        self.assertNotIn("SUMMARY", log)


if __name__ == "__main__":
    unittest.main()
