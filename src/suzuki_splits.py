"""
Two-stage reaction-yield pipeline over the Suzuki random splits.

Stage 1 (``prepare``/``validate``) turns every raw split file into
``<Data_folder>/<processed_npz_dir>/split_<id>/{train,valid,test}.npz`` plus
``split_metadata.json`` and checks every file. Stage 2 (``train``) only starts
when all requested splits pass validation (the preparation barrier) and runs the
existing `finetune` on each split's saved npz files.

Split protocol for split ``<id>`` (file ``random_split_<id>.tsv``):

- every raw file holds the full dataset in a different pre-shuffled order; the
  unnamed first column is the original sample id;
- with the default "ordered" strategy the first 70% of rows (file order) form
  train1 and the last 30% form the test set, preserving the split identity;
- train1 is divided 90/10 into train/valid with
  ``sklearn.model_selection.train_test_split``;
- the seed of split ``<id>`` is ``args.seed + <id>`` (42..51 by default).

Features are the repository's molecular graphs from
`reaction_data.get_graph_data`. Featurization is a fixed per-molecule mapping
(no fitted statistics), so reactions are featurized once and each subset is
sliced from the result; the saved npz files are identical to featurizing each
subset directly.
"""

import copy
import gc
import hashlib
import itertools
import json
import logging
import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from data import GraphDataset
from finetune import finetune
from prepare_data import clean_reaction_data, split_reaction_data
from reaction_data import get_graph_data
from utils import read_reaction_table, set_seed

SUBSETS = ("train", "valid", "test")
METADATA_FILE = "split_metadata.json"
METADATA_FORMAT_VERSION = 1
REQUIRED_NPZ_KEYS = ("rmol", "pmol", "reaction", "sample_ids", "source_row_indices")
MOL_KEYS = ("n_node", "n_edge", "dummy", "node_attr", "edge_attr", "src", "dst")
SOURCE_ROW_COLUMN = "_source_row_index"
SUPPORTED_RAW_EXTENSIONS = (".tsv", ".csv", ".gz")
TEST_METRICS = ("mae", "rmse", "r2")


class PreparationBarrierError(RuntimeError):
    """Raised when prepared split data is missing or invalid, blocking training."""


# ----------------------------------------------------------------------------
# configuration helpers
# ----------------------------------------------------------------------------


def split_seed(args, split_id: int) -> int:
    """Deterministic seed of a split: `args.seed + split_id`."""
    return int(args.seed) + int(split_id)


def resolve_split_strategy(args) -> str:
    """The configured split strategy; "ordered" when not given."""
    return args.split_strategy or "ordered"


def expected_ratios(args) -> dict:
    """Effective train/valid/test fractions implied by the requested ratios."""
    return {
        "train": (1 - args.test_ratio) * (1 - args.valid_ratio),
        "valid": (1 - args.test_ratio) * args.valid_ratio,
        "test": args.test_ratio,
    }


def raw_split_path(args, split_id: int) -> str:
    """Path of the raw file of a split."""
    return os.path.join(
        args.Data_folder,
        args.raw_split_dir,
        args.split_file_pattern.format(split_id=split_id),
    )


def split_dir(args, split_id: int) -> str:
    """Folder holding the prepared npz files of a split."""
    return os.path.join(args.Data_folder, args.processed_npz_dir, "split_%d" % split_id)


def experiment_name(split_ids) -> str:
    """Name used for log/result files, e.g. "suzuki_splits_0_to_9"."""
    ids = sorted(split_ids)
    if len(ids) == 1:
        return "suzuki_split_%d" % ids[0]
    if ids == list(range(ids[0], ids[-1] + 1)):
        return "suzuki_splits_%d_to_%d" % (ids[0], ids[-1])
    return "suzuki_splits_" + "_".join(str(i) for i in ids)


def get_experiment_logger(args) -> logging.Logger:
    """
    Returns a dedicated logger writing to the console and (in append mode) to
    `<log_dir>/<experiment_name>.log`. It does not propagate to the root logger,
    so the per-module `setup_logging` calls do not detach it.
    """
    os.makedirs(args.log_dir, exist_ok=True)
    name = experiment_name(args.split_ids)
    logger = logging.getLogger("suzuki_regression." + name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(os.path.join(args.log_dir, name + ".log"), mode="a"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------
# raw data and split derivation
# ----------------------------------------------------------------------------


def load_raw_split(args, split_id: int) -> pd.DataFrame:
    """
    Loads and checks the raw file of a split.

    The returned table is indexed by the original sample id and carries the
    0-based row position in the raw file in `SOURCE_ROW_COLUMN`.

    Raises
    ------
    FileNotFoundError
        If the raw split file does not exist.
    ValueError
        For unsupported/unreadable files, missing columns, non-unique or
        non-integer sample ids, and (unless `args.drop_invalid_rows`) missing or
        invalid reactions/targets.
    """
    path = raw_split_path(args, split_id)
    if not os.path.isfile(path):
        raise FileNotFoundError("Raw file of split %d not found: %s" % (split_id, path))
    if not path.endswith(SUPPORTED_RAW_EXTENSIONS):
        raise ValueError(
            "Unsupported split file format %s (expected one of %s)"
            % (path, SUPPORTED_RAW_EXTENSIONS)
        )
    try:
        data = read_reaction_table(path)
    except Exception as e:
        raise ValueError("Could not parse raw split file %s: %s" % (path, e)) from e

    if not data.index.is_unique:
        raise ValueError("Sample ids (first column) of %s are not unique" % path)
    if not pd.api.types.is_integer_dtype(data.index):
        raise ValueError(
            "Sample ids (first column) of %s must be integers, got %s"
            % (path, data.index.dtype)
        )
    data[SOURCE_ROW_COLUMN] = np.arange(len(data))

    n_before = len(data)
    data, n_excluded = clean_reaction_data(data, args.reaction_column, args.y_column)
    if n_excluded and not args.drop_invalid_rows:
        raise ValueError(
            "%d/%d rows of %s have a missing/invalid reaction or target value in "
            "column %r; fix the file or pass --drop_invalid_rows"
            % (n_excluded, n_before, path, args.y_column)
        )
    if len(data) == 0:
        raise ValueError("No valid samples left in %s" % path)
    data.attrs["n_excluded_rows"] = n_excluded
    return data


def derive_split(args, data: pd.DataFrame, split_id: int) -> dict:
    """Derives the train/valid/test subsets of a split (deterministic)."""
    train, valid, test = split_reaction_data(
        data,
        args.test_ratio,
        args.valid_ratio,
        split_seed(args, split_id),
        resolve_split_strategy(args),
    )
    return {"train": train, "valid": valid, "test": test}


def check_split_integrity(args, data: pd.DataFrame, subsets: dict) -> dict:
    """
    Checks that subsets are non-empty, mutually exclusive, cover every sample
    exactly once, share no reaction SMILES, and have the expected ratios.

    Returns
    -------
    dict
        Effective ratio of every subset.

    Raises
    ------
    ValueError
        Describing the first violated condition.
    """
    ids = {name: subsets[name].index.to_numpy() for name in SUBSETS}
    for name in SUBSETS:
        if len(ids[name]) == 0:
            raise ValueError("Subset %r is empty" % name)
        if len(np.unique(ids[name])) != len(ids[name]):
            raise ValueError("Subset %r contains duplicated sample ids" % name)
    for a, b in itertools.combinations(SUBSETS, 2):
        overlap = np.intersect1d(ids[a], ids[b])
        if len(overlap):
            raise ValueError(
                "%d sample ids overlap between %s and %s (e.g. %s)"
                % (len(overlap), a, b, overlap[:5].tolist())
            )
        shared = set(subsets[a][args.reaction_column]) & set(
            subsets[b][args.reaction_column]
        )
        if shared:
            raise ValueError(
                "%d reactions appear in both %s and %s (leakage)" % (len(shared), a, b)
            )
    all_ids = np.concatenate([ids[name] for name in SUBSETS])
    if len(all_ids) != len(data) or not np.array_equal(
        np.sort(all_ids), np.sort(data.index.to_numpy())
    ):
        raise ValueError("Subsets do not cover every sample exactly once")

    ratios = {name: len(ids[name]) / len(data) for name in SUBSETS}
    for name, expected in expected_ratios(args).items():
        if abs(ratios[name] - expected) > args.ratio_tolerance:
            raise ValueError(
                "Effective %s ratio %.4f deviates from expected %.4f by more than %.4f"
                % (name, ratios[name], expected, args.ratio_tolerance)
            )
    return ratios


# ----------------------------------------------------------------------------
# featurization
# ----------------------------------------------------------------------------


def featurize_reactions(args, data: pd.DataFrame, logger) -> dict:
    """
    Featurizes every reaction of `data` once with `get_graph_data`.

    Reactions are processed in sample-id order; the padding sizes
    (`rmol_max_cnt`, `pmol_max_cnt`) are computed over all reactions exactly as in
    `prepare_data`.
    """
    start = time.time()
    ordered = data.sort_index()
    rsmi_list = ordered[args.reaction_column].values
    rmol_max_cnt = int(np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_list]))
    pmol_max_cnt = int(np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_list]))
    logger.info(
        "Featurizing %d reactions once for all splits (reactant slots %d, product slots %d)"
        % (len(ordered), rmol_max_cnt, pmol_max_cnt)
    )
    rmol, pmol, reaction = get_graph_data(
        rsmi_list, rmol_max_cnt, pmol_max_cnt, args, None, ordered[args.y_column].values
    )
    runtime = time.time() - start
    logger.info("Featurization finished in %.1f s" % runtime)
    return {
        "reference": ordered[[args.reaction_column, args.y_column]],
        "rmol": rmol,
        "pmol": pmol,
        "reaction": reaction,
        "position": pd.Series(np.arange(len(ordered)), index=ordered.index),
        "rmol_max_cnt": rmol_max_cnt,
        "pmol_max_cnt": pmol_max_cnt,
        "runtime_sec": runtime,
    }


def features_match(args, features: dict, data: pd.DataFrame) -> bool:
    """Whether cached features cover exactly the reactions/targets of `data`."""
    if features is None:
        return False
    candidate = data[[args.reaction_column, args.y_column]].sort_index()
    return candidate.equals(features["reference"])


def _subset_mol_dict(mol: dict, positions: np.ndarray) -> dict:
    n_csum = np.concatenate([[0], np.cumsum(mol["n_node"])])
    e_csum = np.concatenate([[0], np.cumsum(mol["n_edge"])])
    node_rows = np.concatenate(
        [np.arange(n_csum[p], n_csum[p + 1]) for p in positions]
    ).astype(int)
    edge_rows = np.concatenate(
        [np.arange(e_csum[p], e_csum[p + 1]) for p in positions]
    ).astype(int)
    return {
        "n_node": mol["n_node"][positions],
        "n_edge": mol["n_edge"][positions],
        "dummy": [mol["dummy"][p] for p in positions],
        "node_attr": mol["node_attr"][node_rows],
        "edge_attr": mol["edge_attr"][edge_rows],
        "src": mol["src"][edge_rows],
        "dst": mol["dst"][edge_rows],
    }


def subset_graph_data(rmol: list, pmol: list, reaction: dict, positions) -> tuple:
    """
    Selects reactions (by position) from `get_graph_data` output, returning
    (rmol, pmol, reaction) in the same format as featurizing them directly.
    """
    positions = np.asarray(positions, dtype=int)
    return (
        [_subset_mol_dict(m, positions) for m in rmol],
        [_subset_mol_dict(m, positions) for m in pmol],
        {
            "y": np.asarray(reaction["y"])[positions],
            "rsmi": [reaction["rsmi"][p] for p in positions],
        },
    )


# ----------------------------------------------------------------------------
# stage 1: preparation
# ----------------------------------------------------------------------------


def _array_summary(rmol, pmol, reaction, sample_ids) -> dict:
    def mol_summary(mols):
        return {
            "n_slots": len(mols),
            "node_attr": {
                "dtype": str(mols[0]["node_attr"].dtype),
                "n_rows": int(sum(m["node_attr"].shape[0] for m in mols)),
                "feature_dim": int(mols[0]["node_attr"].shape[1]),
            },
            "edge_attr": {
                "dtype": str(mols[0]["edge_attr"].dtype),
                "n_rows": int(sum(m["edge_attr"].shape[0] for m in mols)),
                "feature_dim": int(mols[0]["edge_attr"].shape[1]),
            },
            "n_node": {
                "dtype": str(mols[0]["n_node"].dtype),
                "shape": [len(sample_ids)],
            },
            "src_dst_dtype": str(mols[0]["src"].dtype),
        }

    y = np.asarray(reaction["y"])
    return {
        "rmol": mol_summary(rmol),
        "pmol": mol_summary(pmol),
        "y": {"dtype": str(y.dtype), "shape": list(y.shape)},
        "sample_ids": {"dtype": str(sample_ids.dtype), "shape": list(sample_ids.shape)},
    }


def _save_npz_atomically(path: str, **arrays) -> None:
    tmp_path = path[: -len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def write_split(args, split_id, data, subsets, ratios, features, prep_start) -> dict:
    """Writes the npz files and metadata of one split; returns the metadata."""
    out_dir = split_dir(args, split_id)
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = os.path.join(out_dir, METADATA_FILE)
    if os.path.exists(metadata_path):
        # The metadata marks a complete split; remove it first so an interrupted
        # rewrite is never mistaken for a complete one.
        os.remove(metadata_path)

    files = {}
    for name in SUBSETS:
        subset = subsets[name]
        sample_ids = subset.index.to_numpy().astype(np.int64)
        rmol, pmol, reaction = subset_graph_data(
            features["rmol"],
            features["pmol"],
            features["reaction"],
            features["position"].loc[sample_ids].to_numpy(),
        )
        path = os.path.join(out_dir, name + ".npz")
        _save_npz_atomically(
            path,
            rmol=rmol,
            pmol=pmol,
            reaction=reaction,
            sample_ids=sample_ids,
            source_row_indices=subset[SOURCE_ROW_COLUMN].to_numpy().astype(np.int64),
        )
        files[name] = {
            "path": path,
            "sha256": _sha256(path),
            "n_samples": int(len(subset)),
            "arrays": _array_summary(rmol, pmol, reaction, sample_ids),
        }

    source = raw_split_path(args, split_id)
    metadata = {
        "format_version": METADATA_FORMAT_VERSION,
        "split_id": int(split_id),
        "seed": split_seed(args, split_id),
        "seed_rule": "base_seed + split_id",
        "base_seed": int(args.seed),
        "source_file": source,
        "source_file_sha256": _sha256(source),
        "sample_id_column": "first (index) column of the source file",
        "source_row_index": "0-based row position in the source file",
        "reaction_column": args.reaction_column,
        "target_column": args.y_column,
        "feature_representation": {
            "type": "molecular graphs of reactants and products",
            "featurizer": "reaction_data.get_graph_data",
            "npz_keys": list(REQUIRED_NPZ_KEYS),
            "mol_dict_keys": list(MOL_KEYS),
            "rmol_max_cnt": features["rmol_max_cnt"],
            "pmol_max_cnt": features["pmol_max_cnt"],
        },
        "split_strategy": resolve_split_strategy(args),
        "requested_ratios": {
            "test_of_full": args.test_ratio,
            "valid_of_train1": args.valid_ratio,
        },
        "expected_effective_ratios": expected_ratios(args),
        "effective_ratios": ratios,
        "ratio_tolerance": args.ratio_tolerance,
        "n_total": int(len(data)),
        "n_excluded_rows": int(data.attrs.get("n_excluded_rows", 0)),
        "n_samples": {name: files[name]["n_samples"] for name in SUBSETS},
        "files": files,
        "preprocessing": {
            "learned_transformations": [],
            "note": "Graph featurization is a fixed per-molecule mapping; no statistics "
            "are fitted and targets are used untransformed.",
        },
        "prepared_at": _now(),
        "shared_featurization_runtime_sec": features["runtime_sec"],
        "preparation_runtime_sec": time.time() - prep_start,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def prepare_split(args, split_id, features, logger) -> dict:
    """
    Prepares one split: loads raw data, derives and checks the subsets (twice,
    to confirm determinism), and writes the npz files and metadata.

    Returns
    -------
    dict
        The (possibly newly computed) shared features.
    """
    prep_start = time.time()
    data = load_raw_split(args, split_id)
    subsets = derive_split(args, data, split_id)
    ratios = check_split_integrity(args, data, subsets)
    repeated = derive_split(args, data, split_id)
    for name in SUBSETS:
        if not subsets[name].index.equals(repeated[name].index):
            raise RuntimeError(
                "Split %d is not deterministic (%s differs)" % (split_id, name)
            )

    if not features_match(args, features, data):
        features = featurize_reactions(args, data, logger)
    write_split(args, split_id, data, subsets, ratios, features, prep_start)
    logger.info(
        "Split %d prepared: train/valid/test %d/%d/%d (%.4f/%.4f/%.4f), seed %d -> %s"
        % (
            split_id,
            len(subsets["train"]),
            len(subsets["valid"]),
            len(subsets["test"]),
            ratios["train"],
            ratios["valid"],
            ratios["test"],
            split_seed(args, split_id),
            split_dir(args, split_id),
        )
    )
    return features


def prepare_all_splits(args, logger) -> dict:
    """
    Stage 1a: prepares every requested split sequentially.

    Valid existing splits are skipped unless `args.overwrite` is set; missing or
    invalid ones are (re)generated. A failing split is logged and the remaining
    splits are still prepared so that all failures are reported at once.

    Returns
    -------
    dict
        Mapping of failed split id to error message (empty on success).
    """
    logger.info("=" * 80)
    logger.info(
        "STAGE 1: PREPARE splits %s -> %s" % (args.split_ids, args.processed_npz_dir)
    )
    features = None
    failures = {}
    for split_id in args.split_ids:
        try:
            if not args.overwrite:
                errors = validate_split(args, split_id)
                if not errors:
                    logger.info(
                        "Split %d: valid prepared data found, skipping" % split_id
                    )
                    continue
                if os.path.isdir(split_dir(args, split_id)):
                    logger.warning(
                        "Split %d: existing prepared data is incomplete or invalid and will "
                        "be regenerated: %s" % (split_id, "; ".join(errors))
                    )
            else:
                logger.warning("Split %d: --overwrite given, regenerating" % split_id)
            features = prepare_split(args, split_id, features, logger)
        except Exception as e:
            failures[split_id] = "%s: %s" % (type(e).__name__, e)
            logger.error(
                "Split %d: preparation FAILED: %s" % (split_id, failures[split_id])
            )
            logger.debug(traceback.format_exc())
    if failures:
        logger.error("Preparation failed for splits %s" % sorted(failures))
    return failures


# ----------------------------------------------------------------------------
# stage 1: validation
# ----------------------------------------------------------------------------


def _check_mol_dicts(mols, n: int, label: str, errors: list) -> None:
    if len(mols) == 0:
        errors.append("%s has no molecule slots" % label)
    for j, mol in enumerate(mols):
        tag = "%s[%d]" % (label, j)
        missing = [k for k in MOL_KEYS if k not in mol]
        if missing:
            errors.append("%s is missing keys %s" % (tag, missing))
            continue
        n_node, n_edge = np.asarray(mol["n_node"]), np.asarray(mol["n_edge"])
        if len(n_node) != n or len(n_edge) != n or len(mol["dummy"]) != n:
            errors.append("%s per-sample arrays do not have length %d" % (tag, n))
            continue
        if not np.issubdtype(n_node.dtype, np.integer) or not np.issubdtype(
            n_edge.dtype, np.integer
        ):
            errors.append("%s n_node/n_edge must be integers" % tag)
        if mol["node_attr"].ndim != 2 or mol["node_attr"].shape[0] != n_node.sum():
            errors.append(
                "%s node_attr shape %s does not match n_node"
                % (tag, mol["node_attr"].shape)
            )
        if mol["edge_attr"].ndim != 2 or mol["edge_attr"].shape[0] != n_edge.sum():
            errors.append(
                "%s edge_attr shape %s does not match n_edge"
                % (tag, mol["edge_attr"].shape)
            )
        if len(mol["src"]) != n_edge.sum() or len(mol["dst"]) != n_edge.sum():
            errors.append("%s src/dst length does not match n_edge" % tag)
        if mol["node_attr"].dtype != bool or mol["edge_attr"].dtype != bool:
            errors.append("%s node_attr/edge_attr must be boolean" % tag)
    if (
        len(mols)
        and len({m["node_attr"].shape[1] for m in mols if "node_attr" in m}) > 1
    ):
        errors.append("%s slots have inconsistent node feature dims" % label)


def _check_sample_arrays(label, sample_ids, source_rows, reaction, errors) -> None:
    n = len(sample_ids)
    y = np.asarray(reaction.get("y"))
    if not np.issubdtype(sample_ids.dtype, np.integer) or sample_ids.ndim != 1:
        errors.append("%s sample_ids must be a 1-D integer array" % label)
    if len(np.unique(sample_ids)) != n:
        errors.append("%s has duplicated sample_ids" % label)
    if source_rows.shape != sample_ids.shape:
        errors.append("%s source_row_indices shape does not match sample_ids" % label)
    if y.shape != (n,) or len(reaction.get("rsmi", [])) != n:
        errors.append(
            "%s target/reaction length does not match %d samples" % (label, n)
        )
    elif not np.issubdtype(y.dtype, np.floating) or not np.isfinite(y).all():
        errors.append("%s targets must be finite floats" % label)


def check_npz_file(path: str, errors: list):
    """
    Loads and structurally checks one prepared npz file.

    Returns
    -------
    dict or None
        {"sample_ids", "source_row_indices", "y", "node_dim", "edge_dim",
        "rmol_max_cnt", "pmol_max_cnt"} when the file is readable, else None.
    """
    label = os.path.basename(path)
    try:
        with np.load(path, allow_pickle=True) as npz:
            missing = [k for k in REQUIRED_NPZ_KEYS if k not in npz.files]
            if missing:
                errors.append("%s is missing keys %s" % (label, missing))
                return None
            rmol, pmol = list(npz["rmol"]), list(npz["pmol"])
            reaction = npz["reaction"].item()
            sample_ids = npz["sample_ids"]
            source_rows = npz["source_row_indices"]
    except Exception as e:
        errors.append(
            "%s is corrupted or unreadable: %s: %s" % (label, type(e).__name__, e)
        )
        return None

    y = np.asarray(reaction.get("y"))
    n = len(sample_ids)
    if n == 0:
        errors.append("%s contains no samples" % label)
        return None
    _check_sample_arrays(label, sample_ids, source_rows, reaction, errors)
    _check_mol_dicts(rmol, n, label + ":rmol", errors)
    _check_mol_dicts(pmol, n, label + ":pmol", errors)
    if errors:
        return None

    try:
        dataset = GraphDataset(path)
        if len(dataset) != n:
            raise ValueError("GraphDataset length %d != %d" % (len(dataset), n))
        dataset[0]
        dataset[n - 1]
    except Exception as e:
        errors.append(
            "%s cannot be loaded by GraphDataset: %s: %s" % (label, type(e).__name__, e)
        )
        return None
    return {
        "sample_ids": sample_ids,
        "source_row_indices": source_rows,
        "y": y,
        "node_dim": rmol[0]["node_attr"].shape[1],
        "edge_dim": rmol[0]["edge_attr"].shape[1],
        "rmol_max_cnt": len(rmol),
        "pmol_max_cnt": len(pmol),
    }


def _check_metadata(args, split_id, metadata, errors) -> None:
    expected = {
        "format_version": METADATA_FORMAT_VERSION,
        "split_id": split_id,
        "seed": split_seed(args, split_id),
        "reaction_column": args.reaction_column,
        "target_column": args.y_column,
        "split_strategy": resolve_split_strategy(args),
        "requested_ratios": {
            "test_of_full": args.test_ratio,
            "valid_of_train1": args.valid_ratio,
        },
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(
                "metadata %s=%r does not match the current configuration (%r)"
                % (key, metadata.get(key), value)
            )
    source = raw_split_path(args, split_id)
    if os.path.isfile(source) and metadata.get("source_file_sha256") != _sha256(source):
        errors.append("raw file %s changed since preparation" % source)


def _check_against_raw(args, split_id, loaded, errors) -> None:
    """Re-derives the split from the raw file and compares it with the npz files."""
    data = load_raw_split(args, split_id)
    subsets = derive_split(args, data, split_id)
    check_split_integrity(args, data, subsets)
    for name in SUBSETS:
        subset = subsets[name]
        if not np.array_equal(loaded[name]["sample_ids"], subset.index.to_numpy()):
            errors.append(
                "%s.npz sample ids differ from the deterministic split" % name
            )
            continue
        if not np.array_equal(
            loaded[name]["source_row_indices"], subset[SOURCE_ROW_COLUMN].to_numpy()
        ):
            errors.append("%s.npz source row indices differ from the raw file" % name)
        if not np.allclose(loaded[name]["y"], subset[args.y_column].to_numpy()):
            errors.append("%s.npz targets differ from the raw file" % name)


def validate_split(args, split_id: int) -> list:
    """
    Validates the prepared files of one split.

    Checks file presence, metadata consistency with the current configuration,
    npz readability/keys/shapes/dtypes, file checksums, GraphDataset loading,
    subset disjointness and coverage, effective ratios, and that the saved
    subsets equal a fresh deterministic re-derivation from the raw file.

    Returns
    -------
    list
        Error messages; empty when the split is valid.
    """
    out_dir = split_dir(args, split_id)
    required = [name + ".npz" for name in SUBSETS] + [METADATA_FILE]
    missing = [f for f in required if not os.path.isfile(os.path.join(out_dir, f))]
    if missing:
        return ["missing files in %s: %s" % (out_dir, missing)]

    errors = []
    try:
        with open(os.path.join(out_dir, METADATA_FILE)) as f:
            metadata = json.load(f)
    except Exception as e:
        return ["%s is unreadable: %s" % (METADATA_FILE, e)]
    _check_metadata(args, split_id, metadata, errors)

    loaded = {}
    for name in SUBSETS:
        path = os.path.join(out_dir, name + ".npz")
        recorded = metadata.get("files", {}).get(name, {})
        if recorded.get("sha256") != _sha256(path):
            errors.append(
                "%s.npz checksum does not match metadata (modified or corrupted)" % name
            )
        file_errors = []
        loaded[name] = check_npz_file(path, file_errors)
        errors.extend(file_errors)
        if loaded[name] is not None and recorded.get("n_samples") != len(
            loaded[name]["sample_ids"]
        ):
            errors.append("%s.npz sample count does not match metadata" % name)
    if errors:
        return errors

    dims = {
        (v["node_dim"], v["edge_dim"], v["rmol_max_cnt"], v["pmol_max_cnt"])
        for v in loaded.values()
    }
    if len(dims) != 1:
        errors.append(
            "train/valid/test have incompatible feature dims or molecule slots: %s"
            % dims
        )
    try:
        _check_against_raw(args, split_id, loaded, errors)
    except Exception as e:
        errors.append(
            "integrity check against raw data failed: %s: %s" % (type(e).__name__, e)
        )
    return errors


def validate_all_splits(args, logger) -> dict:
    """
    Stage 1b: validates every requested split (the preparation barrier).

    Returns
    -------
    dict
        Mapping of invalid split id to its error messages (empty when all pass).
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: VALIDATE splits %s" % args.split_ids)
    failures = {}
    n_npz = 0
    for split_id in args.split_ids:
        n_npz += sum(
            os.path.isfile(os.path.join(split_dir(args, split_id), name + ".npz"))
            for name in SUBSETS
        )
        errors = validate_split(args, split_id)
        if errors:
            failures[split_id] = errors
            logger.error("Split %d: INVALID: %s" % (split_id, "; ".join(errors)))
        else:
            logger.info("Split %d: valid (%s)" % (split_id, split_dir(args, split_id)))
    expected_npz = len(SUBSETS) * len(args.split_ids)
    logger.info("Found %d/%d subset npz files" % (n_npz, expected_npz))
    if failures:
        logger.error(
            "Preparation barrier NOT passed: %d/%d splits invalid (%s)"
            % (len(failures), len(args.split_ids), sorted(failures))
        )
    else:
        logger.info(
            "Preparation barrier passed: all %d splits and %d npz files are valid"
            % (len(args.split_ids), expected_npz)
        )
    return failures


# ----------------------------------------------------------------------------
# stage 2: training
# ----------------------------------------------------------------------------


def result_paths(args) -> dict:
    """Output paths of the experiment-level result files."""
    name = experiment_name(args.split_ids)
    return {
        "results": os.path.join(args.log_dir, name + "_results.csv"),
        "summary": os.path.join(args.log_dir, name + "_summary.csv"),
        "predictions": os.path.join(args.log_dir, name + "_test_predictions.csv"),
    }


def _split_args(args, split_id: int, metadata: dict, run_dir: str):
    split_args = copy.copy(args)
    split_args.seed = metadata["seed"]
    # finetune builds npz paths by concatenation and expects a trailing separator
    split_args.Data_folder = os.path.join(args.Data_folder, "")
    split_args.npz_folder = os.path.relpath(split_dir(args, split_id), args.Data_folder)
    out = os.path.join(run_dir, "split_%d" % split_id)
    split_args.monitor_folder = os.path.join(out, "monitor") + os.sep
    split_args.image_folder = os.path.join(out, "images") + os.sep
    split_args.model_path = out + os.sep
    split_args.model_name = "model.pt"
    os.makedirs(split_args.monitor_folder, exist_ok=True)
    os.makedirs(split_args.image_folder, exist_ok=True)
    return split_args


def _fmt(value) -> str:
    return "n/a" if value is None or pd.isna(value) else "%.4f" % value


def train_split(args, split_id: int, run_dir: str, logger):
    """
    Trains and evaluates one split from its prepared npz files.

    Returns
    -------
    tuple
        (result row dict, predictions DataFrame or None)
    """
    out_dir = split_dir(args, split_id)
    with open(os.path.join(out_dir, METADATA_FILE)) as f:
        metadata = json.load(f)
    row = {
        "split_id": split_id,
        "seed": metadata["seed"],
        "status": "failed",
        "error": "",
        "train_npz": os.path.join(out_dir, "train.npz"),
        "valid_npz": os.path.join(out_dir, "valid.npz"),
        "test_npz": os.path.join(out_dir, "test.npz"),
        "n_train": metadata["n_samples"]["train"],
        "n_valid": metadata["n_samples"]["valid"],
        "n_test": metadata["n_samples"]["test"],
        "ratio_train": metadata["effective_ratios"]["train"],
        "ratio_valid": metadata["effective_ratios"]["valid"],
        "ratio_test": metadata["effective_ratios"]["test"],
        "preparation_runtime_sec": metadata["preparation_runtime_sec"],
        "started_at": _now(),
    }
    logger.info("-" * 80)
    logger.info(
        "Split %d: training (seed %d, train/valid/test %d/%d/%d)"
        % (split_id, row["seed"], row["n_train"], row["n_valid"], row["n_test"])
    )
    predictions = None
    try:
        split_args = _split_args(args, split_id, metadata, run_dir)
        row["model_path"] = split_args.model_path + split_args.model_name
        set_seed(split_args.seed)
        result = finetune(split_args, save_attention=False)

        with np.load(row["test_npz"], allow_pickle=True) as npz:
            sample_ids = npz["sample_ids"]
            source_rows = npz["source_row_indices"]
            y_test = npz["reaction"].item()["y"]
        preds = np.asarray(result["test_preds"], dtype=float)
        labels = np.asarray(result["test_labels"], dtype=float)
        if len(preds) != len(sample_ids) or not np.allclose(labels, y_test, atol=1e-5):
            raise RuntimeError(
                "Test predictions (%d) are not aligned with test.npz samples (%d)"
                % (len(preds), len(sample_ids))
            )

        val_metrics, test_metrics = result["val_metrics"], result["test_metrics"]
        row.update(
            {
                "status": "success",
                "best_epoch": result["best_epoch"],
                "val_loss": result["val_loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_pearson": val_metrics["pearson"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_pearson": test_metrics["pearson"],
                "training_runtime_sec": result["train_runtime_sec"],
                "evaluation_runtime_sec": result["eval_runtime_sec"],
            }
        )
        predictions = pd.DataFrame(
            {
                "split_id": split_id,
                "sample_id": sample_ids,
                "original_index": source_rows,
                "true_yield": y_test,
                "predicted_yield": preds,
            }
        )
        logger.info(
            "Split %d: best epoch %d | val MAE %s RMSE %s R2 %s | test MAE %s RMSE %s R2 %s"
            % (
                split_id,
                row["best_epoch"],
                _fmt(row["val_mae"]),
                _fmt(row["val_rmse"]),
                _fmt(row["val_r2"]),
                _fmt(row["test_mae"]),
                _fmt(row["test_rmse"]),
                _fmt(row["test_r2"]),
            )
        )
        logger.info(
            "Split %d: TEST RESULT seed=%d n_train=%d n_valid=%d n_test=%d test_mae=%s "
            "test_rmse=%s test_r2=%s training_runtime_sec=%.1f evaluation_runtime_sec=%.1f"
            % (
                split_id,
                row["seed"],
                row["n_train"],
                row["n_valid"],
                row["n_test"],
                _fmt(row["test_mae"]),
                _fmt(row["test_rmse"]),
                _fmt(row["test_r2"]),
                row["training_runtime_sec"],
                row["evaluation_runtime_sec"],
            )
        )
    except Exception as e:
        row["error"] = "%s: %s" % (type(e).__name__, e)
        logger.error(
            "Split %d: training/evaluation FAILED: %s" % (split_id, row["error"])
        )
        logger.error(traceback.format_exc())
    row["finished_at"] = _now()
    # release the previous split's model/datasets before the next split starts
    gc.collect()
    return row, predictions


def summarize_results(rows: list) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) of test metrics over successful runs."""
    results = pd.DataFrame(rows)
    success = results[results["status"] == "success"] if len(results) else results
    summary = []
    for metric in TEST_METRICS:
        column = "test_" + metric
        values = (
            pd.to_numeric(success[column], errors="coerce").dropna()
            if column in success
            else []
        )
        summary.append(
            {
                "metric": column,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "n_successful_runs": int(len(success)),
                "n_failed_runs": int(len(results) - len(success)),
            }
        )
    return pd.DataFrame(summary)


def check_results_absent(args) -> None:
    """Raises FileExistsError if result files exist and overwriting was not requested."""
    existing = [p for p in result_paths(args).values() if os.path.exists(p)]
    if existing and not args.overwrite_results:
        raise FileExistsError(
            "Result files already exist: %s. Move them away or pass --overwrite_results"
            % existing
        )


def train_all_splits(args, logger) -> int:
    """
    Stage 2: after the preparation barrier passes, trains and evaluates every
    requested split from its npz files. A failed run is logged and skipped.

    Returns
    -------
    int
        0 when every run succeeded, 1 otherwise.

    Raises
    ------
    FileExistsError
        If result files exist and `args.overwrite_results` is not set.
    PreparationBarrierError
        If any requested split is missing or invalid.
    """
    paths = result_paths(args)
    check_results_absent(args)
    failures = validate_all_splits(args, logger)
    if failures:
        raise PreparationBarrierError(
            "Training not started: prepared data is missing or invalid for splits %s"
            % sorted(failures)
        )

    name = experiment_name(args.split_ids)
    run_dir = os.path.join(
        args.log_dir, "runs", "%s_%s" % (name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    logger.info("=" * 80)
    logger.info("STAGE 2: TRAIN splits %s (run folder %s)" % (args.split_ids, run_dir))

    rows, predictions = [], []
    for split_id in args.split_ids:
        row, split_predictions = train_split(args, split_id, run_dir, logger)
        rows.append(row)
        if split_predictions is not None:
            predictions.append(split_predictions)
        # rewritten after every split so partial results survive an interruption
        pd.DataFrame(rows).to_csv(paths["results"], index=False)
        if predictions:
            pd.concat(predictions).to_csv(paths["predictions"], index=False)

    logger.info("=" * 80)
    failed = [r["split_id"] for r in rows if r["status"] != "success"]
    if args.skip_aggregate:
        logger.info("Aggregate metrics skipped (--skip_aggregate)")
        del paths["summary"]
    else:
        summary = summarize_results(rows)
        summary.to_csv(paths["summary"], index=False)
        logger.info("SUMMARY over splits %s" % args.split_ids)
        for record in summary.to_dict("records"):
            logger.info(
                "%s: mean %s, std %s"
                % (record["metric"], _fmt(record["mean"]), _fmt(record["std"]))
            )
    logger.info(
        "successful runs: %d, failed runs: %d" % (len(rows) - len(failed), len(failed))
    )
    if failed:
        logger.error("Failed splits: %s" % failed)
    logger.info("Results: %s" % paths)
    return 0 if not failed else 1


def run_stage(args) -> int:
    """
    Runs a pipeline stage and returns a process exit code (0 on success).

    - "prepare": prepare all requested splits, then validate them.
    - "validate": validate all requested splits.
    - "train": check the preparation barrier, then train every split.
    - "all": prepare and validate all splits; only if every split is valid, train.
    """
    logger = get_experiment_logger(args)
    logger.info("Command stage=%s, split_ids=%s" % (args.stage, args.split_ids))
    try:
        if args.stage in ("train", "all"):
            # fail fast, before any expensive preparation
            check_results_absent(args)
        if args.stage in ("prepare", "all"):
            failures = prepare_all_splits(args, logger)
            if failures:
                logger.error(
                    "Preparation FAILED for splits %s; no training started"
                    % sorted(failures)
                )
                return 1
        if args.stage in ("prepare", "validate") and validate_all_splits(args, logger):
            return 1
        if args.stage in ("train", "all"):
            # train_all_splits validates every split (the barrier) before training
            return train_all_splits(args, logger)
        return 0
    except (PreparationBarrierError, FileExistsError) as e:
        logger.error(str(e))
        return 1
