
"""
Linear probing on multiple dataset roots for all model variants used in
linear_probe_cross_length.py, evaluated at fixed layer 6 and last layer.

No layer search is performed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from transformers import AutoTokenizer


def _load_cross_length_module(thesis_root: Path):
    module_path = thesis_root / "linear_probe_cross_length" / "linear_probe_cross_length.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing source module: {module_path}")
    spec = importlib.util.spec_from_file_location("lp_cross_length", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(thesis_root: Path, all_models: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-dataset linear probing for all DNA-LM variants (fixed layer 6 + last)"
    )
    parser.add_argument(
        "--data_roots",
        nargs="+",
        default=[
            str(thesis_root / "data" / "finetune_data_koo"),
            str(thesis_root / "DNABERT2" / "data"),
            str(thesis_root / "data" / "diff_cells_data" / "splits_csv"),
        ],
        help="List of roots with per-task subdirectories containing train/dev/test CSV",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=all_models,
        choices=all_models,
        help="Model subset to evaluate",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["6", "last"],
        choices=["6", "last"],
        help="Transformer layers to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples_per_task", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(thesis_root / "linear probe" / "results"),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(thesis_root / "linear probe" / "results" / "cache"),
    )
    return parser.parse_args()


def _find_column(columns: List[str], candidates: List[str]) -> str:
    lmap = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lmap:
            return lmap[cand]
    raise ValueError(f"None of {candidates} found in columns: {columns}")


def _load_single_task(task_dir: Path) -> Optional[Dict[str, pd.DataFrame]]:
    train_fp = task_dir / "train.csv"
    dev_fp = task_dir / "dev.csv"
    test_fp = task_dir / "test.csv"
    if not train_fp.exists() or not dev_fp.exists() or not test_fp.exists():
        return None

    train_df = pd.read_csv(train_fp)
    dev_df = pd.read_csv(dev_fp)
    test_df = pd.read_csv(test_fp)

    seq_col_tr = _find_column(list(train_df.columns), ["sequence", "seq", "text", "input", "x"])
    lbl_col_tr = _find_column(list(train_df.columns), ["label", "labels", "target", "y"])
    seq_col_dev = _find_column(list(dev_df.columns), ["sequence", "seq", "text", "input", "x"])
    lbl_col_dev = _find_column(list(dev_df.columns), ["label", "labels", "target", "y"])
    seq_col_te = _find_column(list(test_df.columns), ["sequence", "seq", "text", "input", "x"])
    lbl_col_te = _find_column(list(test_df.columns), ["label", "labels", "target", "y"])

    tr = pd.DataFrame(
        {
            "sequence": train_df[seq_col_tr].astype(str),
            "label": train_df[lbl_col_tr].astype(int),
        }
    ).dropna()
    dv = pd.DataFrame(
        {
            "sequence": dev_df[seq_col_dev].astype(str),
            "label": dev_df[lbl_col_dev].astype(int),
        }
    ).dropna()
    te = pd.DataFrame(
        {
            "sequence": test_df[seq_col_te].astype(str),
            "label": test_df[lbl_col_te].astype(int),
        }
    ).dropna()

    for df in (tr, dv, te):
        df["sequence"] = df["sequence"].str.replace("U", "T").str.replace("u", "t")

    train_dev = (
        pd.concat([tr, dv], ignore_index=True)
        .drop_duplicates(subset=["sequence"]) 
        .reset_index(drop=True)
    )
    test = te.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    if train_dev["label"].nunique() < 2 or test["label"].nunique() < 2:
        return None

    return {"train_dev": train_dev, "test": test}


def load_tasks(data_roots: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    tasks: Dict[str, Dict[str, pd.DataFrame]] = {}
    for root_str in data_roots:
        root = Path(root_str)
        if not root.exists():
            print(f"[WARN] data root missing: {root}")
            continue
        source_tag = root.name
        for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            task = _load_single_task(task_dir)
            if task is None:
                continue
            task_id = f"{source_tag}/{task_dir.name}"
            tasks[task_id] = task
    print(f"[data] Loaded {len(tasks)} valid tasks from {len(data_roots)} roots")
    return tasks


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layer_name, model_name), grp in df.groupby(["layer_name", "model"]):
        rows.append(
            {
                "Layer": layer_name,
                "Model": model_name,
                "Mean AUROC": grp["auroc"].mean(),
                "Std AUROC": grp["auroc"].std(ddof=0),
                "Mean F1": grp["f1"].mean(),
                "Std F1": grp["f1"].std(ddof=0),
                "Mean AUPRC": grp["auprc"].mean(),
                "Std AUPRC": grp["auprc"].std(ddof=0),
                "N tasks": len(grp),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["Layer", "Mean AUROC"], ascending=[True, False]).reset_index(drop=True)


def statistical_tests(per_rbp_df: pd.DataFrame) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for layer_name, layer_df in per_rbp_df.groupby("layer_name"):
        pivot = layer_df.pivot(index="task_id", columns="model", values="auroc")
        all_cols = list(pivot.columns)
        if len(all_cols) < 2:
            continue
        ref_candidates = ["lamar_pretrained", "dnabert2_pretrained", "one_hot", all_cols[0]]
        ref = next((x for x in ref_candidates if x in all_cols), all_cols[0])
        layer_res: Dict[str, dict] = {}
        for other in all_cols:
            if other == ref:
                continue
            paired = pivot[[ref, other]].dropna()
            if len(paired) < 5:
                continue
            try:
                w = wilcoxon(paired[other], paired[ref], alternative="two-sided")
            except Exception:
                continue
            key = f"wilcoxon_{other}_vs_{ref}"
            layer_res[key] = {
                "statistic": float(w.statistic),
                "pvalue": float(w.pvalue),
                "n_tasks": int(len(paired)),
                "mean_delta_auroc": float((paired[other] - paired[ref]).mean()),
                "reference_model": ref,
            }
        out[layer_name] = layer_res
    return out


def main() -> None:
    thesis_root = Path(__file__).resolve().parents[1]
    mod = _load_cross_length_module(thesis_root)

    all_models = list(mod._MODEL_DEFAULTS.keys()) + ["one_hot"]
    args = parse_args(thesis_root, all_models)

    mod.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if torch.cuda.is_available():
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.data_roots)
    if not tasks:
        raise RuntimeError("No valid tasks discovered.")

    # Keep consistent integer task ids for optional future projections.
    task_to_int = {name: i for i, name in enumerate(sorted(tasks.keys()))}
    _ = task_to_int

    requested = args.models
    active_models: Dict[str, dict] = {}
    for name in requested:
        if name == "one_hot":
            continue
        spec = mod._MODEL_DEFAULTS.get(name)
        if spec is None:
            continue
        wpath = spec["weights_path"]
        if wpath != "__random__":
            is_local = wpath.startswith("/") or wpath.startswith(".")
            if is_local and not Path(wpath).exists():
                print(f"[WARN] missing weights for {name}: {wpath}")
                continue
        active_models[name] = spec

    use_one_hot = "one_hot" in requested
    if not active_models and not use_one_hot:
        raise RuntimeError("No valid models selected.")

    tokenizer_cache: Dict[str, object] = {}

    def _get_tokenizer(spec: dict):
        tok_path = spec["tokenizer"]
        if tok_path not in tokenizer_cache:
            extra = {}
            if spec["type"] == "dnabert2":
                extra = {"use_fast": True, "trust_remote_code": True, "padding_side": "right"}
            elif spec["type"] == "dnabert2_singlenuc":
                extra = {"use_fast": True}
            tokenizer_cache[tok_path] = AutoTokenizer.from_pretrained(
                tok_path,
                model_max_length=spec["max_length"],
                **extra,
            )
        return tokenizer_cache[tok_path]

    layer_plan: List[Tuple[str, Optional[int]]] = []
    for layer_flag in args.layers:
        if layer_flag == "6":
            layer_plan.append(("layer6", 6))
        elif layer_flag == "last":
            layer_plan.append(("last", None))

    per_rbp_records: List[dict] = []

    # One-hot baseline (independent from transformer layer choices).
    if use_one_hot:
        print("\n" + "=" * 72)
        print("Model: one_hot")
        print("=" * 72)
        max_len_onehot = 512
        for task_id, splits in sorted(tasks.items()):
            df_train = splits["train_dev"]
            df_test = splits["test"]
            if args.max_samples_per_task > 0 and len(df_train) > args.max_samples_per_task:
                df_train = df_train.sample(n=args.max_samples_per_task, random_state=args.seed)
            if args.max_samples_per_task > 0 and len(df_test) > args.max_samples_per_task:
                df_test = df_test.sample(n=args.max_samples_per_task, random_state=args.seed)

            seq_train = df_train["sequence"].tolist()
            y_train = df_train["label"].to_numpy(dtype=np.int64)
            seq_test = df_test["sequence"].tolist()
            y_test = df_test["label"].to_numpy(dtype=np.int64)

            X_train = mod.get_or_build_emb(
                cache_dir,
                "one_hot",
                mod._cache_layer_index(None),
                mod._cache_task_name(task_id, "train_dev"),
                len(y_train),
                lambda seqs=seq_train: mod.one_hot_embed(seqs, max_len_onehot),
            )
            X_test = mod.get_or_build_emb(
                cache_dir,
                "one_hot",
                mod._cache_layer_index(None),
                mod._cache_task_name(task_id, "test"),
                len(y_test),
                lambda seqs=seq_test: mod.one_hot_embed(seqs, max_len_onehot),
            )
            m = mod.evaluate_linear_probe(X_train, y_train, X_test, y_test, seed=args.seed)
            per_rbp_records.append(
                {
                    "task_id": task_id,
                    "dataset": task_id.split("/", 1)[0],
                    "rbp": task_id.split("/", 1)[1],
                    "model": "one_hot",
                    "layer_name": "one_hot",
                    "layer_index": None,
                    "auroc": m["auroc_mean"],
                    "accuracy": m["accuracy_mean"],
                    "f1": m["f1_mean"],
                    "auprc": m["auprc_mean"],
                }
            )

    for layer_name, layer_idx in layer_plan:
        print("\n" + "#" * 72)
        print(f"[layer] {layer_name} (index={layer_idx if layer_idx is not None else 'last'})")
        print("#" * 72)

        for model_name, spec in active_models.items():
            is_random = spec["weights_path"] == "__random__"
            mtype = spec["type"]
            max_length = spec["max_length"]
            tokenizer = _get_tokenizer(spec)

            print("\n" + "=" * 72)
            print(
                f"Model: {model_name} | type={mtype} | max_length={max_length} | layer={layer_name}"
            )
            print("=" * 72)

            if mtype == "lamar":
                model = (
                    mod.build_lamar_model_random(tokenizer, device, seed=args.seed)
                    if is_random
                    else mod.build_lamar_model(tokenizer, spec["weights_path"], device)
                )
            elif mtype == "dnabert2_singlenuc":
                model = mod.build_dnabert2_singlenuc_random(tokenizer, device, seed=args.seed)
            else:
                pad_id = tokenizer.pad_token_id or 0
                model = (
                    mod.build_dnabert2_model_random(spec["weights_path"], pad_id, device, seed=args.seed)
                    if is_random
                    else mod.build_dnabert2_model(spec["weights_path"], pad_id, device)
                )

            for task_id, splits in sorted(tasks.items()):
                df_train = splits["train_dev"]
                df_test = splits["test"]
                if args.max_samples_per_task > 0 and len(df_train) > args.max_samples_per_task:
                    df_train = df_train.sample(n=args.max_samples_per_task, random_state=args.seed)
                if args.max_samples_per_task > 0 and len(df_test) > args.max_samples_per_task:
                    df_test = df_test.sample(n=args.max_samples_per_task, random_state=args.seed)

                seq_train = df_train["sequence"].tolist()
                y_train = df_train["label"].to_numpy(dtype=np.int64)
                seq_test = df_test["sequence"].tolist()
                y_test = df_test["label"].to_numpy(dtype=np.int64)

                if mtype == "lamar":
                    build_train = lambda seqs=seq_train: mod.extract_lamar_embeddings(
                        seqs,
                        model,
                        tokenizer,
                        device,
                        max_length,
                        args.batch_size,
                        layer_idx,
                        desc=f"{model_name}/{layer_name}/{task_id[:40]}/train",
                    )
                    build_test = lambda seqs=seq_test: mod.extract_lamar_embeddings(
                        seqs,
                        model,
                        tokenizer,
                        device,
                        max_length,
                        args.batch_size,
                        layer_idx,
                        desc=f"{model_name}/{layer_name}/{task_id[:40]}/test",
                    )
                else:
                    build_train = lambda seqs=seq_train: mod.extract_dnabert2_embeddings(
                        seqs,
                        model,
                        tokenizer,
                        device,
                        max_length,
                        args.batch_size,
                        layer_idx,
                        desc=f"{model_name}/{layer_name}/{task_id[:40]}/train",
                    )
                    build_test = lambda seqs=seq_test: mod.extract_dnabert2_embeddings(
                        seqs,
                        model,
                        tokenizer,
                        device,
                        max_length,
                        args.batch_size,
                        layer_idx,
                        desc=f"{model_name}/{layer_name}/{task_id[:40]}/test",
                    )

                cache_layer_idx = mod._cache_layer_index(layer_idx)
                cache_model_key = f"{model_name}__{layer_name}"
                X_train = mod.get_or_build_emb(
                    cache_dir,
                    cache_model_key,
                    cache_layer_idx,
                    mod._cache_task_name(task_id, "train_dev"),
                    len(y_train),
                    build_train,
                )
                X_test = mod.get_or_build_emb(
                    cache_dir,
                    cache_model_key,
                    cache_layer_idx,
                    mod._cache_task_name(task_id, "test"),
                    len(y_test),
                    build_test,
                )

                m = mod.evaluate_linear_probe(X_train, y_train, X_test, y_test, seed=args.seed)
                per_rbp_records.append(
                    {
                        "task_id": task_id,
                        "dataset": task_id.split("/", 1)[0],
                        "rbp": task_id.split("/", 1)[1],
                        "model": model_name,
                        "layer_name": layer_name,
                        "layer_index": layer_idx,
                        "auroc": m["auroc_mean"],
                        "accuracy": m["accuracy_mean"],
                        "f1": m["f1_mean"],
                        "auprc": m["auprc_mean"],
                    }
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    per_rbp_df = pd.DataFrame(per_rbp_records)
    per_rbp_csv = output_dir / "per_rbp_metrics.csv"
    per_rbp_df.to_csv(per_rbp_csv, index=False)

    summary_df = summarize(per_rbp_df)
    summary_csv = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    stat_res = statistical_tests(per_rbp_df)
    stat_json = output_dir / "statistical_tests.json"
    with stat_json.open("w") as fh:
        json.dump(stat_res, fh, indent=2)

    print("\n[done] Results written:")
    print(f"  - {per_rbp_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {stat_json}")


if __name__ == "__main__":
    main()
