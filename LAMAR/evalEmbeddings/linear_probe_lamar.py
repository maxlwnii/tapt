"""
linear_probe_lamar.py
---------------------
Linear probing evaluation for LAMAR embeddings.

Evaluates two model variants:
  1. lamar_pretrained  — base LAMAR weights (LAMAR/weights)
  2. lamar_tapt        — task-adaptive pretraining checkpoint

For each variant, every transformer layer is probed via mean-pooling +
logistic regression (stratified 5-fold CV).  Results and plots are saved
to --output_dir.

Usage:
  python linear_probe_lamar.py
  python linear_probe_lamar.py --enable_layer_search --max_rbps 5
  python linear_probe_lamar.py --skip_plots --max_rbps 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

# LAMAR uses a custom ESM variant installed as an editable package under
# Thesis/LAMAR/.  Add that directory to sys.path so the import works regardless
# of the working directory the script is launched from.
_LAMAR_PKG_ROOT = Path(__file__).resolve().parent.parent  # …/Thesis/LAMAR/
if str(_LAMAR_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAMAR_PKG_ROOT))

from LAMAR.modeling_nucESM2 import EsmForMaskedLM  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear probing for RBP binding prediction using frozen LAMAR embeddings"
    )

    # --- Data ---
    p.add_argument(
        "--data_roots",
        nargs="+",
        default=[
            "/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/data",
            "/home/fr/fr_fr/fr_ml642/Thesis/data/finetune_data_koo",
        ],
        help="Root folders containing per-RBP subfolders with train/dev/test CSV files",
    )
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rbps", type=int, default=0, help="If >0, evaluate only first N RBPs")
    p.add_argument("--max_samples_per_rbp", type=int, default=0, help="If >0, cap samples per RBP")

    # --- Model paths ---
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/src/pretrain/"
                "saving_model/tapt_1024_standard_collator/checkpoint-134000",
        help="Path to HF tokenizer directory (must contain vocab.txt / tokenizer_config.json)",
    )
    p.add_argument(
        "--pretrained_weights",
        type=str,
        default="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/weights",
        help="Path to pretrained LAMAR weights file (safetensors)",
    )
    p.add_argument(
        "--tapt_checkpoint",
        type=str,
        default="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/src/pretrain/saving_model/tapt_1024_standard_collator/checkpoint-134000",
        help="Path to TAPT checkpoint directory (must contain model.safetensors)",
    )

    # --- Which models to run ---
    p.add_argument(
        "--models",
        nargs="+",
        default=["lamar_pretrained", "lamar_tapt"],
        choices=["lamar_pretrained", "lamar_tapt"],
        help="Which model variants to evaluate",
    )

    # --- Layer search ---
    p.add_argument(
        "--enable_layer_search",
        action="store_true",
        help="Run layer search on a pilot RBP; use best layer for probing",
    )
    p.add_argument(
        "--best_layer_override",
        type=int,
        default=None,
        help="Skip layer search and use this fixed layer index for all models",
    )
    p.add_argument(
        "--layer_search_pilot_rbp",
        type=str,
        default=None,
        help="RBP name to use as pilot for layer search (defaults to first alphabetically)",
    )
    p.add_argument(
        "--max_pilot_samples",
        type=int,
        default=4000,
        help="Max samples to use for layer search pilot",
    )

    # --- Output ---
    p.add_argument(
        "--output_dir",
        type=str,
        default="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/"
                "evalEmbeddings/results/linear_probe_full",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/"
                "evalEmbeddings/results/linear_probe_full/cache",
    )
    p.add_argument("--skip_plots", action="store_true", help="Skip all matplotlib output")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_column(columns: List[str], candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(f"Could not find column among candidates {candidates} in {columns}")


def load_rbp_tasks(data_roots: List[str]) -> Dict[str, pd.DataFrame]:
    """Walk data roots and collect all RBP tasks into a single dict."""
    tasks: Dict[str, List[pd.DataFrame]] = {}

    for root in data_roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[WARN] Data root not found, skipping: {root}")
            continue

        for rbp_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            csvs = [rbp_dir / s for s in ("train.csv", "dev.csv", "test.csv") if (rbp_dir / s).exists()]
            if not csvs:
                continue

            pieces = []
            for csv_path in csvs:
                df = pd.read_csv(csv_path)
                seq_col = _find_column(list(df.columns), ["sequence", "seq", "text", "input", "x"])
                lbl_col = _find_column(list(df.columns), ["label", "labels", "target", "y"])
                chunk = pd.DataFrame(
                    {"sequence": df[seq_col].astype(str), "label": df[lbl_col].astype(int)}
                ).dropna()
                pieces.append(chunk)

            if not pieces:
                continue

            merged = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["sequence", "label"])
            if merged["label"].nunique() < 2:
                continue

            tasks.setdefault(rbp_dir.name, []).append(merged)

    out: Dict[str, pd.DataFrame] = {}
    for rbp, dfs in tasks.items():
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["sequence", "label"])
        if merged["label"].nunique() >= 2:
            out[rbp] = merged

    return out


# ---------------------------------------------------------------------------
# LAMAR model loading
# ---------------------------------------------------------------------------

def _build_lamar_config(tokenizer) -> AutoConfig:
    """Construct the LAMAR ESM config from known architecture parameters."""
    cfg = AutoConfig.for_model(
        "esm",
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        token_dropout=False,
        positional_embedding_type="rotary",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        max_position_embeddings=1026,
    )
    return cfg


def _remap_lamar_weights(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap weight keys so they match EsmForMaskedLM's state_dict layout.

    The pretrained 'weights' file stores tensors without the 'esm.' prefix;
    checkpoint directories already have the correct layout.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("esm.lm_head"):
            # esm.lm_head.* → lm_head.*
            out[k[len("esm."):]] = v
        elif k.startswith("lm_head") or k.startswith("esm."):
            out[k] = v
        else:
            # bare tensors (encoder layers, embeddings) → prepend esm.
            out["esm." + k] = v
    return out


def load_lamar_weights(weights_path: str) -> Dict[str, torch.Tensor]:
    """Load weights from a file or from a directory containing model.safetensors."""
    p = Path(weights_path)
    if p.is_dir():
        st = p / "model.safetensors"
        if st.exists():
            return load_file(str(st))
        pb = p / "pytorch_model.bin"
        if pb.exists():
            return torch.load(str(pb), map_location="cpu")
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {weights_path}")
    # Single file (pretrained 'weights')
    return load_file(str(p))


def build_lamar_model(tokenizer, weights_path: str, device: torch.device) -> EsmForMaskedLM:
    """Build and return a ready-to-eval EsmForMaskedLM on ``device``."""
    config = _build_lamar_config(tokenizer)
    model = EsmForMaskedLM(config)

    raw = load_lamar_weights(weights_path)
    remapped = _remap_lamar_weights(raw)
    result = model.load_state_dict(remapped, strict=False)

    if result.missing_keys:
        # lm_head weights are not needed for embeddings; ignore silently
        non_trivial = [k for k in result.missing_keys if "lm_head" not in k]
        if non_trivial:
            print(f"  [WARN] Missing weight keys: {non_trivial[:10]}")
    if result.unexpected_keys:
        print(f"  [WARN] Unexpected weight keys: {result.unexpected_keys[:5]}")

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_all_hidden_states(
    model: EsmForMaskedLM,
    tokens: Dict[str, torch.Tensor],
) -> List[torch.Tensor]:
    """
    Return per-layer hidden states as a list of (B, seq_len, H) tensors.
    Index 0 = token embedding layer, index 1..N = transformer blocks.
    """
    out = model.esm(**tokens, output_hidden_states=True)
    return list(out.hidden_states)


def mean_pool(
    hidden: torch.Tensor,          # (B, seq_len, H)
    attention_mask: torch.Tensor,  # (B, seq_len)
    input_ids: torch.Tensor,       # (B, seq_len)
    special_ids: List[int],
) -> torch.Tensor:                 # (B, H)
    """Mean-pool hidden states, excluding padding and special tokens."""
    valid = attention_mask.bool()
    if special_ids:
        smask = torch.zeros_like(valid)
        for sid in special_ids:
            smask |= input_ids.eq(sid)
        valid = valid & ~smask
    denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (hidden * valid.unsqueeze(-1)).sum(dim=1) / denom


def extract_embeddings(
    sequences: List[str],
    model: EsmForMaskedLM,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    layer_idx: Optional[int],
    desc: str = "embed",
) -> np.ndarray:
    """
    Extract mean-pooled embeddings from ``layer_idx``.
    If ``layer_idx`` is None, uses the last transformer layer.
    """
    all_vecs: List[np.ndarray] = []
    special_ids = tokenizer.all_special_ids

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc=desc, unit="batch"):
            batch = sequences[start : start + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            hidden_states = get_all_hidden_states(model, tokens)
            idx = layer_idx if layer_idx is not None else (len(hidden_states) - 1)
            hidden = hidden_states[idx]

            pooled = mean_pool(hidden, tokens["attention_mask"], tokens["input_ids"], special_ids)
            all_vecs.append(pooled.cpu().numpy().astype(np.float32))

    return np.concatenate(all_vecs, axis=0)


def extract_all_layers(
    sequences: List[str],
    model: EsmForMaskedLM,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Dict[int, np.ndarray]:
    """
    Single forward-pass extraction of ALL layers.
    Returns {layer_idx: ndarray(N, H)}.
    """
    special_ids = tokenizer.all_special_ids

    # Discover layer count from dummy forward pass
    with torch.no_grad():
        dummy = tokenizer(
            sequences[:2], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        dummy = {k: v.to(device) for k, v in dummy.items()}
        n_layers = len(get_all_hidden_states(model, dummy))

    print(f"  [layer_search] {n_layers} hidden states "
          f"(1 embedding + {n_layers - 1} transformer blocks)")

    layer_vecs: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_layers)}

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="all-layers", unit="batch"):
            batch = sequences[start : start + batch_size]
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            hidden_states = get_all_hidden_states(model, tokens)
            for li, h in enumerate(hidden_states):
                pooled = mean_pool(h, tokens["attention_mask"], tokens["input_ids"], special_ids)
                layer_vecs[li].append(pooled.cpu().numpy().astype(np.float32))

    return {i: np.concatenate(vecs, axis=0) for i, vecs in layer_vecs.items()}


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _emb_cache_path(cache_dir: Path, model_name: str, layer_idx: int, rbp: str) -> Path:
    d = cache_dir / model_name / f"layer_{layer_idx:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{rbp.replace('/', '_')}.npy"


def get_or_build_layer_emb(
    cache_dir: Path,
    model_name: str,
    layer_idx: int,
    rbp: str,
    build_fn,
) -> np.ndarray:
    p = _emb_cache_path(cache_dir, model_name, layer_idx, rbp)
    if p.exists():
        return np.load(p)
    X = build_fn()
    np.save(p, X)
    return X


# ---------------------------------------------------------------------------
# Layer search
# ---------------------------------------------------------------------------

def probe_layer_auroc(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> float:
    """Mean AUROC across stratified K-folds for a single layer."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
        )
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y[te], prob))
    return float(np.mean(aurocs))


def run_layer_search(
    sequences: List[str],
    labels: np.ndarray,
    model: EsmForMaskedLM,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    model_name: str,
    cache_dir: Path,
    output_dir: Path,
    rbp_name: str,
    num_folds: int,
    seed: int,
) -> int:
    """
    Probe every layer on ``sequences``/``labels``; return best layer index.
    Results are saved to ``output_dir / layer_search_{model_name}.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Layer Search | model={model_name} | pilot RBP={rbp_name} ===")

    # Check cache first
    n_layers_cached: Optional[int] = None
    layer_dirs = sorted((cache_dir / model_name).glob("layer_*")) if (cache_dir / model_name).exists() else []
    if layer_dirs:
        n_layers_cached = len(layer_dirs)

    all_layer_emb: Optional[Dict[int, np.ndarray]] = None
    if n_layers_cached is not None:
        # Try to load from cache
        cached = {}
        ok = True
        for li in range(n_layers_cached):
            p = _emb_cache_path(cache_dir, model_name, li, rbp_name)
            if not p.exists():
                ok = False
                break
            cached[li] = np.load(p)
        if ok:
            all_layer_emb = cached
            print(f"  [layer_search] Loaded {n_layers_cached} layers from cache")

    if all_layer_emb is None:
        print(f"  [layer_search] Extracting all layers …")
        all_layer_emb = extract_all_layers(
            sequences, model, tokenizer, device, max_length, batch_size
        )
        # Save to cache
        for li, X in all_layer_emb.items():
            np.save(_emb_cache_path(cache_dir, model_name, li, rbp_name), X)

    # Probe each layer
    layer_aurocs: Dict[int, float] = {}
    for li, X in sorted(all_layer_emb.items()):
        if np.linalg.norm(X, axis=1).mean() < 1e-6:
            print(f"    Layer {li:2d}: SKIPPED (zero embeddings)")
            continue
        auroc = probe_layer_auroc(X, labels, n_splits=num_folds, seed=seed)
        layer_aurocs[li] = auroc
        print(f"    Layer {li:2d}: AUROC = {auroc:.4f}")

    best_layer = max(layer_aurocs, key=layer_aurocs.get)
    best_auroc = layer_aurocs[best_layer]

    results = {
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "pilot_rbp": rbp_name,
        "model": model_name,
        "layer_aurocs": {str(k): v for k, v in layer_aurocs.items()},
    }
    out_json = output_dir / f"layer_search_{model_name}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n  Best layer for {model_name}: "
        f"Layer {best_layer} → AUROC = {best_auroc:.4f}\n"
        f"  Saved → {out_json}"
    )
    return best_layer


# ---------------------------------------------------------------------------
# Linear probe evaluation
# ---------------------------------------------------------------------------

def evaluate_linear_probe(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    metrics: Dict[str, List[float]] = {"auroc": [], "accuracy": [], "f1": [], "auprc": []}

    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
        )
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics["auroc"].append(roc_auc_score(y[te], prob))
        metrics["accuracy"].append(accuracy_score(y[te], pred))
        metrics["f1"].append(f1_score(y[te], pred))
        metrics["auprc"].append(average_precision_score(y[te], prob))

    return {f"{k}_mean": float(np.mean(v)) for k, v in metrics.items()} | \
           {f"{k}_std":  float(np.std(v))  for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_auroc_curve(layer_aurocs: Dict[int, float], title: str, out_path: Path) -> None:
    layers = sorted(layer_aurocs)
    aurocs = [layer_aurocs[i] for i in layers]
    best = max(layer_aurocs, key=layer_aurocs.get)

    plt.figure(figsize=(10, 4))
    plt.plot(layers, aurocs, marker="o", linewidth=1.5)
    plt.axvline(best, linestyle="--", color="red", label=f"Best: layer {best}")
    plt.scatter([best], [layer_aurocs[best]], color="red", zorder=5)
    plt.xlabel("Layer index  (0 = token embeddings)")
    plt.ylabel("Mean AUROC (CV)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    model_order = sorted(df["model"].unique())
    data = [df.loc[df["model"] == m, "auroc"].values for m in model_order]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=model_order, showmeans=True)
    plt.ylabel("AUROC")
    plt.title("AUROC distribution across RBPs")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_rbp_bars(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="rbp", columns="model", values="auroc")
    base = pivot.columns[0]
    pivot = pivot.sort_values(base)
    ax = pivot.plot(kind="bar", figsize=(14, 6), width=0.8)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-RBP AUROC by model")
    ax.legend(loc="lower right")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_delta(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="rbp", columns="model", values="auroc")
    cols = list(pivot.columns)
    if len(cols) < 2:
        return
    ref, other = cols[0], cols[1]
    delta = (pivot[other] - pivot[ref]).sort_values()
    plt.figure(figsize=(12, 5))
    plt.bar(delta.index, delta.values)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel(f"ΔAUROC ({other} - {ref})")
    plt.title("Per-RBP AUROC delta")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(per_rbp_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, grp in per_rbp_df.groupby("model"):
        rows.append(
            {
                "Model": model_name,
                "Mean AUROC": grp["auroc"].mean(),
                "Std AUROC": grp["auroc"].std(ddof=0),
                "Mean F1": grp["f1"].mean(),
                "Std F1": grp["f1"].std(ddof=0),
                "Mean AUPRC": grp["auprc"].mean(),
                "Std AUPRC": grp["auprc"].std(ddof=0),
                "N RBPs": len(grp),
            }
        )
    return pd.DataFrame(rows).sort_values("Mean AUROC", ascending=False).reset_index(drop=True)


def statistical_tests(per_rbp_df: pd.DataFrame) -> Dict:
    pivot = per_rbp_df.pivot(index="rbp", columns="model", values="auroc")
    all_cols = list(pivot.columns)
    if len(all_cols) < 2:
        return {}

    ref = all_cols[0]
    results = {}
    for other in all_cols[1:]:
        paired = pivot[[ref, other]].dropna()
        if len(paired) < 5:
            print(f"[WARN] Not enough shared RBPs for Wilcoxon {other} vs {ref}")
            continue
        try:
            w = wilcoxon(paired[other], paired[ref], alternative="two-sided")
            key = f"wilcoxon_{other}_vs_{ref}"
            results[key] = {
                "statistic": float(w.statistic),
                "pvalue": float(w.pvalue),
                "n_rbps": int(len(paired)),
                "mean_delta_auroc": float((paired[other] - paired[ref]).mean()),
                "reference_model": ref,
            }
            print(f"  {key}: p={w.pvalue:.4f}, Δ={results[key]['mean_delta_auroc']:+.4f}")
        except Exception as exc:
            print(f"[WARN] Wilcoxon failed for {other} vs {ref}: {exc}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data ---
    tasks = load_rbp_tasks(args.data_roots)
    if not tasks:
        raise RuntimeError("No valid RBP tasks found in provided data roots.")

    tasks_items = sorted(tasks.items())
    if args.max_rbps > 0:
        tasks_items = tasks_items[: args.max_rbps]
        tasks = dict(tasks_items)

    print(f"Loaded {len(tasks)} RBP tasks")

    # --- Define model variants ---
    model_weight_paths: Dict[str, str] = {}
    if "lamar_pretrained" in args.models:
        model_weight_paths["lamar_pretrained"] = args.pretrained_weights
    if "lamar_tapt" in args.models:
        model_weight_paths["lamar_tapt"] = args.tapt_checkpoint

    # Validate paths
    for name, wpath in list(model_weight_paths.items()):
        if not Path(wpath).exists():
            print(f"[WARN] Weights not found for '{name}': {wpath} — skipping")
            del model_weight_paths[name]

    if not model_weight_paths:
        raise RuntimeError("No valid model variants to evaluate.")

    # --- Tokenizer (shared across both variants) ---
    print(f"\nLoading LAMAR tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.max_length,
    )
    print(f"  vocab_size={len(tokenizer)}, "
          f"pad_token_id={tokenizer.pad_token_id}, "
          f"mask_token_id={tokenizer.mask_token_id}")

    # --- Layer search ---
    best_layer_per_model: Dict[str, Optional[int]] = {m: None for m in model_weight_paths}

    if args.best_layer_override is not None:
        for m in best_layer_per_model:
            best_layer_per_model[m] = args.best_layer_override
        print(f"\n[INFO] Using best_layer_override={args.best_layer_override} for all models")

    elif args.enable_layer_search:
        pilot_rbp = args.layer_search_pilot_rbp or tasks_items[0][0]
        if pilot_rbp not in tasks:
            print(f"[WARN] Pilot RBP '{pilot_rbp}' not found; using first alphabetically.")
            pilot_rbp = tasks_items[0][0]

        pilot_df = tasks[pilot_rbp]
        if args.max_pilot_samples > 0 and len(pilot_df) > args.max_pilot_samples:
            pilot_df = pilot_df.sample(n=args.max_pilot_samples, random_state=args.seed)
        pilot_seqs = pilot_df["sequence"].tolist()
        pilot_labels = pilot_df["label"].to_numpy(dtype=np.int64)

        print(f"\nLayer search pilot: '{pilot_rbp}' ({len(pilot_seqs)} samples)")

        for model_name, wpath in model_weight_paths.items():
            print(f"\n  Loading {model_name} for layer search …")
            model = build_lamar_model(tokenizer, wpath, device)
            best = run_layer_search(
                sequences=pilot_seqs,
                labels=pilot_labels,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                model_name=model_name,
                cache_dir=cache_dir / "layer_search",
                output_dir=output_dir / "layer_search",
                rbp_name=pilot_rbp,
                num_folds=args.num_folds,
                seed=args.seed,
            )
            best_layer_per_model[model_name] = best
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Main probing loop ---
    per_rbp_records: List[dict] = []

    for model_name, wpath in model_weight_paths.items():
        layer_idx = best_layer_per_model[model_name]
        cache_label = f"{model_name}_L{layer_idx}" if layer_idx is not None else model_name

        print(f"\n{'='*60}")
        print(f"Model: {model_name}  |  layer={layer_idx if layer_idx is not None else 'last'}")
        print(f"Weights: {wpath}")
        print(f"{'='*60}")

        model = build_lamar_model(tokenizer, wpath, device)

        for rbp, df in sorted(tasks.items()):
            if args.max_samples_per_rbp > 0 and len(df) > args.max_samples_per_rbp:
                df = df.sample(n=args.max_samples_per_rbp, random_state=args.seed)

            sequences = df["sequence"].tolist()
            labels = df["label"].to_numpy(dtype=np.int64)

            X = get_or_build_layer_emb(
                cache_dir,
                model_name,
                layer_idx if layer_idx is not None else -1,
                rbp,
                lambda seqs=sequences, li=layer_idx: extract_embeddings(
                    seqs, model, tokenizer, device,
                    args.max_length, args.batch_size, li,
                    desc=f"{model_name} [{rbp}]",
                ),
            )

            metrics = evaluate_linear_probe(X, labels, n_splits=args.num_folds, seed=args.seed)
            per_rbp_records.append(
                {
                    "rbp": rbp,
                    "model": cache_label,
                    "layer": layer_idx,
                    "auroc": metrics["auroc_mean"],
                    "auroc_std": metrics["auroc_std"],
                    "accuracy": metrics["accuracy_mean"],
                    "f1": metrics["f1_mean"],
                    "auprc": metrics["auprc_mean"],
                }
            )
            print(
                f"  {rbp:35s}  auroc={metrics['auroc_mean']:.4f}±{metrics['auroc_std']:.4f}"
                f"  f1={metrics['f1_mean']:.4f}  auprc={metrics['auprc_mean']:.4f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Save results ---
    per_rbp_df = pd.DataFrame(per_rbp_records)
    per_rbp_df.to_csv(output_dir / "per_rbp_metrics.csv", index=False)

    summary_df = summarize(per_rbp_df)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    print("\n=== Summary (mean ± std across RBPs) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Statistical tests ---
    stat_results = statistical_tests(per_rbp_df)
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)

    # --- Plots ---
    if not args.skip_plots:
        plot_boxplot(per_rbp_df, plots_dir / "auroc_boxplot.png")
        plot_per_rbp_bars(per_rbp_df, plots_dir / "per_rbp_auroc_bars.png")
        plot_delta(per_rbp_df, plots_dir / "delta_auroc.png")
        print(f"\nPlots saved to: {plots_dir}")

    if best_layer_per_model and any(v is not None for v in best_layer_per_model.values()):
        print(f"\nBest layers used: {best_layer_per_model}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
