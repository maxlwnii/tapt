"""
linear_probe_cross_length.py
============================
Unified linear probing evaluation for ALL model variants using the
cross-length generalisation dataset (Thesis/data/cross_length/).

Model variants supported
------------------------
  LAMAR family (EsmForMaskedLM, max_length=1024 or 512):
    lamar_pretrained          – LAMAR/weights  (max_length=1024)
    lamar_tapt_1024           – tapt_1024_standard_collator/ckpt-134000
    lamar_tapt_standard_1gpu  – tapt_1024_standard_collator_1gpu/ckpt-232000
    lamar_tapt_custom_1gpu    – tapt_1024_custom_collator_1gpu/ckpt-232000
    lamar_tapt_512            – tapt_lamar/checkpoint-98000  (max_length=512)
    lamar_random              – randomly-initialised LAMAR (lower-bound baseline)

  DNABERT-2 family (AutoModel, max_length=512):
    dnabert2_pretrained       – zhihan1996/DNABERT-2-117M
    dnabert2_tapt             – dnabert2_standard_mlm/checkpoint-25652
    dnabert2_random           – randomly-initialised DNABERT-2 (lower-bound baseline)

  Baseline:
    one_hot                   – one-hot encoding (no model)

Per-model × per-task the script:
  1. Extracts mean-pooled embeddings from the best (or last) transformer layer
  2. Caches embeddings as .npy so reruns are cheap
  3. Runs stratified K-fold logistic-regression probing (AUROC, F1, AUPRC, Acc)

Outputs (in --output_dir)
-------------------------
  per_rbp_metrics.csv         – one row per (task × model)
  summary_metrics.csv         – mean ± std across tasks per model
  statistical_tests.json      – paired Wilcoxon tests vs reference model
  embedding_health_stats.csv  – norm / cosine stats per model
  plots/auroc_boxplot.png
  plots/per_rbp_auroc_bars.png
  plots/delta_auroc.png
  plots/{model}_umap.png or {model}_tsne.png

Usage
-----
    python linear_probe_cross_length.py                          # uses DNABERT-2 layer 4 / LAMAR layer 6 by default
    python linear_probe_cross_length.py --models lamar_pretrained dnabert2_pretrained
    python linear_probe_cross_length.py --best_layer_override 9
    python linear_probe_cross_length.py --enable_layer_search
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm

# ── locate LAMAR package ──────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent          # …/linear_probe_cross_length/
_THESIS_ROOT = _SCRIPT_DIR.parent                       # …/Thesis/
_LAMAR_PKG   = _THESIS_ROOT / "LAMAR"
if str(_LAMAR_PKG) not in sys.path:
    sys.path.insert(0, str(_LAMAR_PKG))

from LAMAR.modeling_nucESM2 import EsmForMaskedLM  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Paths (all relative to _THESIS_ROOT so they survive directory changes)
# ─────────────────────────────────────────────────────────────────────────────
_BASE     = str(_THESIS_ROOT)
_DB2_BASE = f"{_BASE}/DNABERT2"

_DEFAULT_DATA_ROOT = f"{_BASE}/data/cross_length"
_DEFAULT_NUM_RANDOM_RBPS = 44

_MODEL_DEFAULTS = {
    # LAMAR -------------------------------------------------------------------
    "lamar_pretrained": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/weights",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   1024,
    },
    "lamar_tapt_1024": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   1024,
    },
    "lamar_tapt_standard_1gpu": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator_1gpu/checkpoint-232000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   1024,
    },
    "lamar_tapt_custom_1gpu": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_custom_collator_1gpu/checkpoint-232000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   1024,
    },
    "lamar_tapt_512": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_lamar/checkpoint-98000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   512,
    },
    "lamar_tapt_512_std": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_512_standard_collator_1gpu/checkpoint-265000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   512,
    },
    "lamar_random": {
        "type":         "lamar",
        "weights_path": "__random__",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   1024,
    },
    # DNABERT-2 ---------------------------------------------------------------
    "dnabert2_pretrained": {
        "type":         "dnabert2",
        "weights_path": "zhihan1996/DNABERT-2-117M",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
    "dnabert2_tapt": {
        "type":         "dnabert2",
        "weights_path": f"{_DB2_BASE}/pretrain/models"
                        "/dnabert2_standard_mlm/checkpoint-25652",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
    "dnabert2_tapt_v3": {
        "type":         "dnabert2",
        "weights_path": f"{_DB2_BASE}/pretrain/models"
                        "/dnabert2_tapt_v3/checkpoint-2566",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
    "dnabert2_random": {
        "type":         "dnabert2",
        "weights_path": "__random__",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
}

_ALL_MODEL_NAMES = list(_MODEL_DEFAULTS.keys()) + ["one_hot"]

# Default fixed probing layers.
# LAMAR hidden states are indexed as: 0 = embeddings, 1..N = transformer blocks.
# DNABERT-2 encoder layers are indexed directly from the BertLayer stack.
_DEFAULT_LAYER_BY_TYPE = {
    "lamar": 6,
    "dnabert2": 4,
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear probing on cross-length data for all DNA-LM variants"
    )

    # Data
    p.add_argument(
        "--data_root",
        type=str,
        default=_DEFAULT_DATA_ROOT,
        help="Root folder whose sub-directories are per-RBP tasks with train/dev/test CSV files",
    )
    p.add_argument("--batch_size",             type=int, default=32)
    p.add_argument("--num_folds",              type=int, default=5)
    p.add_argument("--seed",                   type=int, default=42)
    p.add_argument("--num_random_rbps",        type=int, default=_DEFAULT_NUM_RANDOM_RBPS,
                   help="Number of randomly sampled tasks to evaluate (0 = use all tasks)")
    p.add_argument("--max_samples_per_rbp",    type=int, default=0,
                   help="If >0, cap samples per task")
    p.add_argument("--max_points_vis",         type=int, default=5000)
    p.add_argument("--max_points_cosine",      type=int, default=1200)

    # Model selection
    p.add_argument(
        "--models",
        nargs="+",
        default=["lamar_pretrained", "lamar_tapt_1024", "lamar_tapt_standard_1gpu",
                 "lamar_tapt_custom_1gpu", "lamar_tapt_512", "lamar_tapt_512_std", "lamar_random",
                 "dnabert2_pretrained", "dnabert2_tapt", "dnabert2_tapt_v3", "dnabert2_random",
                 "one_hot"],
        choices=_ALL_MODEL_NAMES,
        help="Subset of model variants to evaluate",
    )

    # Layer search
    p.add_argument("--enable_layer_search",    action="store_true",
                   help="Run per-model layer search on a pilot task; use best layer for all tasks")
    p.add_argument("--best_layer_override",    type=int, default=None,
                   help="Skip layer search and use this fixed layer index for ALL transformer models")
    p.add_argument("--layer_search_pilot_rbp", type=str, default=None,
                   help="Task name to use as pilot (default: first alphabetically)")
    p.add_argument("--max_pilot_samples",      type=int, default=4000)

    # Paths
    p.add_argument(
        "--output_dir",
        type=str,
        default=f"{_BASE}/linear_probe_cross_length/results",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=f"{_BASE}/linear_probe_cross_length/results/cache",
    )

    # Misc
    p.add_argument("--skip_plots",  action="store_true", help="Skip all matplotlib output")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_column(columns: List[str], candidates: List[str]) -> str:
    lmap = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lmap:
            return lmap[cand]
    raise ValueError(f"None of {candidates} found in columns: {columns}")


def load_tasks(data_root: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Walk data_root; each immediate sub-directory is one task.
    Returns {task_name: {"train_dev": df, "test": df}} with columns [sequence, label].
    Tasks with fewer than 2 unique labels are skipped.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    tasks: Dict[str, Dict[str, pd.DataFrame]] = {}

    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        train_dev_files = [task_dir / s for s in ("train.csv", "dev.csv")
                           if (task_dir / s).exists()]
        test_file = task_dir / "test.csv"
        if not train_dev_files or not test_file.exists():
            continue

        pieces_train_dev = []
        for csv_path in train_dev_files:
            df = pd.read_csv(csv_path)
            seq_col   = _find_column(list(df.columns), ["sequence", "seq", "text", "input", "x"])
            label_col = _find_column(list(df.columns), ["label", "labels", "target", "y"])
            chunk = pd.DataFrame({
                "sequence": df[seq_col].astype(str),
                "label":    df[label_col].astype(int),
            }).dropna()
            # Replace RNA uracil → thymine so all models run on DNA alphabet
            chunk["sequence"] = chunk["sequence"].str.replace("U", "T").str.replace("u", "t")
            pieces_train_dev.append(chunk)

        df_test = pd.read_csv(test_file)
        seq_col_t   = _find_column(list(df_test.columns), ["sequence", "seq", "text", "input", "x"])
        label_col_t = _find_column(list(df_test.columns), ["label", "labels", "target", "y"])
        test_chunk = pd.DataFrame({
            "sequence": df_test[seq_col_t].astype(str),
            "label":    df_test[label_col_t].astype(int),
        }).dropna()
        test_chunk["sequence"] = test_chunk["sequence"].str.replace("U", "T").str.replace("u", "t")

        if not pieces_train_dev:
            continue

        merged_train_dev = (
            pd.concat(pieces_train_dev, ignore_index=True)
            .drop_duplicates(subset=["sequence"])  # FIX: dedupe by sequence only to remove conflicting labels.
        )
        merged_test = test_chunk.drop_duplicates(subset=["sequence"])  # FIX: keep one label per sequence in test split.

        if merged_train_dev["label"].nunique() < 2 or merged_test["label"].nunique() < 2:
            continue

        # FIX: preserve cross-length protocol by keeping train+dev and test separated.
        tasks[task_dir.name] = {"train_dev": merged_train_dev, "test": merged_test}

    print(f"[data] Found {len(tasks)} valid tasks in {data_root}")
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# One-hot encoding
# ─────────────────────────────────────────────────────────────────────────────

def one_hot_embed(sequences: List[str], max_length: int) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    embs = np.zeros((len(sequences), max_length * 5), dtype=np.float32)  # FIX: add explicit padding channel.
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        mat = np.zeros((max_length, 5), dtype=np.float32)
        mat[:, 4] = 1.0  # FIX: mark padded positions by default; overwritten for real tokens.
        for j, ch in enumerate(seq[:max_length]):
            idx = mapping.get(ch)
            if idx is not None:
                mat[j, idx] = 1.0
                mat[j, 4] = 0.0
        embs[i] = mat.reshape(-1)
    return embs


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation helper (lower-bound random baselines)
# ─────────────────────────────────────────────────────────────────────────────

def _wolf_init(module: torch.nn.Module) -> None:
    """GPT-style (Wolf) weight initialisation."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────────
# LAMAR model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_lamar_config(tokenizer) -> AutoConfig:
    return AutoConfig.for_model(
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


def _remap_lamar_weights(raw: dict) -> dict:
    out: dict = {}
    for k, v in raw.items():
        if k.startswith("esm.lm_head"):
            out[k[len("esm."):]] = v
        elif k.startswith("lm_head") or k.startswith("esm."):
            out[k] = v
        else:
            out["esm." + k] = v
    return out


def _load_weights_file(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        st = p / "model.safetensors"
        if st.exists():
            return load_file(str(st))
        pb = p / "pytorch_model.bin"
        if pb.exists():
            return torch.load(str(pb), map_location="cpu")
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {path}")
    return load_file(str(p))


def build_lamar_model(tokenizer, weights_path: str, device: torch.device) -> EsmForMaskedLM:
    config = _build_lamar_config(tokenizer)
    model  = EsmForMaskedLM(config)
    raw    = _load_weights_file(weights_path)
    remapped = _remap_lamar_weights(raw)
    result   = model.load_state_dict(remapped, strict=False)
    non_trivial = [k for k in result.missing_keys if "lm_head" not in k]
    non_trivial_unexpected = [
        k for k in result.unexpected_keys
        if "rotary_embeddings.inv_freq" not in k
    ]
    if non_trivial:
        print(f"  [WARN] Missing (non-lm_head) keys: {non_trivial[:5]}")
    if non_trivial_unexpected:
        print(f"  [WARN] Unexpected LAMAR keys: {non_trivial_unexpected[:5]}")
    model.to(device).eval()
    return model


def build_lamar_model_random(tokenizer, device: torch.device, seed: int) -> EsmForMaskedLM:
    set_seed(seed)
    config = _build_lamar_config(tokenizer)
    model  = EsmForMaskedLM(config)
    model.apply(_wolf_init)
    print(f"  [lamar_random] Wolf-initialised, seed={seed}")
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# DNABERT-2 model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _materialize_meta_tensors(model: torch.nn.Module) -> torch.nn.Module:
    """Fix any tensors stuck on the 'meta' device (DNABERT-2 alibi tensors)."""
    for name, mod in model.named_modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "meta":
                new = torch.zeros(buf.shape, dtype=buf.dtype)
                mod.register_buffer(bname, new)
                print(f"  [meta→cpu] buffer {name}.{bname}")
        for pname, par in list(mod.named_parameters(recurse=False)):
            if par.device.type == "meta":
                new = torch.nn.Parameter(
                    torch.zeros(par.shape, dtype=par.dtype),
                    requires_grad=par.requires_grad,
                )
                setattr(mod, pname, new)
                print(f"  [meta→cpu] param  {name}.{pname}")
    return model


def _load_dnabert2_state_dict(weights_path: str) -> dict:
    p = Path(weights_path)
    if p.is_dir():
        st = p / "model.safetensors"
        if st.exists():
            return load_file(str(st))
        pb = p / "pytorch_model.bin"
        if pb.exists():
            state = torch.load(str(pb), map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                return state.get("state_dict", state.get("model", state))
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {weights_path}")
    if p.suffix == ".safetensors":
        return load_file(str(p))
    state = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        return state.get("state_dict", state.get("model", state))
    return state


def _dnabert2_needs_remote_code_fallback(weights_path: str) -> bool:
    p = Path(weights_path)
    config_path = p / "config.json"
    if not p.is_dir() or not config_path.exists():
        return False

    with open(config_path) as f:
        cfg = json.load(f)

    auto_map = cfg.get("auto_map") or {}
    uses_local_custom_code = any(
        isinstance(value, str) and "--" not in value and "." in value
        for value in auto_map.values()
    )
    missing_files = [
        name for name in (
            "configuration_bert.py",
            "bert_layers.py",
            "bert_padding.py",
            "flash_attn_triton.py",
        )
        if not (p / name).exists()
    ]
    if uses_local_custom_code and missing_files:
        print(
            f"[INFO] DNABERT-2 fallback remote code for {weights_path}; missing local files: {missing_files}"
        )
        return True
    return False


def _load_dnabert2_config(weights_path: str, fallback_model: str, pad_token_id: int):
    if _dnabert2_needs_remote_code_fallback(weights_path):
        config = AutoConfig.from_pretrained(fallback_model, trust_remote_code=True)
        with open(Path(weights_path) / "config.json") as f:
            local_cfg = json.load(f)
        for key, value in local_cfg.items():
            if key in {"auto_map", "_name_or_path", "transformers_version"}:
                continue
            try:
                object.__setattr__(config, key, value)
            except Exception:
                pass
    else:
        config = AutoConfig.from_pretrained(weights_path, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or getattr(config, "pad_token_id") is None:
        object.__setattr__(config, "pad_token_id", int(pad_token_id))
    return config


def build_dnabert2_model(weights_path: str, pad_token_id: int, device: torch.device) -> AutoModel:
    print(f"[INFO] Loading DNABERT-2: {weights_path}")
    fallback_model = "zhihan1996/DNABERT-2-117M"
    config = _load_dnabert2_config(weights_path, fallback_model, pad_token_id)
    if _dnabert2_needs_remote_code_fallback(weights_path):
        model = AutoModel.from_config(config, trust_remote_code=True)
        state_dict = _load_dnabert2_state_dict(weights_path)
        result = model.load_state_dict(state_dict, strict=False)
        non_trivial = [k for k in result.missing_keys if "position_ids" not in k]
        if non_trivial:
            print(f"  [WARN] Missing keys after fallback load: {non_trivial[:5]}")
    else:
        model = AutoModel.from_pretrained(
            weights_path,
            trust_remote_code=True,
            config=config,
            _fast_init=False,
        )
    model = _materialize_meta_tensors(model)
    model.to(device).eval()
    return model


def build_dnabert2_model_random(weights_path: str, pad_token_id: int,
                                device: torch.device, seed: int) -> AutoModel:
    """Random-init DNABERT-2 with the same architecture (lower-bound baseline)."""
    set_seed(seed)
    fallback_model = "zhihan1996/DNABERT-2-117M"
    config_source = (
        fallback_model
        if weights_path == "__random__" or _dnabert2_needs_remote_code_fallback(weights_path)
        else weights_path
    )
    config = _load_dnabert2_config(config_source, fallback_model, pad_token_id)
    model = AutoModel.from_config(config, trust_remote_code=True)
    model.apply(_wolf_init)
    model = _materialize_meta_tensors(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [dnabert2_random] Wolf-init from {weights_path}, {n_params:,} params, seed={seed}")
    model.to(device).eval()
    return model


def _get_dnabert2_last_layer(model: AutoModel) -> torch.nn.Module:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("DNABERT-2 model has no 'encoder' attribute")
    layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError("Cannot find encoder.layer / encoder.layers on DNABERT-2")
    return layers[-1]


def _get_dnabert2_layer(model: AutoModel, layer_idx: int) -> torch.nn.Module:
    """Return the BertLayer at ``layer_idx`` (supports negative indexing)."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("DNABERT-2 model has no 'encoder' attribute")
    layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError("Cannot find encoder.layer / encoder.layers on DNABERT-2")
    return layers[layer_idx]


def _is_unpadded_layer_output(raw_hidden: torch.Tensor, attention_mask: torch.Tensor) -> bool:
    # FIX: robustly infer unpadded behavior from actual output shape instead of brittle signature inspection.
    # FIX: the ndim==2 guard prevents false positives from regular padded outputs (which are 3D).
    if raw_hidden.ndim != 2:
        return False
    expected_tokens = int(attention_mask.sum().item())
    return int(raw_hidden.shape[0]) == expected_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Mean-pool helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mean_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    special_ids: Set[int],
) -> np.ndarray:
    valid = attention_mask.bool()
    for sid in special_ids:
        valid = valid & ~input_ids.eq(sid)
    denom  = valid.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
    pooled = (hidden.float() * valid.unsqueeze(-1).float()).sum(dim=1) / denom
    return pooled.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# LAMAR embedding extraction (supports all-layers mode for layer search)
# ─────────────────────────────────────────────────────────────────────────────

def _lamar_hidden_states(model: EsmForMaskedLM, tokens: dict) -> List[torch.Tensor]:
    out = model.esm(**tokens, output_hidden_states=True)
    return list(out.hidden_states)   # index 0 = embeddings, 1..N = transformer blocks


def extract_lamar_embeddings(
    sequences: List[str],
    model: EsmForMaskedLM,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    layer_idx: Optional[int],
    desc: str = "lamar",
) -> np.ndarray:
    special_ids: Set[int] = set(tokenizer.all_special_ids)
    all_vecs: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc=desc, unit="batch"):
            batch  = sequences[start : start + batch_size]
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hs     = _lamar_hidden_states(model, tokens)
            idx    = layer_idx if layer_idx is not None else (len(hs) - 1)
            pooled = _mean_pool(hs[idx], tokens["attention_mask"],
                                tokens["input_ids"], special_ids)
            all_vecs.append(pooled)

    return np.concatenate(all_vecs, axis=0)


def extract_lamar_all_layers(
    sequences: List[str],
    model: EsmForMaskedLM,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Dict[int, np.ndarray]:
    """Extract mean-pooled embeddings for every layer in a single pass."""
    special_ids: Set[int] = set(tokenizer.all_special_ids)

    # Probe layer count
    with torch.no_grad():
        dummy  = tokenizer(sequences[:2], return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)
        dummy  = {k: v.to(device) for k, v in dummy.items()}
        n_layers = len(_lamar_hidden_states(model, dummy))

    layer_vecs: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_layers)}

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="lamar-all-layers", unit="batch"):
            batch  = sequences[start : start + batch_size]
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hs     = _lamar_hidden_states(model, tokens)
            for li, h in enumerate(hs):
                layer_vecs[li].append(
                    _mean_pool(h, tokens["attention_mask"], tokens["input_ids"], special_ids)
                )

    return {i: np.concatenate(v, axis=0) for i, v in layer_vecs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# DNABERT-2 embedding extraction (hook-based to avoid FlashAttention crash)
# ─────────────────────────────────────────────────────────────────────────────

def extract_dnabert2_embeddings(
    sequences: List[str],
    model: AutoModel,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    layer_idx: Optional[int],
    desc: str = "dnabert2",
) -> np.ndarray:
    special_ids: Set[int] = set()
    for tid in (tokenizer.cls_token_id, tokenizer.sep_token_id,
                tokenizer.eos_token_id, tokenizer.bos_token_id):
        if tid is not None:
            special_ids.add(tid)

    target_layer = (
        _get_dnabert2_layer(model, layer_idx)
        if layer_idx is not None
        else _get_dnabert2_last_layer(model)
    )

    all_vecs: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc=desc, unit="batch"):
            batch  = sequences[start : start + batch_size]
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            captured: List[torch.Tensor] = []

            def _hook(_m, _inp, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                captured.append(h.detach())

            handle = target_layer.register_forward_hook(_hook)
            try:
                model(**tokens)
            finally:
                handle.remove()

            raw = captured[0]  # (ntokens_unpad, H) or (B, S, H)

            if _is_unpadded_layer_output(raw, tokens["attention_mask"]):
                B, S = tokens["attention_mask"].shape
                H    = raw.shape[-1]
                flat = tokens["attention_mask"].bool().view(-1)
                buf  = torch.zeros(B, S, H, device=raw.device, dtype=raw.dtype)
                buf.view(-1, H)[flat] = raw
                last_hidden = buf
            else:
                last_hidden = raw

            pooled = _mean_pool(last_hidden, tokens["attention_mask"],
                                tokens["input_ids"], special_ids)
            all_vecs.append(pooled)

    return np.concatenate(all_vecs, axis=0)


def extract_dnabert2_all_layers(
    sequences: List[str],
    model: AutoModel,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Dict[int, np.ndarray]:
    """Extract mean-pooled embeddings for ALL BertLayers via hooks."""
    special_ids: Set[int] = set()
    for tid in (tokenizer.cls_token_id, tokenizer.sep_token_id,
                tokenizer.eos_token_id, tokenizer.bos_token_id):
        if tid is not None:
            special_ids.add(tid)

    encoder  = model.encoder
    layers   = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
    n_layers = len(layers)
    print(f"  [layer_search] DNABERT-2 has {n_layers} BertLayers")

    layer_vecs: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_layers)}

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size),
                          desc="dnabert2-all-layers", unit="batch"):
            batch  = sequences[start : start + batch_size]
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            tokens  = {k: v.to(device) for k, v in tokens.items()}
            B, S    = tokens["attention_mask"].shape
            captures: Dict[int, torch.Tensor] = {}

            hooks = []
            for li, layer in enumerate(layers):
                def _make_hook(idx):
                    def _h(_m, _inp, out):
                        raw_h = out[0] if isinstance(out, (tuple, list)) else out
                        if _is_unpadded_layer_output(raw_h, tokens["attention_mask"]):
                            flat = tokens["attention_mask"].bool().view(-1)
                            H    = raw_h.shape[-1]
                            buf  = torch.zeros(B, S, H, device=raw_h.device, dtype=raw_h.dtype)
                            buf.view(-1, H)[flat] = raw_h.detach()
                            captures[idx] = buf
                        else:
                            captures[idx] = raw_h.detach()
                    return _h
                hooks.append(layer.register_forward_hook(_make_hook(li)))
            try:
                model(**tokens)
            finally:
                for h in hooks:
                    h.remove()

            for li, hidden in captures.items():
                layer_vecs[li].append(
                    _mean_pool(hidden, tokens["attention_mask"],
                               tokens["input_ids"], special_ids)
                )

    return {i: np.concatenate(v, axis=0) for i, v in layer_vecs.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding cache
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, model_name: str, layer_idx: int, task_name: str) -> Path:
    safe = task_name.replace("/", "_")
    d    = cache_dir / model_name / f"layer_{layer_idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{safe}.npy"


def get_or_build_emb(
    cache_dir: Path,
    model_name: str,
    layer_idx: int,
    task_name: str,
    expected_n: Optional[int],
    build_fn,            # () -> np.ndarray
) -> np.ndarray:
    p = _cache_path(cache_dir, model_name, layer_idx, task_name)
    if p.exists():
        X = np.load(p)
        if expected_n is None or len(X) == expected_n:
            return X
        print(
            f"  [cache] Rebuilding stale cache for {model_name}/{task_name}: "
            f"n={len(X)} != expected {expected_n}"
        )
    X = build_fn()
    np.save(p, X)
    return X


def _cache_layer_index(layer_idx: Optional[int]) -> int:
    # FIX: normalize cache path for "last layer" to a stable non-negative index.
    return 999 if layer_idx is None else int(layer_idx)


def _cache_task_name(task_name: str, split: str) -> str:
    return f"{task_name}__{split}"


# ─────────────────────────────────────────────────────────────────────────────
# Layer search
# ─────────────────────────────────────────────────────────────────────────────

def _probe_layer_auroc(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> float:
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0, random_state=seed))
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y[te], prob))
    return float(np.mean(aurocs))


def run_layer_search(
    sequences: List[str],
    labels: np.ndarray,
    model_name: str,
    model,             # EsmForMaskedLM | AutoModel
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    cache_dir: Path,
    output_dir: Path,
    pilot_rbp: str,
    num_folds: int,
    seed: int,
    model_type: str,   # "lamar" | "dnabert2"
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Layer Search | model={model_name} | pilot={pilot_rbp} ===")

    # Check whether all-layer cache exists
    cached: Optional[Dict[int, np.ndarray]] = None
    layer_dirs = sorted((cache_dir / model_name).glob("layer_*")) \
        if (cache_dir / model_name).exists() else []
    if layer_dirs:
        tmp: Dict[int, np.ndarray] = {}
        ok = True
        expected_n = int(len(labels))
        for d in layer_dirs:
            li = int(d.name.replace("layer_", ""))
            p  = d / f"{_cache_task_name(pilot_rbp, 'pilot').replace('/', '_')}.npy"
            if not p.exists():
                ok = False
                break
            X_cached = np.load(p)
            if len(X_cached) != expected_n:
                # Stale cache from a different pilot sample size; force recompute.
                print(
                    f"  [cache] Ignoring stale layer {li} cache: "
                    f"n={len(X_cached)} != expected {expected_n}"
                )
                ok = False
                break
            tmp[li] = X_cached
        if ok:
            cached = tmp
            print(f"  [cache] Loaded {len(cached)} layers from cache")

    if cached is None:
        print("  [extract] All-layer extraction …")
        if model_type == "lamar":
            cached = extract_lamar_all_layers(
                sequences, model, tokenizer, device, max_length, batch_size)
        else:
            cached = extract_dnabert2_all_layers(
                sequences, model, tokenizer, device, max_length, batch_size)
        for li, X in cached.items():
            p = _cache_path(cache_dir, model_name, li, _cache_task_name(pilot_rbp, "pilot"))
            np.save(p, X)

    layer_aurocs: Dict[int, float] = {}
    for li, X in sorted(cached.items()):
        if np.linalg.norm(X, axis=1).mean() < 1e-6:
            print(f"    Layer {li:3d}: SKIPPED (zero embeddings)")
            continue
        auroc = _probe_layer_auroc(X, labels, n_splits=num_folds, seed=seed)
        layer_aurocs[li] = auroc
        print(f"    Layer {li:3d}: AUROC = {auroc:.4f}")

    best_layer = int(max(layer_aurocs, key=layer_aurocs.get))

    results = {
        "best_layer":   best_layer,
        "best_auroc":   layer_aurocs[best_layer],
        "pilot_rbp":    pilot_rbp,
        "model":        model_name,
        "layer_aurocs": {str(k): v for k, v in layer_aurocs.items()},
    }
    out_json = output_dir / f"layer_search_{model_name}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → Best layer {best_layer} (AUROC={layer_aurocs[best_layer]:.4f}) → {out_json}")
    return best_layer


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    # FIX: fit probe on train+dev and evaluate once on held-out test to preserve cross-length split.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "auroc_mean": float(roc_auc_score(y_test, prob)),
        "accuracy_mean": float(accuracy_score(y_test, pred)),
        "f1_mean": float(f1_score(y_test, pred)),
        "auprc_mean": float(average_precision_score(y_test, prob)),
        # FIX: single held-out split has no fold variance; mark std as N/A via NaN.
        "auroc_std": float("nan"),
        "accuracy_std": float("nan"),
        "f1_std": float("nan"),
        "auprc_std": float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, grp in df.groupby("model"):
        rows.append({
            "Model":     model_name,
            "Mean AUROC": grp["auroc"].mean(),
            "Std AUROC":  grp["auroc"].std(ddof=0),
            "Mean F1":    grp["f1"].mean(),
            "Std F1":     grp["f1"].std(ddof=0),
            "Mean AUPRC": grp["auprc"].mean(),
            "Std AUPRC":  grp["auprc"].std(ddof=0),
            "N tasks":    len(grp),
        })
    return (pd.DataFrame(rows)
            .sort_values("Mean AUROC", ascending=False)
            .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def statistical_tests(df: pd.DataFrame) -> Dict:
    pivot    = df.pivot(index="rbp", columns="model", values="auroc")
    all_cols = list(pivot.columns)
    if len(all_cols) < 2:
        return {}

    # Choose reference: prefer a pretrained model, else first column
    _ref_candidates = [
        "lamar_pretrained", "dnabert2_pretrained", "one_hot", all_cols[0]
    ]  # FIX: per_rbp_records store canonical model names, so refs resolve deterministically.
    ref = next((c for c in _ref_candidates if c in all_cols), all_cols[0])

    results: Dict = {}
    for other in all_cols:
        if other == ref:
            continue
        paired = pivot[[ref, other]].dropna()
        if len(paired) < 5:
            print(f"[WARN] Skipping Wilcoxon {other} vs {ref}: only {len(paired)} shared tasks")
            continue
        try:
            w   = wilcoxon(paired[other], paired[ref], alternative="two-sided")
            key = f"wilcoxon_{other}_vs_{ref}"
            results[key] = {
                "statistic":      float(w.statistic),
                "pvalue":         float(w.pvalue),
                "n_tasks":        int(len(paired)),
                "mean_delta_auroc": float((paired[other] - paired[ref]).mean()),
                "reference_model": ref,
            }
            print(f"  {key}: p={w.pvalue:.4f}, Δ={results[key]['mean_delta_auroc']:+.4f}")
        except Exception as exc:
            print(f"[WARN] Wilcoxon {other} vs {ref} failed: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    model_order = sorted(df["model"].unique())
    data = [df.loc[df["model"] == m, "auroc"].values for m in model_order]
    plt.figure(figsize=(max(10, len(model_order) * 1.4), 6))
    plt.boxplot(data, labels=model_order, showmeans=True)
    plt.ylabel("AUROC")
    plt.title("AUROC distribution across cross-length tasks")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_rbp_bars(df: pd.DataFrame, out_path: Path) -> None:
    pivot = df.pivot(index="rbp", columns="model", values="auroc")
    # Sort rows by first pretrained model or first column
    sort_col = next(
        (c for c in ["lamar_pretrained", "dnabert2_pretrained"] if c in pivot.columns),
        pivot.columns[0],
    )
    pivot = pivot.sort_values(sort_col)
    ax = pivot.plot(kind="bar", figsize=(max(16, len(pivot) * 0.35), 7), width=0.8)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-task AUROC by model — cross-length generalisation")
    ax.legend(loc="lower right", fontsize=7)
    plt.xticks(rotation=70, ha="right", fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_delta(df: pd.DataFrame, out_path: Path) -> None:
    """Delta AUROC: best LAMAR TAPT vs LAMAR pretrained (or first two available models)."""
    pivot = df.pivot(index="rbp", columns="model", values="auroc")
    cols  = list(pivot.columns)
    if len(cols) < 2:
        return

    # Prefer a meaningful comparison
    _pairs = [
        ("lamar_tapt_1024",        "lamar_pretrained"),
        ("lamar_tapt_standard_1gpu","lamar_pretrained"),
        ("dnabert2_tapt",          "dnabert2_pretrained"),
    ]
    ref, other = None, None
    for a, b in _pairs:
        if a in cols and b in cols:
            other, ref = a, b
            break
    if ref is None:
        ref, other = cols[0], cols[1]

    delta = (pivot[other] - pivot[ref]).dropna().sort_values()
    plt.figure(figsize=(max(12, len(delta) * 0.25), 5))
    plt.bar(delta.index, delta.values)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.ylabel(f"ΔAUROC ({other} − {ref})")
    plt.title("Per-task AUROC delta — cross-length generalisation")
    plt.xticks(rotation=70, ha="right", fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _fit_projection(X: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(X), "umap"
    except Exception:
        reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
        return reducer.fit_transform(X), "tsne"


def plot_embedding_projections(
    embeddings_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    task_names: Dict[int, str],
    out_dir: Path,
    seed: int,
    max_points: int,
) -> None:
    rng = np.random.default_rng(seed)
    for model_name, (X, task_idx) in embeddings_by_model.items():
        if len(X) > max_points:
            pick   = rng.choice(len(X), size=max_points, replace=False)
            X_plot = X[pick]
            y_plot = task_idx[pick]
        else:
            X_plot = X
            y_plot = task_idx

        Z, method = _fit_projection(X_plot, seed)

        plt.figure(figsize=(8, 6))
        uniq = np.unique(y_plot)
        for u in uniq:
            m = y_plot == u
            plt.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7, label=task_names.get(int(u), str(u)))
        plt.title(f"{model_name}: {method.upper()} — cross-length tasks")
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        if len(uniq) <= 15:
            plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{method}.png", dpi=200)
        plt.close()


def embedding_health_stats(
    embeddings_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    max_points_cosine: int,
    seed: int,
) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    rows = []
    for model_name, (X, _) in embeddings_by_model.items():
        norms = np.linalg.norm(X, axis=1)
        Xc    = X[rng.choice(len(X), size=max_points_cosine, replace=False)] \
                if len(X) > max_points_cosine else X
        C       = cosine_similarity(Xc)
        iu      = np.triu_indices_from(C, k=1)
        cos_vals= C[iu]
        rows.append({
            "model":       model_name,
            "norm_mean":   float(np.mean(norms)),
            "norm_std":    float(np.std(norms)),
            "cosine_mean": float(np.mean(cos_vals)),
            "cosine_std":  float(np.std(cos_vals)),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    cache_dir  = Path(args.cache_dir)
    plots_dir  = output_dir / "plots"
    for d in (output_dir, cache_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if torch.cuda.is_available():
        print(f"[GPU]    {torch.cuda.get_device_name(0)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    tasks = load_tasks(args.data_root)
    if not tasks:
        raise RuntimeError(f"No valid tasks found in {args.data_root}")

    tasks_items = sorted(tasks.items())
    if args.num_random_rbps > 0 and len(tasks_items) > args.num_random_rbps:
        task_names = [name for name, _ in tasks_items]
        rng = np.random.default_rng(args.seed)
        chosen = set(rng.choice(task_names, size=args.num_random_rbps, replace=False).tolist())
        tasks_items = [item for item in tasks_items if item[0] in chosen]
        tasks = dict(tasks_items)
        print(
            f"[data] Randomly sampled {len(tasks_items)} tasks out of {len(task_names)} "
            f"with seed={args.seed}"
        )

    print(f"[data] Using {len(tasks)} tasks")

    # ── Filter requested models + validate paths ───────────────────────────
    requested: List[str] = args.models
    active_models: Dict[str, dict] = {}
    for name in requested:
        if name == "one_hot":
            continue
        spec = _MODEL_DEFAULTS.get(name)
        if spec is None:
            print(f"[WARN] Unknown model name '{name}' — skipping")
            continue
        wpath = spec["weights_path"]
        if wpath != "__random__":
            is_local = wpath.startswith("/") or wpath.startswith(".")
            if is_local and not Path(wpath).exists():
                print(f"[WARN] Weights not found for '{name}': {wpath} — skipping")
                continue
        active_models[name] = spec

    use_one_hot = "one_hot" in requested
    if not active_models and not use_one_hot:
        raise RuntimeError("No valid models to evaluate.")

    # ── Load tokenizers (cached per unique tokenizer path) ────────────────
    tokenizer_cache: Dict[str, object] = {}

    def _get_tokenizer(spec: dict):
        tok_path = spec["tokenizer"]
        if tok_path not in tokenizer_cache:
            extra = {}
            if spec["type"] == "dnabert2":
                extra = {"use_fast": True, "trust_remote_code": True, "padding_side": "right"}
            tokenizer_cache[tok_path] = AutoTokenizer.from_pretrained(
                tok_path,
                model_max_length=spec["max_length"],
                **extra,
            )
        return tokenizer_cache[tok_path]

    # ── Layer search / override ────────────────────────────────────────────
    # FIX: use validated intermediate-layer defaults unless a global override or
    #      per-model layer search is explicitly requested.
    best_layer_per_model: Dict[str, Optional[int]] = {
        m: _DEFAULT_LAYER_BY_TYPE[spec["type"]] for m, spec in active_models.items()
    }

    if active_models:
        print(f"[INFO] default probing layers: {_DEFAULT_LAYER_BY_TYPE}")

    if args.best_layer_override is not None:
        for m in best_layer_per_model:
            best_layer_per_model[m] = args.best_layer_override
        print(f"[INFO] best_layer_override={args.best_layer_override} applied to all models")

    elif args.enable_layer_search:
        pilot_rbp = args.layer_search_pilot_rbp or tasks_items[0][0]
        if pilot_rbp not in tasks:
            print(f"[WARN] Pilot RBP '{pilot_rbp}' not found; using '{tasks_items[0][0]}'")
            pilot_rbp = tasks_items[0][0]

        pilot_df = tasks[pilot_rbp]["train_dev"]  # FIX: layer-search uses train+dev only.
        if args.max_pilot_samples > 0 and len(pilot_df) > args.max_pilot_samples:
            pilot_df = pilot_df.sample(n=args.max_pilot_samples, random_state=args.seed)
        pilot_seqs   = pilot_df["sequence"].tolist()
        pilot_labels = pilot_df["label"].to_numpy(dtype=np.int64)
        print(f"[layer_search] Pilot task='{pilot_rbp}' ({len(pilot_seqs)} samples)")

        for model_name, spec in active_models.items():
            tokenizer  = _get_tokenizer(spec)
            is_random  = spec["weights_path"] == "__random__"
            mtype      = spec["type"]
            max_length = spec["max_length"]

            print(f"\n  Loading {model_name} for layer search …")
            if mtype == "lamar":
                model = (build_lamar_model_random(tokenizer, device, seed=args.seed)
                         if is_random
                         else build_lamar_model(tokenizer, spec["weights_path"], device))
            else:
                pad_id = tokenizer.pad_token_id or 0
                model = (build_dnabert2_model_random(
                             spec["weights_path"], pad_id, device, seed=args.seed)
                         if is_random
                         else build_dnabert2_model(spec["weights_path"], pad_id, device))

            best = run_layer_search(
                sequences=pilot_seqs,
                labels=pilot_labels,
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_length=max_length,
                batch_size=args.batch_size,
                # FIX: share cache root with main probing so pilot embeddings are reused.
                cache_dir=cache_dir,
                output_dir=output_dir / "layer_search",
                pilot_rbp=pilot_rbp,
                num_folds=args.num_folds,
                seed=args.seed,
                model_type=mtype,
            )
            best_layer_per_model[model_name] = best
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Main probing loop (one model at a time to save GPU memory) ─────────
    per_rbp_records: List[dict] = []
    embedding_pool:  Dict[str, List[np.ndarray]] = {}
    task_pool:       Dict[str, List[np.ndarray]] = {}
    task_to_int      = {name: i for i, name in enumerate(sorted(tasks.keys()))}

    def _fmt_main_metric(mean_val: float, std_val: float) -> str:
        # FIX: avoid misleading ±0.0000 in single-split evaluation where std is undefined.
        return f"{mean_val:.4f}" if np.isnan(std_val) else f"{mean_val:.4f}±{std_val:.4f}"

    # --- one-hot baseline ---
    if use_one_hot:
        MAX_LEN_ONEHOT = 512  # fixed for all sequences
        print(f"\n{'='*60}\nModel: one_hot (max_length={MAX_LEN_ONEHOT})\n{'='*60}")
        for task_name, splits in sorted(tasks.items()):
            df_train = splits["train_dev"]
            df_test  = splits["test"]
            if args.max_samples_per_rbp > 0 and len(df_train) > args.max_samples_per_rbp:
                df_train = df_train.sample(n=args.max_samples_per_rbp, random_state=args.seed)
            if args.max_samples_per_rbp > 0 and len(df_test) > args.max_samples_per_rbp:
                df_test = df_test.sample(n=args.max_samples_per_rbp, random_state=args.seed)
            sequences_train = df_train["sequence"].tolist()
            labels_train    = df_train["label"].to_numpy(dtype=np.int64)
            sequences_test  = df_test["sequence"].tolist()
            labels_test     = df_test["label"].to_numpy(dtype=np.int64)

            def _build_oh_train(seqs=sequences_train):
                return one_hot_embed(seqs, MAX_LEN_ONEHOT)

            def _build_oh_test(seqs=sequences_test):
                return one_hot_embed(seqs, MAX_LEN_ONEHOT)

            # FIX: canonical train-dev cache key keeps caching convention consistent.
            X_train = get_or_build_emb(
                cache_dir,
                "one_hot",
                _cache_layer_index(None),
                _cache_task_name(task_name, "train_dev"),
                len(labels_train),
                _build_oh_train,
            )
            X_test  = get_or_build_emb(
                cache_dir,
                "one_hot",
                _cache_layer_index(None),
                _cache_task_name(task_name, "test"),
                len(labels_test),
                _build_oh_test,
            )
            m = evaluate_linear_probe(X_train, labels_train, X_test, labels_test, seed=args.seed)
            per_rbp_records.append({
                "rbp":      task_name,
                "model":    "one_hot",
                "layer":    None,
                "auroc":    m["auroc_mean"],
                "auroc_std": m["auroc_std"],
                "accuracy": m["accuracy_mean"],
                "f1":       m["f1_mean"],
                "auprc":    m["auprc_mean"],
            })
            print(f"  {task_name:55s}  auroc={_fmt_main_metric(m['auroc_mean'], m['auroc_std'])}"
                  f"  f1={m['f1_mean']:.4f}  auprc={m['auprc_mean']:.4f}")
            embedding_pool.setdefault("one_hot", []).append(X_test)
            task_pool.setdefault("one_hot", []).append(
                np.full(len(X_test), task_to_int[task_name], dtype=np.int32))

    # --- transformer models ---
    for model_name, spec in active_models.items():
        layer_idx  = best_layer_per_model.get(model_name, None)
        # FIX: use canonical model_name key so per_rbp tables and embedding pools stay aligned.
        cache_label = model_name
        is_random   = spec["weights_path"] == "__random__"
        mtype       = spec["type"]
        max_length  = spec["max_length"]
        tokenizer   = _get_tokenizer(spec)

        print(f"\n{'='*60}")
        print(f"Model: {model_name}  |  type={mtype}  |  max_length={max_length}"
              f"  |  layer={layer_idx if layer_idx is not None else 'last'}")
        print(f"Weights: {spec['weights_path']}")
        print(f"{'='*60}")

        if mtype == "lamar":
            model = (build_lamar_model_random(tokenizer, device, seed=args.seed)
                     if is_random
                     else build_lamar_model(tokenizer, spec["weights_path"], device))
        else:
            pad_id = tokenizer.pad_token_id or 0
            model  = (build_dnabert2_model_random(
                          spec["weights_path"], pad_id, device, seed=args.seed)
                      if is_random
                      else build_dnabert2_model(spec["weights_path"], pad_id, device))

        for task_name, splits in sorted(tasks.items()):
            df_train = splits["train_dev"]
            df_test  = splits["test"]
            if args.max_samples_per_rbp > 0 and len(df_train) > args.max_samples_per_rbp:
                df_train = df_train.sample(n=args.max_samples_per_rbp, random_state=args.seed)
            if args.max_samples_per_rbp > 0 and len(df_test) > args.max_samples_per_rbp:
                df_test = df_test.sample(n=args.max_samples_per_rbp, random_state=args.seed)

            sequences_train = df_train["sequence"].tolist()
            labels_train    = df_train["label"].to_numpy(dtype=np.int64)
            sequences_test  = df_test["sequence"].tolist()
            labels_test     = df_test["label"].to_numpy(dtype=np.int64)
            print(f"  [task] {task_name}: n_train_dev={len(df_train)} n_test={len(df_test)}")

            if mtype == "lamar":
                def _build_train(
                    seqs=sequences_train,
                    li=layer_idx,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=max_length,
                    batch_size=args.batch_size,
                    model_name=model_name,
                    task_name=task_name,
                ):
                    # FIX: bind closure variables at definition-time to avoid late-binding bugs.
                    return extract_lamar_embeddings(
                        seqs, model, tokenizer, device, max_length,
                        batch_size, li, desc=f"{model_name}/{task_name[:30]}/train")

                def _build_test(
                    seqs=sequences_test,
                    li=layer_idx,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=max_length,
                    batch_size=args.batch_size,
                    model_name=model_name,
                    task_name=task_name,
                ):
                    # FIX: bind closure variables at definition-time to avoid late-binding bugs.
                    return extract_lamar_embeddings(
                        seqs, model, tokenizer, device, max_length,
                        batch_size, li, desc=f"{model_name}/{task_name[:30]}/test")
            else:
                def _build_train(
                    seqs=sequences_train,
                    li=layer_idx,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=max_length,
                    batch_size=args.batch_size,
                    model_name=model_name,
                    task_name=task_name,
                ):
                    # FIX: bind closure variables at definition-time to avoid late-binding bugs.
                    return extract_dnabert2_embeddings(
                        seqs, model, tokenizer, device, max_length,
                        batch_size, li, desc=f"{model_name}/{task_name[:30]}/train")

                def _build_test(
                    seqs=sequences_test,
                    li=layer_idx,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_length=max_length,
                    batch_size=args.batch_size,
                    model_name=model_name,
                    task_name=task_name,
                ):
                    # FIX: bind closure variables at definition-time to avoid late-binding bugs.
                    return extract_dnabert2_embeddings(
                        seqs, model, tokenizer, device, max_length,
                        batch_size, li, desc=f"{model_name}/{task_name[:30]}/test")

            cache_li = _cache_layer_index(layer_idx)
            X_train = get_or_build_emb(
                cache_dir, cache_label,
                cache_li,
                _cache_task_name(task_name, "train_dev"),
                len(labels_train),
                _build_train,
            )
            X_test = get_or_build_emb(
                cache_dir, cache_label,
                cache_li,
                _cache_task_name(task_name, "test"),
                len(labels_test),
                _build_test,
            )

            m = evaluate_linear_probe(X_train, labels_train, X_test, labels_test, seed=args.seed)
            per_rbp_records.append({
                "rbp":       task_name,
                "model":     model_name,  # FIX: keep canonical names so summary/tests resolve intended references.
                "layer":     layer_idx,
                "auroc":     m["auroc_mean"],
                "auroc_std": m["auroc_std"],
                "accuracy":  m["accuracy_mean"],
                "f1":        m["f1_mean"],
                "auprc":     m["auprc_mean"],
            })
            print(f"    auroc={_fmt_main_metric(m['auroc_mean'], m['auroc_std'])}"
                  f"  f1={m['f1_mean']:.4f}  auprc={m['auprc_mean']:.4f}")

            embedding_pool.setdefault(cache_label, []).append(X_test)
            task_pool.setdefault(cache_label, []).append(
                np.full(len(X_test), task_to_int[task_name], dtype=np.int32))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save tabular results ──────────────────────────────────────────────
    per_rbp_df = pd.DataFrame(per_rbp_records)
    per_rbp_df.to_csv(output_dir / "per_rbp_metrics.csv", index=False)

    summary_df = summarize(per_rbp_df)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    print("\n=== Summary (mean ± std across tasks) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── Statistical tests ─────────────────────────────────────────────────
    stat_results = statistical_tests(per_rbp_df)
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)

    # ── Aggregate embeddings for visualisation ────────────────────────────
    embeddings_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for mname in embedding_pool:
        X_all = np.concatenate(embedding_pool[mname], axis=0)
        y_all = np.concatenate(task_pool[mname], axis=0)
        embeddings_by_model[mname] = (X_all, y_all)

    health_df = embedding_health_stats(
        embeddings_by_model, args.max_points_cosine, args.seed)
    health_df.to_csv(output_dir / "embedding_health_stats.csv", index=False)
    print("\n[health stats]\n" + health_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.skip_plots:
        plot_boxplot(per_rbp_df,    plots_dir / "auroc_boxplot.png")
        plot_per_rbp_bars(per_rbp_df, plots_dir / "per_rbp_auroc_bars.png")
        plot_delta(per_rbp_df,      plots_dir / "delta_auroc.png")
        plot_embedding_projections(
            embeddings_by_model=embeddings_by_model,
            task_names={v: k for k, v in task_to_int.items()},
            out_dir=plots_dir,
            seed=args.seed,
            max_points=args.max_points_vis,
        )
        print(f"[plots] Saved to: {plots_dir}")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n[done] Results saved to: {output_dir}")
    if any(v is not None for v in best_layer_per_model.values()):
        print(f"[layers] {best_layer_per_model}")


if __name__ == "__main__":
    main()
