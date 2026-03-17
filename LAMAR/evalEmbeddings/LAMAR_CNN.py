"""
LAMAR_CNN.py
------------
LAMAR (all canonical variants) + per-RBP layer search + CNN evaluation.

Evaluated variants:
    lamar_pretrained
    lamar_tapt_1024
    lamar_tapt_standard_1gpu
    lamar_tapt_custom_1gpu
    lamar_tapt_512
    lamar_tapt_512_std
    lamar_random

For each (variant, RBP):
    1. Run layer search on that RBP's train+dev split
    2. Extract per-token hidden states from the best layer
    3. Train the same 1-D CNN with the same hyperparameters used in DNABERT2_CNN.py
    4. Evaluate on the held-out test split

Only the LAMAR input length differs by variant (512 or 1024).
Results are saved incrementally to --output_dir/LAMAR_CNN_results.csv.
"""

from __future__ import annotations

import argparse
import glob
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _configure_xla_cuda_data_dir() -> None:
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir=" in xla_flags:
        return

    candidate_roots: List[str] = []
    for env_var in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_DIR"):
        value = os.environ.get(env_var)
        if value:
            candidate_roots.append(value)

    candidate_roots.extend(["/usr/local/cuda", "/opt/cuda"])
    candidate_roots.extend(sorted(glob.glob("/usr/local/cuda-*"), reverse=True))

    seen = set()
    for root in candidate_roots:
        if not root or root in seen:
            continue
        seen.add(root)
        libdevice = os.path.join(root, "nvvm", "libdevice", "libdevice.10.bc")
        if os.path.exists(libdevice):
            append_flag = f"--xla_gpu_cuda_data_dir={root}"
            os.environ["XLA_FLAGS"] = (
                f"{xla_flags} {append_flag}".strip() if xla_flags else append_flag
            )
            return


# TF log level must be set before the import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_DISABLE_MKL"] = "0"
_configure_xla_cuda_data_dir()

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from safetensors.torch import load_file
from tensorflow import keras
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

# ── LAMAR package path ────────────────────────────────────────────────────────
# The LAMAR package lives at Thesis/LAMAR/LAMAR/ and is imported as `LAMAR.*`.
# Insert the parent of that package directory so Python can resolve the import
# from any working directory.
_LAMAR_PKG_ROOT = Path(__file__).resolve().parent.parent   # …/Thesis/LAMAR/
if str(_LAMAR_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAMAR_PKG_ROOT))

from LAMAR.modeling_nucESM2 import EsmForMaskedLM  # noqa: E402

# ── device setup ──────────────────────────────────────────────────────────────
tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options(
    {
        "disable_meta_optimizer": False,
        "auto_mixed_precision": False,
        "constant_folding": True,
        "arithmetic_optimization": True,
        "dependency_optimization": True,
        "layout_optimizer": False,
        "remapping": False,
        "auto_parallel": False,
    }
)
tf_gpus = tf.config.list_physical_devices("GPU")
for gpu in tf_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"TensorFlow  : {tf.__version__}  (GPUs visible: {len(tf_gpus)})")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA (torch): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
else:
    print("  WARNING: no CUDA – embedding extraction will be slow")
print(f"PyTorch device: {device}")

# ── default paths ─────────────────────────────────────────────────────────────
_BASE = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"

_DATA_ROOTS = [
    f"{_BASE}/DNABERT2/data",
    f"{_BASE}/data/finetune_data_koo",
]

_TOKENIZER_PATH = (
    f"{_BASE}/LAMAR/src/pretrain/saving_model"
    "/tapt_1024_standard_collator/checkpoint-134000"
)

_MODEL_VARIANTS = {
    "lamar_pretrained": {
        "weights_path": f"{_BASE}/LAMAR/weights",
        "max_length": 1024,
        "hidden_dim": 768,
    },
    "lamar_tapt_1024": {
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length": 1024,
        "hidden_dim": 768,
    },
    "lamar_tapt_standard_1gpu": {
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator_1gpu/checkpoint-232000",
        "max_length": 1024,
        "hidden_dim": 768,
    },
    "lamar_tapt_custom_1gpu": {
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_custom_collator_1gpu/checkpoint-232000",
        "max_length": 1024,
        "hidden_dim": 768,
    },
    "lamar_tapt_512": {
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_lamar/checkpoint-98000",
        "max_length": 512,
        "hidden_dim": 768,
    },
    "lamar_tapt_512_std": {
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_512_standard_collator_1gpu/checkpoint-265000",
        "max_length": 512,
        "hidden_dim": 768,
    },
    "lamar_random": {
        "weights_path": None,
        "max_length": 1024,
        "hidden_dim": 768,
    },
}

_OUTPUT_DIR = f"{_BASE}/LAMAR/evalEmbeddings/results/cnn_all_variants"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LAMAR variants – per-RBP layer search + CNN")
    p.add_argument("--data_roots",       nargs="+", default=_DATA_ROOTS)
    p.add_argument("--output_dir",       default=_OUTPUT_DIR)
    p.add_argument("--tokenizer_path",   default=_TOKENIZER_PATH)
    p.add_argument("--embed_batch_size", type=int, default=8)
    p.add_argument("--downstream_batch", type=int, default=256)
    p.add_argument("--n_repeats",        type=int, default=5)
    p.add_argument("--max_epochs",       type=int, default=100)
    p.add_argument("--layer_search_folds", type=int, default=5)
    p.add_argument("--layer_search_probe_samples", type=int, default=512)
    p.add_argument("--layer_search_probe_epochs", type=int, default=5)
    p.add_argument("--layer_search_probe_batch", type=int, default=32)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(_MODEL_VARIANTS.keys()),
        choices=list(_MODEL_VARIANTS.keys()),
        help="Which model variants to run.",
    )
    p.add_argument(
        "--best_layer",
        type=int,
        default=None,
        help="Skip layer search and force this layer index for all variants.",
    )
    return p.parse_args()


# ── data helpers ──────────────────────────────────────────────────────────────
def _find_col(columns, candidates):
    lmap = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lmap:
            return lmap[cand]
    raise ValueError(f"None of {candidates} found in {columns}")


def load_splits(rbp_dir: str):
    """Return {split: (seqs, labels)} from train/dev/test CSVs, or None."""
    fname_map = {"train": "train.csv", "valid": "dev.csv", "test": "test.csv"}
    splits = {}
    for split, fname in fname_map.items():
        path = os.path.join(rbp_dir, fname)
        if not os.path.exists(path):
            return None
        df        = pd.read_csv(path)
        seq_col   = _find_col(list(df.columns), ["sequence", "seq", "text", "input"])
        label_col = _find_col(list(df.columns), ["label", "labels", "target", "y"])
        splits[split] = (
            df[seq_col].astype(str).tolist(),
            df[label_col].to_numpy(dtype="int64"),
        )
    return splits


def discover_tasks(data_roots):
    """List of (task_id, rbp_name, source_tag, rbp_dir) for all valid RBPs."""
    tasks, seen = [], set()
    for root in data_roots:
        if not os.path.isdir(root):
            print(f"[WARN] Data root not found, skipping: {root}")
            continue
        source_tag = os.path.basename(root)
        for rbp_dir in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(rbp_dir):
                continue
            if not os.path.exists(os.path.join(rbp_dir, "train.csv")):
                continue
            rbp_name = os.path.basename(rbp_dir)
            task_id  = f"{source_tag}/{rbp_name}"
            if task_id in seen:
                continue
            seen.add(task_id)
            tasks.append((task_id, rbp_name, source_tag, rbp_dir))
    return tasks


def completed_pairs(results_csv):
    """Return set of (model_variant, task_id) strings already in the results CSV."""
    if not os.path.exists(results_csv):
        return set()
    try:
        df = pd.read_csv(results_csv)
        return set(zip(df["model_variant"], df["task_id"]))
    except Exception:
        return set()


# ── LAMAR model loading ───────────────────────────────────────────────────────
def _build_lamar_config(tokenizer):
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


def _remap_weights(raw: dict) -> dict:
    """Remap state-dict keys to match EsmForMaskedLM layout."""
    out = {}
    for k, v in raw.items():
        if k.startswith("esm.lm_head"):
            out[k[len("esm."):]] = v          # esm.lm_head.* → lm_head.*
        elif k.startswith("lm_head") or k.startswith("esm."):
            out[k] = v
        else:
            out["esm." + k] = v               # bare keys → prepend esm.
    return out


def _load_weights_from(weights_path: str) -> dict:
    """Load safetensors from a file or from a checkpoint directory."""
    p = Path(weights_path)
    if p.is_dir():
        st = p / "model.safetensors"
        if st.exists():
            return load_file(str(st))
        pb = p / "pytorch_model.bin"
        if pb.exists():
            return torch.load(str(pb), map_location="cpu")
        raise FileNotFoundError(f"No weights file in {weights_path}")
    return load_file(str(p))


def _wolf_init(module: torch.nn.Module) -> None:
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


def build_lamar_model(tokenizer, weights_path, is_random: bool, seed: int) -> EsmForMaskedLM:
    """Build, load weights, and return a ready-to-eval LAMAR model."""
    config = _build_lamar_config(tokenizer)
    model  = EsmForMaskedLM(config)

    if not is_random:
        raw     = _load_weights_from(weights_path)
        remapped = _remap_weights(raw)
        res     = model.load_state_dict(remapped, strict=False)
        non_trivial_missing = [k for k in res.missing_keys if "lm_head" not in k]
        if non_trivial_missing:
            print(f"  [WARN] Missing (non-lm_head) keys: {non_trivial_missing[:5]}")
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model.apply(_wolf_init)
        print(f"  [INFO] Random initialisation (Wolf), seed={seed}")

    model.to(device)
    model.eval()
    return model


# ── layer utilities ───────────────────────────────────────────────────────────
def _get_all_hidden(model: EsmForMaskedLM, tokens: dict):
    """Return list of (B, seq_len, H) tensors for all hidden states."""
    out = model.esm(**tokens, output_hidden_states=True)
    return list(out.hidden_states)   # index 0 = embeddings, 1..N = transformer blocks


def mean_pool(hidden, attention_mask, input_ids, special_ids):
    """Mean-pool hidden states, excluding padding and special tokens."""
    valid = attention_mask.bool()
    if special_ids:
        smask = torch.zeros_like(valid)
        for sid in special_ids:
            smask |= input_ids.eq(sid)
        valid = valid & ~smask
    denom  = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (hidden * valid.unsqueeze(-1)).sum(dim=1) / denom


# ── layer search ──────────────────────────────────────────────────────────────
def _subsample_layer_search_data(seqs, labels, max_samples, seed):
    if max_samples is None or max_samples <= 0 or len(seqs) <= max_samples:
        return seqs, labels

    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    selected_idx, _ = next(splitter.split(np.zeros(len(labels)), labels))
    selected_idx = np.sort(selected_idx)
    return [seqs[i] for i in selected_idx], labels[selected_idx]


def _extract_all_layer_tokens(seqs, model, tokenizer, max_length, batch_size, variant_name):
    layer_vecs: Optional[Dict[int, List[np.ndarray]]] = None

    with torch.no_grad():
        for start in tqdm(
            range(0, len(seqs), batch_size),
            desc=f"  layer-search {variant_name}",
            unit="batch",
        ):
            batch = seqs[start : start + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            all_hidden = _get_all_hidden(model, tokens)

            if layer_vecs is None:
                layer_vecs = {li: [] for li in range(len(all_hidden))}

            for li, hidden in enumerate(all_hidden):
                layer_vecs[li].append(hidden.detach().cpu().numpy().astype("float32"))

    assert layer_vecs is not None
    return {li: np.concatenate(chunks, axis=0) for li, chunks in layer_vecs.items()}


def _build_probe_cnn(seq_len, hidden_dim):
    class ProbeCNN(torch.nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, padding=3)
            self.relu = torch.nn.ReLU()
            self.fc = torch.nn.Linear(32, 1)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.relu(self.conv(x))
            x = x.max(dim=2).values
            return self.fc(x).squeeze(-1)

    return ProbeCNN(hidden_dim)


def _probe_layer(X, y, n_splits=5, seed=42, epochs=5, batch_size=32):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    rng = np.random.default_rng(seed)
    X = X.astype("float32", copy=False)
    y = y.astype("float32", copy=False)

    for tr, te in skf.split(X, y):
        probe = _build_probe_cnn(X.shape[1], X.shape[2]).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        Xtr = torch.from_numpy(X[tr]).to(device)
        ytr = torch.from_numpy(y[tr]).to(device)
        Xte = torch.from_numpy(X[te]).to(device)
        yte = y[te]

        best_val_loss = float("inf")
        best_state = None
        stale_epochs = 0

        for _epoch in range(epochs):
            probe.train()
            perm = rng.permutation(Xtr.size(0))
            for start in range(0, Xtr.size(0), batch_size):
                idx = perm[start:start + batch_size]
                xb = Xtr[idx]
                yb = ytr[idx]

                optimizer.zero_grad(set_to_none=True)
                logits = probe(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            probe.eval()
            with torch.no_grad():
                val_logits = probe(Xte)
                val_loss = float(criterion(val_logits, torch.from_numpy(yte).to(device)).item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= 2:
                    break

        if best_state is not None:
            probe.load_state_dict(best_state)

        probe.eval()
        with torch.no_grad():
            preds = torch.sigmoid(probe(Xte)).detach().cpu().numpy().ravel()
        aurocs.append(float(roc_auc_score(yte, preds)))

        del probe, optimizer, Xtr, ytr, Xte
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(np.mean(aurocs))


def run_layer_search_for_task(
    seqs,
    labels,
    model,
    tokenizer,
    max_length,
    batch_size,
    variant_name,
    task_id,
    cache_dir,
    seed,
    n_splits=5,
    probe_max_samples=512,
    probe_epochs=5,
    probe_batch_size=32,
):
    """
    Extract mean-pool embeddings for every layer on a pilot RBP, probe each one,
    return the index with the highest AUROC.  Results are cached.
    """
    cache_dir = Path(cache_dir) / variant_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_json = cache_dir / f"{task_id.replace('/', '__')}.json"

    if result_json.exists():
        with open(result_json) as f:
            cached = json.load(f)
        print(f"  [layer_search] Loaded cached result for {variant_name} / {task_id}: "
              f"best layer = {cached['best_layer']} (AUROC={cached['best_auroc']:.4f})")
        return cached["best_layer"]

    print(f"  [layer_search] {variant_name} / {task_id}")
    probe_seqs, probe_labels = _subsample_layer_search_data(
        seqs=seqs,
        labels=labels,
        max_samples=probe_max_samples,
        seed=seed,
    )
    if len(probe_seqs) != len(seqs):
        print(f"    Using stratified probe subset: {len(probe_seqs)} / {len(seqs)} sequences")

    layer_embs = _extract_all_layer_tokens(
        seqs=probe_seqs,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        variant_name=variant_name,
    )

    # Probe each layer
    layer_aurocs = {}
    for li, X in sorted(layer_embs.items()):
        if np.linalg.norm(X.reshape(X.shape[0], -1), axis=1).mean() < 1e-6:
            print(f"    Layer {li:2d}: SKIPPED (zero norm)")
            continue
        auroc = _probe_layer(
            X,
            probe_labels,
            n_splits=n_splits,
            seed=seed,
            epochs=probe_epochs,
            batch_size=probe_batch_size,
        )
        layer_aurocs[li] = auroc
        print(f"    Layer {li:2d}: AUROC = {auroc:.4f}")

    if not layer_aurocs:
        raise RuntimeError(f"No valid layers found during layer search for {variant_name} / {task_id}")

    best_layer = int(max(layer_aurocs, key=lambda li: layer_aurocs[li]))
    best_auroc = layer_aurocs[best_layer]

    with open(result_json, "w") as f:
        json.dump({
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "variant":    variant_name,
            "task_id":     task_id,
            "layer_aurocs": {str(k): v for k, v in layer_aurocs.items()},
        }, f, indent=2)

    print(f"  [layer_search] {variant_name} / {task_id}: best layer = {best_layer} "
          f"(AUROC={best_auroc:.4f})")
    return best_layer


# ── embedding extraction ──────────────────────────────────────────────────────
def extract_embeddings(seqs, model, tokenizer, max_length, batch_size,
                       layer_idx, save_path, split_label=""):
    """
    Extract per-token embeddings and write directly to a memmap .npy file.
    Returns the memmap array (read-only) — never holds full data in RAM.
    """
    hidden_dim = int(getattr(model.config, "hidden_size", 768))
    n_seqs = len(seqs)

    emb_path = Path(save_path)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(
        str(emb_path), mode="w+",
        dtype="float32", shape=(n_seqs, max_length, hidden_dim),
    )

    with torch.no_grad():
        for start in tqdm(range(0, len(seqs), batch_size),
                          desc=f"  embed {split_label}", unit="batch"):
            batch  = seqs[start : start + batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=max_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hidden = _get_all_hidden(model, tokens)[layer_idx]   # (B, seq_len, H)
            mmap[start:start + len(batch)] = hidden.detach().cpu().numpy().astype("float32")

    mmap.flush()
    return np.load(str(emb_path), mmap_mode="r")


def make_dataset(emb_path, labels, batch_size, training, seed):
    mmap = np.load(str(emb_path), mmap_mode="r")
    y = labels.astype("int64")
    n = len(y)
    idx = np.arange(n, dtype="int64")
    x_shape = tuple(int(dim) for dim in mmap.shape[1:])

    ds = tf.data.Dataset.from_tensor_slices((idx, y))
    if training:
        ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    def _fetch_batch(batch_idx, batch_y):
        def _np_fetch(i):
            return mmap[i].astype("float32", copy=False)

        batch_x = tf.numpy_function(_np_fetch, [batch_idx], tf.float32)
        batch_x.set_shape((None,) + x_shape)
        batch_y = tf.cast(batch_y, tf.int64)
        return batch_x, batch_y

    ds = ds.map(_fetch_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── CNN architecture ──────────────────────────────────────────────────────────
def chip_cnn(input_shape, output_shape):
    init = tf.keras.initializers.HeUniform(seed=42)
    inp  = keras.Input(shape=input_shape)

    nn = keras.layers.BatchNormalization()(inp)
    nn = keras.layers.Conv1D(512, 1, kernel_initializer=init)(nn)

    for filters, kernel, pool in [(64, 7, 4), (96, 5, 4), (128, 5, 2)]:
        nn = keras.layers.Conv1D(filters, kernel, padding="same", kernel_initializer=init)(nn)
        nn = keras.layers.BatchNormalization()(nn)
        nn = keras.layers.Activation("relu")(nn)
        nn = keras.layers.MaxPooling1D(pool)(nn)
        nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    out = keras.layers.Activation("sigmoid")(keras.layers.Dense(output_shape)(nn))
    return keras.Model(inputs=inp, outputs=out)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    output_dir   = args.output_dir
    results_csv  = os.path.join(output_dir, "LAMAR_CNN_results.csv")
    cache_dir    = Path(output_dir) / "cache"
    ls_cache_dir = cache_dir / "layer_search"
    os.makedirs(output_dir, exist_ok=True)
    ls_cache_dir.mkdir(parents=True, exist_ok=True)

    # Discover all RBP tasks
    tasks = discover_tasks(args.data_roots)
    if not tasks:
        raise RuntimeError("No valid RBP tasks found. Check --data_roots.")
    print(f"Tasks discovered : {len(tasks)}")

    # Already-completed (variant, task_id) pairs
    done = completed_pairs(results_csv)
    print(f"Already done     : {len(done)}\n")

    # Load tokenizer once (shared across all variants)
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=max(spec["max_length"] for spec in _MODEL_VARIANTS.values()),
    )
    print(f"  vocab={len(tokenizer)}, "
          f"pad={tokenizer.pad_token_id}, mask={tokenizer.mask_token_id}\n")

    # ── iterate over model variants ───────────────────────────────────────────
    for variant_name in args.variants:
        spec = _MODEL_VARIANTS[variant_name]
        weights_path = spec["weights_path"]
        max_length = spec["max_length"]
        is_random    = (weights_path is None)

        # Skip entire variant if all tasks are already done
        pending = [(tid, rb, src, rd) for tid, rb, src, rd in tasks
                   if (variant_name, tid) not in done]

        print(f"\n{'='*70}")
        print(f"Variant : {variant_name}")
        if not is_random:
            print(f"Weights : {weights_path}")
        else:
            print(f"Weights : (random initialisation)")
        print(f"Max len : {max_length}")
        print(f"Tasks pending: {len(pending)} / {len(tasks)}")
        print(f"{'='*70}")

        if not pending:
            print("  All tasks already done for this variant – skipping.")
            continue

        print(f"\n[Phase 1] Extracting embeddings for {variant_name}")
        print(f"\n  Loading LAMAR model ({variant_name}) …")
        model = build_lamar_model(tokenizer, weights_path, is_random, seed=args.seed)
        task_best_layers: Dict[str, int] = {}

        # ── Phase 1: layer search + embedding extraction ─────────────────────
        for task_id, rbp_name, source_tag, rbp_dir in pending:
            print(f"\n  --- {task_id} ---")

            splits = load_splits(rbp_dir)
            if splits is None:
                print("    [SKIP] missing CSV files")
                continue

            if args.best_layer is not None:
                best_layer = args.best_layer
                print(f"    [layer_search] forced layer {best_layer}")
            else:
                layer_search_seqs = splits["train"][0] + splits["valid"][0]
                layer_search_labels = np.concatenate([
                    splits["train"][1],
                    splits["valid"][1],
                ])
                best_layer = run_layer_search_for_task(
                    seqs=layer_search_seqs,
                    labels=layer_search_labels,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    batch_size=args.embed_batch_size,
                    variant_name=variant_name,
                    task_id=task_id,
                    cache_dir=ls_cache_dir,
                    seed=args.seed,
                    n_splits=args.layer_search_folds,
                    probe_max_samples=args.layer_search_probe_samples,
                    probe_epochs=args.layer_search_probe_epochs,
                    probe_batch_size=args.layer_search_probe_batch,
                )
            print(f"    Best layer: {best_layer}")
            task_best_layers[task_id] = best_layer

            emb_dir = cache_dir / variant_name / task_id.replace("/", "__")
            emb_dir.mkdir(parents=True, exist_ok=True)

            for split_name in ("train", "valid", "test"):
                seqs, _ = splits[split_name]
                emb_path = emb_dir / f"{split_name}.npy"

                if emb_path.exists():
                    print(f"    {split_name:>5}: {len(seqs)} sequences (cached)")
                else:
                    print(f"    {split_name:>5}: {len(seqs)} sequences")
                    extract_embeddings(
                        seqs=seqs,
                        model=model,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        batch_size=args.embed_batch_size,
                        layer_idx=best_layer,
                        save_path=emb_path,
                        split_label=f"{variant_name}/{rbp_name}/{split_name}",
                    )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"\n[Phase 1] Done for {variant_name}. PyTorch model released.")
        print(f"\n[Phase 2] Training CNNs for {variant_name}")

        hidden_dim = int(spec["hidden_dim"])
        emb_shape = (max_length, hidden_dim)

        for task_id, rbp_name, source_tag, rbp_dir in pending:
            print(f"\n  --- {task_id} ---")

            splits = load_splits(rbp_dir)
            if splits is None:
                print("    [SKIP] missing CSV files")
                continue

            emb_dir = cache_dir / variant_name / task_id.replace("/", "__")
            if not emb_dir.exists():
                print(f"    [SKIP] missing embedding cache: {emb_dir}")
                continue

            best_layer = task_best_layers.get(task_id)
            if best_layer is None:
                print(f"    [SKIP] missing cached best layer for {task_id}")
                continue

            print(f"    Best layer: {best_layer}")
            print(f"    Embedding shape: {emb_shape}")

            dataset_tf = {}
            for split_name in ("train", "valid", "test"):
                _, labels = splits[split_name]
                emb_path = emb_dir / f"{split_name}.npy"
                if not emb_path.exists():
                    raise FileNotFoundError(f"Missing cached embeddings: {emb_path}")

                dataset_tf[split_name] = make_dataset(
                    emb_path=emb_path,
                    labels=labels,
                    batch_size=args.downstream_batch,
                    training=(split_name == "train"),
                    seed=args.seed,
                )

            # Train CNN (n_repeats)
            acc_list, auroc_list, aupr_list = [], [], []

            for rep in range(args.n_repeats):
                cnn = chip_cnn(emb_shape, 1)
                cnn.compile(
                    loss      = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metrics   = [
                        "accuracy",
                        tf.keras.metrics.AUC(curve="ROC", name="auroc"),
                        tf.keras.metrics.AUC(curve="PR",  name="aupr"),
                    ],
                    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
                    jit_compile=False,
                    run_eagerly=True,
                )
                cnn.fit(
                    dataset_tf["train"],
                    validation_data = dataset_tf["valid"],
                    epochs          = args.max_epochs,
                    verbose         = 0,
                    callbacks       = [
                        tf.keras.callbacks.EarlyStopping(
                            patience=10, restore_best_weights=True
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
                        ),
                    ],
                )

                metrics = np.atleast_1d(cnn.evaluate(dataset_tf["test"], verbose=0)).tolist()
                _, acc, roc, pr = metrics
                acc_list.append(acc)
                auroc_list.append(roc)
                aupr_list.append(pr)
                print(f"    rep {rep+1}/{args.n_repeats}: "
                      f"acc={acc:.4f}  auroc={roc:.4f}  aupr={pr:.4f}")
                tf.keras.backend.clear_session()

            rec: Dict[str, Any] = {
                "model_variant": variant_name,
                "task_id":       task_id,
                "rbp":           rbp_name,
                "source":        source_tag,
                "best_layer":    best_layer,
                "max_length":    max_length,
                "embed_batch_size": args.embed_batch_size,
                "downstream_batch": args.downstream_batch,
                "n_repeats":       args.n_repeats,
                "max_epochs":      args.max_epochs,
                "seed":            args.seed,
                "acc_mean":      float(np.mean(acc_list)),
                "acc_std":       float(np.std(acc_list)),
                "auroc_mean":    float(np.mean(auroc_list)),
                "auroc_std":     float(np.std(auroc_list)),
                "aupr_mean":     float(np.mean(aupr_list)),
                "aupr_std":      float(np.std(aupr_list)),
            }
            print(f"    → AUROC {rec['auroc_mean']:.4f}±{rec['auroc_std']:.4f}  "
                  f"AUPR {rec['aupr_mean']:.4f}±{rec['aupr_std']:.4f}")

            # Incremental save after each RBP
            existing: List[Dict[Any, Any]] = []
            if os.path.exists(results_csv):
                try:
                    existing = pd.read_csv(results_csv).to_dict("records")
                except Exception:
                    existing = []
            pd.DataFrame(existing + [rec]).to_csv(results_csv, index=False)

            dataset_tf.clear()
            if emb_dir.exists():
                shutil.rmtree(emb_dir)
                print(f"    [cleanup] removed {emb_dir}")
            gc.collect()
            tf.keras.backend.clear_session()

    # ── final summary ─────────────────────────────────────────────────────────
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv).drop_duplicates(subset=["model_variant", "task_id"], keep="last")
        print("\n" + "=" * 70)
        print("FINAL SUMMARY (mean AUROC per model variant)")
        print("=" * 70)
        print(df.groupby("model_variant")[["auroc_mean", "aupr_mean", "acc_mean"]]
              .mean().round(4).to_string())
        print("\nPer variant × source:")
        print(df.groupby(["model_variant", "source"])[["auroc_mean", "aupr_mean"]]
              .mean().round(4).to_string())
    print(f"\nResults saved to: {results_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
