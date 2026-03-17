"""
Combined DNABERT-2 + CNN evaluation for all DNABERT-2 variants.

Evaluated variants
------------------
  dnabert2_pretrained  – zhihan1996/DNABERT-2-117M
  dnabert2_tapt        – dnabert2_standard_mlm/checkpoint-25652
  dnabert2_tapt_v3     – dnabert2_tapt_v3/checkpoint-2566
  dnabert2_random      – randomly initialised DNABERT-2 baseline

Protocol per (model_variant, RBP)
---------------------------------
  1. Run layer search on that RBP's train+dev split
  2. Extract per-token embeddings from the best layer
  3. Train the same 1-D CNN with the same hyperparameters used in LAMAR_CNN.py
  4. Evaluate on the held-out test split

Results are saved incrementally to:
  --output_dir / DNABERT2_CNN_results.csv
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil
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


# TF log level before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_DISABLE_MKL"] = "0"
_configure_xla_cuda_data_dir()

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from tensorflow import keras
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from layer_search import _call_with_all_layers

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
    print("  WARNING: no CUDA – DNABERT-2 embedding will be slow")
print(f"DNABERT-2 device: {device}")

# ── defaults ──────────────────────────────────────────────────────────────────
_BASE = "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis"
_DB2_BASE = f"{_BASE}/DNABERT2"

_DATA_ROOTS = [
    f"{_BASE}/DNABERT2/data",
    f"{_BASE}/data/finetune_data_koo",
]

_MODEL_VARIANTS = {
    "dnabert2_pretrained": {
        "weights_path": "zhihan1996/DNABERT-2-117M",
        "tokenizer": "zhihan1996/DNABERT-2-117M",
        "max_length": 512,
        "hidden_dim": 768,
    },
    "dnabert2_tapt": {
        "weights_path": f"{_DB2_BASE}/pretrain/models/dnabert2_standard_mlm/checkpoint-25652",
        "tokenizer": "zhihan1996/DNABERT-2-117M",
        "max_length": 512,
        "hidden_dim": 768,
    },
    "dnabert2_tapt_v3": {
        "weights_path": f"{_DB2_BASE}/pretrain/models/dnabert2_tapt_v3/checkpoint-2566",
        "tokenizer": "zhihan1996/DNABERT-2-117M",
        "max_length": 512,
        "hidden_dim": 768,
    },
    "dnabert2_random": {
        "weights_path": "__random__",
        "tokenizer": "zhihan1996/DNABERT-2-117M",
        "max_length": 512,
        "hidden_dim": 768,
    },
}

_OUTPUT_DIR = f"{_BASE}/DNABERT2/evalEmbeddings/results/cnn_all_variants"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DNABERT-2 variants + per-RBP layer search + CNN")
    p.add_argument("--data_roots", nargs="+", default=_DATA_ROOTS)
    p.add_argument("--output_dir", default=_OUTPUT_DIR)
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(_MODEL_VARIANTS.keys()),
        choices=list(_MODEL_VARIANTS.keys()),
        help="Which DNABERT-2 variants to run.",
    )
    p.add_argument("--embed_batch_size", type=int, default=8)
    p.add_argument("--downstream_batch", type=int, default=256)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--layer_search_folds", type=int, default=2)
    p.add_argument("--layer_search_probe_samples", type=int, default=512)
    p.add_argument("--layer_search_probe_epochs", type=int, default=5)
    p.add_argument("--layer_search_probe_batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--best_layer",
        type=int,
        default=None,
        help="Skip per-RBP layer search and force this layer index for all variants/tasks.",
    )
    return p.parse_args()


# ── helpers: data loading ─────────────────────────────────────────────────────
def find_column(columns, candidates):
    lmap = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lmap:
            return lmap[cand]
    raise ValueError(f"None of {candidates} found in {columns}")


def load_splits(rbp_dir: str):
    """Return dict split→(seqs, labels) from train/dev/test CSVs, or None."""
    fname_map = {"train": "train.csv", "valid": "dev.csv", "test": "test.csv"}
    splits = {}
    for split, fname in fname_map.items():
        path = os.path.join(rbp_dir, fname)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        seq_col = find_column(list(df.columns), ["sequence", "seq", "text", "input"])
        label_col = find_column(list(df.columns), ["label", "labels", "target", "y"])
        splits[split] = (
            df[seq_col].astype(str).tolist(),
            df[label_col].to_numpy(dtype="int64"),
        )
    return splits


def discover_tasks(data_roots):
    """Return list of (task_id, rbp_name, source_tag, rbp_dir)."""
    tasks = []
    seen = set()
    for root in data_roots:
        if not os.path.isdir(root):
            print(f"[WARN] Data root not found, skipping: {root}")
            continue
        source_tag = os.path.basename(root)
        for rbp_dir in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(rbp_dir):
                continue
            rbp_name = os.path.basename(rbp_dir)
            task_id = f"{source_tag}/{rbp_name}"
            if task_id in seen:
                continue
            if not os.path.exists(os.path.join(rbp_dir, "train.csv")):
                continue
            seen.add(task_id)
            tasks.append((task_id, rbp_name, source_tag, rbp_dir))
    return tasks


def completed_pairs(results_csv):
    if not os.path.exists(results_csv):
        return set()
    try:
        df = pd.read_csv(results_csv)
        return set(zip(df["model_variant"], df["task_id"]))
    except Exception:
        return set()


# ── DNABERT-2 model helpers ──────────────────────────────────────────────────
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


def _materialize_meta_tensors(model: torch.nn.Module) -> torch.nn.Module:
    for name, mod in model.named_modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "meta":
                mod.register_buffer(bname, torch.zeros(buf.shape, dtype=buf.dtype))
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


def _load_state_dict_from_path(weights_path: str) -> dict:
    p = Path(weights_path)
    if p.is_dir():
        st = p / "model.safetensors"
        if st.exists():
            from safetensors.torch import load_file
            return load_file(str(st))
        pb = p / "pytorch_model.bin"
        if pb.exists():
            state = torch.load(str(pb), map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                return state.get("state_dict", state.get("model", state))
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {weights_path}")
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(p))
    state = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        return state.get("state_dict", state.get("model", state))
    return state


def _dnabert2_checkpoint_needs_remote_fallback(weights_path: str) -> bool:
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
    required_files = (
        "configuration_bert.py",
        "bert_layers.py",
        "bert_padding.py",
        "flash_attn_triton.py",
    )
    missing_files = [name for name in required_files if not (p / name).exists()]

    if uses_local_custom_code and missing_files:
        print(
            f"  [dnabert2] local checkpoint is missing custom code files {missing_files}; "
            f"falling back to remote DNABERT-2 code for {weights_path}"
        )
        return True
    return False


def _load_dnabert2_config(weights_path: str, fallback_model: str, pad_token_id: int):
    if _dnabert2_checkpoint_needs_remote_fallback(weights_path):
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

    if getattr(config, "pad_token_id", None) is None:
        object.__setattr__(config, "pad_token_id", int(pad_token_id))
    return config


def load_dnabert2_variant(spec: dict, seed: int):
    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer"],
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
        model_max_length=spec["max_length"],
    )

    if spec["weights_path"] == "__random__":
        config = _load_dnabert2_config(
            weights_path=spec["tokenizer"],
            fallback_model=spec["tokenizer"],
            pad_token_id=int(tokenizer.pad_token_id or 0),
        )
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = AutoModel.from_config(config, trust_remote_code=True)
        model.apply(_wolf_init)
        print(f"  [random] Wolf-initialised DNABERT-2, seed={seed}")
    else:
        config = _load_dnabert2_config(
            weights_path=spec["weights_path"],
            fallback_model=spec["tokenizer"],
            pad_token_id=int(tokenizer.pad_token_id or 0),
        )
        if _dnabert2_checkpoint_needs_remote_fallback(spec["weights_path"]):
            model = AutoModel.from_config(config, trust_remote_code=True)
            state_dict = _load_state_dict_from_path(spec["weights_path"])
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            non_trivial_missing = [k for k in missing if "position_ids" not in k]
            if non_trivial_missing:
                print(f"  [WARN] Missing keys after fallback load: {non_trivial_missing[:5]}")
            if unexpected:
                print(f"  [WARN] Unexpected keys after fallback load: {unexpected[:5]}")
        else:
            model = AutoModel.from_pretrained(
                spec["weights_path"],
                trust_remote_code=True,
                config=config,
                _fast_init=False,
            )

    model = _materialize_meta_tensors(model)
    model.to(device)
    model.eval()
    return model, tokenizer


# ── layer search ──────────────────────────────────────────────────────────────
def _mean_pool(hidden, attention_mask, input_ids, special_ids):
    valid = attention_mask.bool()
    if special_ids:
        special_mask = torch.zeros_like(valid)
        for sid in special_ids:
            special_mask |= input_ids.eq(sid)
        valid = valid & ~special_mask
    denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (hidden * valid.unsqueeze(-1)).sum(dim=1) / denom


def _subsample_layer_search_data(seqs, labels, max_samples, seed):
    if max_samples is None or max_samples <= 0 or len(seqs) <= max_samples:
        return seqs, labels

    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_samples, random_state=seed)
    selected_idx, _ = next(splitter.split(np.zeros(len(labels)), labels))
    selected_idx = np.sort(selected_idx)
    return [seqs[i] for i in selected_idx], labels[selected_idx]


def _extract_all_layer_tokens(seqs, model, tokenizer, model_max_length, embed_batch_size):
    layer_vecs: Optional[Dict[int, List[np.ndarray]]] = None

    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), embed_batch_size), desc="  layer-search", unit="batch"):
            batch = seqs[i:i + embed_batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=model_max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            all_hidden = _call_with_all_layers(model, dict(enc))

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

    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
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
    variant_name,
    task_id,
    model_max_length,
    embed_batch_size,
    cache_dir,
    num_folds,
    seed,
    probe_max_samples,
    probe_epochs,
    probe_batch_size,
):
    variant_cache_dir = Path(cache_dir) / variant_name
    variant_cache_dir.mkdir(parents=True, exist_ok=True)
    safe_task = task_id.replace("/", "__")
    result_json = variant_cache_dir / f"{safe_task}.json"

    if result_json.exists():
        with open(result_json) as f:
            cached = json.load(f)
        print(
            f"  [layer_search] cached {variant_name} / {task_id}: "
            f"layer {cached['best_layer']} (AUROC={cached['best_auroc']:.4f})"
        )
        return int(cached["best_layer"])

    print(f"  [layer_search] {variant_name} / {task_id}")
    probe_seqs, probe_labels = _subsample_layer_search_data(
        seqs=seqs,
        labels=labels,
        max_samples=probe_max_samples,
        seed=seed,
    )
    if len(probe_seqs) != len(seqs):
        print(f"    Using stratified probe subset: {len(probe_seqs)} / {len(seqs)} sequences")

    layer_embeddings = _extract_all_layer_tokens(
        seqs=probe_seqs,
        model=model,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        embed_batch_size=embed_batch_size,
    )

    layer_aurocs = {}
    for li, X in sorted(layer_embeddings.items()):
        if np.linalg.norm(X.reshape(X.shape[0], -1), axis=1).mean() < 1e-6:
            print(f"    Layer {li:2d}: SKIPPED (zero norm)")
            continue
        auroc = _probe_layer(
            X,
            probe_labels,
            n_splits=num_folds,
            seed=seed,
            epochs=probe_epochs,
            batch_size=probe_batch_size,
        )
        layer_aurocs[li] = auroc
        print(f"    Layer {li:2d}: AUROC = {auroc:.4f}")

    if not layer_aurocs:
        raise RuntimeError(f"No valid layers found during layer search for {variant_name} / {task_id}")

    best_layer = int(max(layer_aurocs, key=lambda li: layer_aurocs[li]))
    best_auroc = float(layer_aurocs[best_layer])
    with open(result_json, "w") as f:
        json.dump(
            {
                "task_id": task_id,
                "model_variant": variant_name,
                "best_layer": best_layer,
                "best_auroc": best_auroc,
                "layer_aurocs": {str(k): v for k, v in layer_aurocs.items()},
            },
            f,
            indent=2,
        )
    return best_layer


# ── embedding extraction ──────────────────────────────────────────────────────
def extract_embeddings(seqs, model, tokenizer, model_max_length,
                       embed_batch_size, layer_idx, save_path, split_label=""):
    """
    Extract per-token embeddings and write directly to a memmap .npy file.
    Returns the memmap array (read-only) — never holds full data in RAM.
    """
    hidden_dim = int(getattr(model.config, "hidden_size"))
    n_seqs = len(seqs)

    emb_path = Path(save_path)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(
        str(emb_path), mode="w+",
        dtype="float32", shape=(n_seqs, model_max_length, hidden_dim),
    )

    with torch.no_grad():
        for i in tqdm(
            range(0, n_seqs, embed_batch_size),
            desc=f"  embed {split_label}",
            unit="batch",
        ):
            batch = seqs[i:i + embed_batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=model_max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = _call_with_all_layers(model, dict(enc))[layer_idx]
            mmap[i:i + len(batch)] = hidden.detach().cpu().numpy().astype("float32")

    mmap.flush()
    # Re-open read-only so OS can page it in on demand
    return np.load(str(emb_path), mmap_mode="r")


# ── CNN architecture ──────────────────────────────────────────────────────────
def chip_cnn(input_shape, output_shape):
    initializer = tf.keras.initializers.HeUniform(seed=42)
    inp = keras.Input(shape=input_shape)

    nn = keras.layers.BatchNormalization()(inp)
    nn = keras.layers.Conv1D(filters=512, kernel_size=1,
                             kernel_initializer=initializer)(nn)

    nn = keras.layers.Conv1D(filters=64, kernel_size=7, padding="same",
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Conv1D(filters=96, kernel_size=5, padding="same",
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same",
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)

    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation("relu")(nn)
    nn = keras.layers.Dropout(0.5)(nn)

    logits = keras.layers.Dense(output_shape)(nn)
    output = keras.layers.Activation("sigmoid")(logits)
    return keras.Model(inputs=inp, outputs=output)


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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    results_csv = output_dir / "DNABERT2_CNN_results.csv"
    cache_dir = output_dir / "cache"
    layer_search_dir = cache_dir / "layer_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    layer_search_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(args.data_roots)
    done = completed_pairs(str(results_csv))

    print(f"Tasks total  : {len(tasks)}")
    print(f"Already done : {len(done)}")

    if not tasks:
        raise RuntimeError("No valid tasks found. Check --data_roots.")

    for variant_name in args.variants:
        spec = _MODEL_VARIANTS[variant_name]
        pending = [t for t in tasks if (variant_name, t[0]) not in done]

        print(f"\n{'=' * 70}")
        print(f"Variant      : {variant_name}")
        print(f"Weights      : {spec['weights_path']}")
        print(f"Max length   : {spec['max_length']}")
        print(f"Tasks pending: {len(pending)} / {len(tasks)}")
        print(f"{'=' * 70}")

        if not pending:
            print("Nothing to do for this variant.")
            continue

        print(f"\n[Phase 1] Extracting embeddings for {variant_name}")
        model, tokenizer = load_dnabert2_variant(spec, seed=args.seed)
        task_best_layers: Dict[str, int] = {}

        for task_id, rbp_name, source_tag, rbp_dir in pending:
            print(f"\n--- {variant_name} | {task_id} ---")

            splits = load_splits(rbp_dir)
            if splits is None:
                print("  [SKIP] missing CSV files")
                continue

            if args.best_layer is not None:
                best_layer = args.best_layer
                print(f"  [layer_search] forced layer {best_layer}")
            else:
                layer_search_seqs = splits["train"][0] + splits["valid"][0]
                layer_search_labels = np.concatenate([splits["train"][1], splits["valid"][1]])
                best_layer = run_layer_search_for_task(
                    seqs=layer_search_seqs,
                    labels=layer_search_labels,
                    model=model,
                    tokenizer=tokenizer,
                    variant_name=variant_name,
                    task_id=task_id,
                    model_max_length=spec["max_length"],
                    embed_batch_size=args.embed_batch_size,
                    cache_dir=layer_search_dir,
                    num_folds=args.layer_search_folds,
                    seed=args.seed,
                    probe_max_samples=args.layer_search_probe_samples,
                    probe_epochs=args.layer_search_probe_epochs,
                    probe_batch_size=args.layer_search_probe_batch,
                )

            print(f"  Best layer : {best_layer}")
            task_best_layers[task_id] = best_layer

            emb_dir = cache_dir / variant_name / task_id.replace("/", "__")
            emb_dir.mkdir(parents=True, exist_ok=True)

            for split_name in ("train", "valid", "test"):
                seqs, _ = splits[split_name]
                emb_path = emb_dir / f"{split_name}.npy"

                if emb_path.exists():
                    print(f"  {split_name:>5}: {len(seqs)} sequences (cached)")
                else:
                    print(f"  {split_name:>5}: {len(seqs)} sequences")
                    extract_embeddings(
                        seqs=seqs,
                        model=model,
                        tokenizer=tokenizer,
                        model_max_length=spec["max_length"],
                        embed_batch_size=args.embed_batch_size,
                        layer_idx=best_layer,
                        save_path=emb_path,
                        split_label=f"{variant_name}/{rbp_name}/{split_name}",
                    )

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n[Phase 1] Done for {variant_name}. PyTorch model released.")
        print(f"\n[Phase 2] Training CNNs for {variant_name}")

        hidden_dim = int(spec["hidden_dim"])
        emb_shape = (spec["max_length"], hidden_dim)

        for task_id, rbp_name, source_tag, rbp_dir in pending:
            print(f"\n--- {variant_name} | {task_id} ---")

            splits = load_splits(rbp_dir)
            if splits is None:
                print("  [SKIP] missing CSV files")
                continue

            emb_dir = cache_dir / variant_name / task_id.replace("/", "__")
            if not emb_dir.exists():
                print(f"  [SKIP] missing embedding cache: {emb_dir}")
                continue

            best_layer = task_best_layers.get(task_id)
            if best_layer is None:
                print(f"  [SKIP] missing cached best layer for {task_id}")
                continue

            print(f"  Best layer : {best_layer}")
            print(f"  Embedding shape : {emb_shape}")

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

            # Force one eager fetch to fail-fast on any tf.data / mmap issue.
            try:
                warmup_x, warmup_y = next(iter(dataset_tf["train"]))
                print(f"  train batch warmup: X={tuple(warmup_x.shape)}, y={tuple(warmup_y.shape)}")
            except Exception as exc:
                raise RuntimeError(f"Failed to fetch first training batch for {variant_name}/{task_id}") from exc

            acc_list, auroc_list, aupr_list = [], [], []
            for rep in range(args.n_repeats):
                auroc_metric = tf.keras.metrics.AUC(curve="ROC", name="auroc")
                aupr_metric = tf.keras.metrics.AUC(curve="PR", name="aupr")

                cnn = chip_cnn(emb_shape, 1)
                cnn.compile(
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metrics=["accuracy", auroc_metric, aupr_metric],
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    jit_compile=False,
                    run_eagerly=True,
                )
                print(
                    f"  [train] {variant_name} | {task_id} | "
                    f"repeat {rep + 1}/{args.n_repeats} | starting fit"
                )
                cnn.fit(
                    dataset_tf["train"],
                    validation_data=dataset_tf["valid"],
                    epochs=args.max_epochs,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
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
                print(
                    f"    rep {rep + 1}/{args.n_repeats}: "
                    f"acc={acc:.4f}  auroc={roc:.4f}  aupr={pr:.4f}"
                )
                tf.keras.backend.clear_session()

            rec: Dict[str, Any] = {
                "model_variant": variant_name,
                "task_id": task_id,
                "rbp": rbp_name,
                "source": source_tag,
                "best_layer": best_layer,
                "max_length": spec["max_length"],
                "embed_batch_size": args.embed_batch_size,
                "downstream_batch": args.downstream_batch,
                "n_repeats": args.n_repeats,
                "max_epochs": args.max_epochs,
                "seed": args.seed,
                "acc_mean": float(np.mean(acc_list)),
                "acc_std": float(np.std(acc_list)),
                "auroc_mean": float(np.mean(auroc_list)),
                "auroc_std": float(np.std(auroc_list)),
                "aupr_mean": float(np.mean(aupr_list)),
                "aupr_std": float(np.std(aupr_list)),
            }

            existing: List[Dict[Any, Any]] = []
            if results_csv.exists():
                try:
                    existing = pd.read_csv(results_csv).to_dict("records")
                except Exception:
                    existing = []
            pd.DataFrame(existing + [rec]).to_csv(results_csv, index=False)

            print(
                f"  → AUROC {rec['auroc_mean']:.4f}±{rec['auroc_std']:.4f}  "
                f"AUPR {rec['aupr_mean']:.4f}±{rec['aupr_std']:.4f}"
            )
            print(f"  Results saved → {results_csv}")

            dataset_tf.clear()
            if emb_dir.exists():
                shutil.rmtree(emb_dir)
                print(f"  [cleanup] removed {emb_dir}")
            gc.collect()
            tf.keras.backend.clear_session()

    if results_csv.exists():
        df = pd.read_csv(results_csv).drop_duplicates(subset=["model_variant", "task_id"], keep="last")
        print("\n" + "=" * 70)
        print("FINAL SUMMARY (mean per model variant)")
        print("=" * 70)
        print(df.groupby("model_variant")[["auroc_mean", "aupr_mean", "acc_mean"]].mean().round(4).to_string())
        print("\nPer variant × source:")
        print(df.groupby(["model_variant", "source"])[["auroc_mean", "aupr_mean", "acc_mean"]].mean().round(4).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
