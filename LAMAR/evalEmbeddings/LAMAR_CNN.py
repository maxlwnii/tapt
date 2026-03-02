"""
LAMAR_CNN.py
------------
LAMAR (all variants) + layer search + CNN evaluation.

Evaluates four LAMAR model variants on every RBP task found under --data_roots:
  pretrained   – base LAMAR weights (LAMAR/weights safetensors)
  tapt_1024    – TAPT checkpoint-131000 (tapt_1024_standard_collator)
  tapt_lamar   – TAPT checkpoint-98000  (tapt_lamar)
  random       – random initialisation (same architecture, no pretraining)

For each variant:
  Step 1 – Layer search  : probes all transformer layers via mean-pool +
                           logistic regression on a pilot RBP.  Picks the layer
                           with the highest AUROC.  Result is cached.
  Step 2 – CNN training  : extracts per-token hidden states (seq_len, 768) from
                           the best layer, then trains the 1-D CNN (N_REPEATS
                           per RBP) and records AUROC / AUPR / Accuracy.

Results are saved incrementally to --output_dir/LAMAR_CNN_results.csv.
Already-completed (model_variant, task_id) pairs are skipped on re-runs.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# TF log level must be set before the import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
tf.config.set_visible_devices([], "GPU")   # TF → CPU only
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"TensorFlow  : {tf.__version__}  (CPU only)")
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
    "/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/data",
    "/home/fr/fr_fr/fr_ml642/Thesis/data/finetune_data_koo",
]

_TOKENIZER_PATH = (
    f"{_BASE}/LAMAR/src/pretrain/saving_model"
    "/tapt_1024_standard_collator/checkpoint-134000"
)

_MODEL_VARIANTS = {
    "pretrained": f"{_BASE}/LAMAR/weights",
    "tapt_1024":  f"{_BASE}/LAMAR/src/pretrain/saving_model"
                  "/tapt_1024_standard_collator/checkpoint-134000",
    "tapt_lamar": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                  "/tapt_lamar/checkpoint-98000",
    "random":     None,   # random init – no weights file
}

_OUTPUT_DIR = f"{_BASE}/LAMAR/evalEmbeddings/results/LAMAR_CNN"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LAMAR variants – layer search + CNN")
    p.add_argument("--data_roots",       nargs="+", default=_DATA_ROOTS)
    p.add_argument("--output_dir",       default=_OUTPUT_DIR)
    p.add_argument("--tokenizer_path",   default=_TOKENIZER_PATH)
    p.add_argument("--model_max_length", type=int, default=512)
    p.add_argument("--embed_batch_size", type=int, default=32)
    p.add_argument("--downstream_batch", type=int, default=256)
    p.add_argument("--n_repeats",        type=int, default=5)
    p.add_argument("--max_epochs",       type=int, default=100)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(_MODEL_VARIANTS.keys()),
        choices=list(_MODEL_VARIANTS.keys()),
        help="Which model variants to run (default: all four).",
    )
    p.add_argument(
        "--best_layer",
        type=int,
        default=None,
        help="Skip layer search and force this layer index for all variants.",
    )
    p.add_argument(
        "--pilot_rbp",
        type=str,
        default=None,
        help="RBP name to use as layer-search pilot (default: first alphabetically).",
    )
    p.add_argument(
        "--max_pilot_samples",
        type=int,
        default=4000,
        help="Cap pilot samples used for layer search.",
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
            df[label_col].to_numpy(dtype=np.int64),
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


def build_lamar_model(tokenizer, weights_path, is_random: bool) -> EsmForMaskedLM:
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
        print("  [INFO] Random initialisation – no weights loaded")

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
def _probe_layer(X, y, n_splits=5, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
        )
        clf.fit(X[tr], y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aurocs))


def run_layer_search(
    seqs, labels, model, tokenizer, max_length, batch_size,
    variant_name, cache_dir, seed, n_splits=5
):
    """
    Extract mean-pool embeddings for every layer on a pilot RBP, probe each one,
    return the index with the highest AUROC.  Results are cached.
    """
    import json

    cache_dir = Path(cache_dir) / variant_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    special_ids = tokenizer.all_special_ids
    result_json = cache_dir.parent / f"layer_search_{variant_name}.json"

    # Load from JSON cache if available
    if result_json.exists():
        with open(result_json) as f:
            cached = json.load(f)
        print(f"  [layer_search] Loaded cached result for {variant_name}: "
              f"best layer = {cached['best_layer']} (AUROC={cached['best_auroc']:.4f})")
        return cached["best_layer"]

    print(f"  [layer_search] Extracting all layers for {variant_name} …")

    # Discover layer count
    with torch.no_grad():
        dummy = tokenizer(seqs[:2], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length)
        dummy = {k: v.to(device) for k, v in dummy.items()}
        n_layers = len(_get_all_hidden(model, dummy))
    print(f"  [layer_search] {n_layers} hidden states "
          f"(1 embedding + {n_layers - 1} transformer blocks)")

    # Accumulate per-layer mean-pool vectors
    layer_vecs = {i: [] for i in range(n_layers)}
    with torch.no_grad():
        for start in tqdm(range(0, len(seqs), batch_size),
                          desc=f"  layer-search {variant_name}", unit="batch"):
            batch  = seqs[start : start + batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            for li, h in enumerate(_get_all_hidden(model, tokens)):
                p = mean_pool(h, tokens["attention_mask"], tokens["input_ids"], special_ids)
                layer_vecs[li].append(p.cpu().numpy().astype(np.float32))

    layer_embs = {li: np.concatenate(vecs) for li, vecs in layer_vecs.items()}

    # Probe each layer
    layer_aurocs = {}
    for li, X in sorted(layer_embs.items()):
        if np.linalg.norm(X, axis=1).mean() < 1e-6:
            print(f"    Layer {li:2d}: SKIPPED (zero norm)")
            continue
        auroc = _probe_layer(X, labels, n_splits=n_splits, seed=seed)
        layer_aurocs[li] = auroc
        print(f"    Layer {li:2d}: AUROC = {auroc:.4f}")

    best_layer = max(layer_aurocs, key=layer_aurocs.get)
    best_auroc = layer_aurocs[best_layer]

    with open(result_json, "w") as f:
        json.dump({
            "best_layer": best_layer,
            "best_auroc": best_auroc,
            "variant":    variant_name,
            "layer_aurocs": {str(k): v for k, v in layer_aurocs.items()},
        }, f, indent=2)

    print(f"  [layer_search] {variant_name}: best layer = {best_layer} "
          f"(AUROC={best_auroc:.4f})")
    return best_layer


# ── embedding extraction ──────────────────────────────────────────────────────
def extract_embeddings(seqs, model, tokenizer, max_length, batch_size,
                       layer_idx, split_label=""):
    """Return list of (seq_len, 768) numpy arrays, one per sequence."""
    special_ids = tokenizer.all_special_ids
    all_embs    = []
    with torch.no_grad():
        for start in tqdm(range(0, len(seqs), batch_size),
                          desc=f"  embed {split_label}", unit="batch"):
            batch  = seqs[start : start + batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding="max_length",
                               truncation=True, max_length=max_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hidden = _get_all_hidden(model, tokens)[layer_idx]   # (B, seq_len, H)
            all_embs.extend(hidden.cpu().numpy())
    return all_embs


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
        args.tokenizer_path, model_max_length=args.model_max_length
    )
    print(f"  vocab={len(tokenizer)}, "
          f"pad={tokenizer.pad_token_id}, mask={tokenizer.mask_token_id}\n")

    # Select pilot RBP for layer search
    if args.pilot_rbp:
        pilot_candidates = [t for t in tasks if t[1] == args.pilot_rbp]
        pilot_task = pilot_candidates[0] if pilot_candidates else tasks[0]
        if not pilot_candidates:
            print(f"[WARN] --pilot_rbp '{args.pilot_rbp}' not found; using first task.")
    else:
        pilot_task = tasks[0]

    _, pilot_rbp_name, _, pilot_dir = pilot_task
    pilot_splits  = load_splits(pilot_dir)
    pilot_seqs    = pilot_splits["train"][0] + pilot_splits["valid"][0]
    pilot_labels  = np.concatenate([pilot_splits["train"][1], pilot_splits["valid"][1]])

    if args.max_pilot_samples and len(pilot_seqs) > args.max_pilot_samples:
        rng          = np.random.default_rng(args.seed)
        idx          = rng.choice(len(pilot_seqs), args.max_pilot_samples, replace=False)
        pilot_seqs   = [pilot_seqs[i] for i in idx]
        pilot_labels = pilot_labels[idx]

    print(f"Pilot RBP: '{pilot_rbp_name}'  ({len(pilot_seqs)} samples)\n")

    # ── iterate over model variants ───────────────────────────────────────────
    for variant_name in args.variants:
        weights_path = _MODEL_VARIANTS[variant_name]
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
        print(f"Tasks pending: {len(pending)} / {len(tasks)}")
        print(f"{'='*70}")

        if not pending:
            print("  All tasks already done for this variant – skipping.")
            continue

        # Load model for this variant
        print(f"\n  Loading LAMAR model ({variant_name}) …")
        model = build_lamar_model(tokenizer, weights_path, is_random)

        # ── layer search ──────────────────────────────────────────────────────
        if args.best_layer is not None:
            best_layer = args.best_layer
            print(f"  [INFO] Forced best_layer={best_layer} (--best_layer flag)")
        else:
            best_layer = run_layer_search(
                seqs        = pilot_seqs,
                labels      = pilot_labels,
                model       = model,
                tokenizer   = tokenizer,
                max_length  = args.model_max_length,
                batch_size  = args.embed_batch_size,
                variant_name = variant_name,
                cache_dir   = ls_cache_dir,
                seed        = args.seed,
            )

        print(f"\n  Best layer for {variant_name}: {best_layer}\n")

        # ── CNN training per RBP ──────────────────────────────────────────────
        records = []

        for task_id, rbp_name, source_tag, rbp_dir in pending:
            print(f"\n  --- {task_id}  [layer {best_layer}] ---")

            splits = load_splits(rbp_dir)
            if splits is None:
                print("    [SKIP] missing CSV files")
                continue

            # Extract per-token embeddings
            dataset_tf = {}
            emb_shape  = None

            for split_name in ("train", "valid", "test"):
                seqs, labels = splits[split_name]
                embs = extract_embeddings(
                    seqs, model, tokenizer,
                    max_length  = args.model_max_length,
                    batch_size  = args.embed_batch_size,
                    layer_idx   = best_layer,
                    split_label = f"{variant_name}/{split_name}",
                )
                if emb_shape is None:
                    emb_shape = embs[0].shape
                with tf.device("CPU"):
                    dataset_tf[split_name] = (
                        tf.data.Dataset.from_tensor_slices((embs, labels))
                        .shuffle(args.downstream_batch * 4)
                        .batch(args.downstream_batch)
                    )

            print(f"    Embedding shape: {emb_shape}")

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

                _, acc, roc, pr = cnn.evaluate(dataset_tf["test"], verbose=0)
                acc_list.append(acc)
                auroc_list.append(roc)
                aupr_list.append(pr)
                print(f"    rep {rep+1}/{args.n_repeats}: "
                      f"acc={acc:.4f}  auroc={roc:.4f}  aupr={pr:.4f}")
                tf.keras.backend.clear_session()

            rec = {
                "model_variant": variant_name,
                "task_id":       task_id,
                "rbp":           rbp_name,
                "source":        source_tag,
                "best_layer":    best_layer,
                "acc_mean":      float(np.mean(acc_list)),
                "acc_std":       float(np.std(acc_list)),
                "auroc_mean":    float(np.mean(auroc_list)),
                "auroc_std":     float(np.std(auroc_list)),
                "aupr_mean":     float(np.mean(aupr_list)),
                "aupr_std":      float(np.std(aupr_list)),
            }
            records.append(rec)
            print(f"    → AUROC {rec['auroc_mean']:.4f}±{rec['auroc_std']:.4f}  "
                  f"AUPR {rec['aupr_mean']:.4f}±{rec['aupr_std']:.4f}")

            # Incremental save after each RBP
            existing = []
            if os.path.exists(results_csv):
                try:
                    existing = pd.read_csv(results_csv).to_dict("records")
                except Exception:
                    pass
            pd.DataFrame(existing + records).to_csv(results_csv, index=False)

        # Free GPU memory before loading next variant
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── final summary ─────────────────────────────────────────────────────────
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
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
