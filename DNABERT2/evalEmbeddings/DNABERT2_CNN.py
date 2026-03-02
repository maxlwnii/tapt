"""
Combined DNABERT2 + CNN evaluation script.
Loads data from two CSV-based data roots:
  - DNABERT2/data                  (eCLIP / CLIP RBPs)
  - Thesis/data/finetune_data_koo  (Koo RBPs)

Step 1 – Layer search: probes every transformer layer via mean-pooling +
         logistic regression on a pilot RBP.  Finds the layer with the
         highest linear-probe AUROC.  Result is cached so re-runs skip it.

Step 2 – CNN training: extracts full per-token hidden states from the best
         layer (seq_len, hidden_dim), then trains the 1-D CNN (N_REPEATS
         repeats per RBP) and records AUROC / AUPR / Accuracy.
"""

import argparse
import os
import glob
from pathlib import Path

# TF log level before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from layer_search import run_layer_search, _call_with_all_layers

# ── device setup ──────────────────────────────────────────────────────────────
tf.config.set_visible_devices([], "GPU")          # TF  → CPU (small CNN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"TensorFlow  : {tf.__version__}  (CPU only)")
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA (torch): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}")
else:
    print("  WARNING: no CUDA – DNABERT2 embedding will be slow")
print(f"DNABERT2 device: {device}")

# ── defaults ──────────────────────────────────────────────────────────────────
_DATA_ROOTS = [
    "/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/data",
    "/home/fr/fr_fr/fr_ml642/Thesis/data/finetune_data_koo",
]
_OUTPUT_DIR = (
    "/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/DNABERT2/evalEmbeddings"
    "/results/results_combined"
)
_MODEL_NAME = "zhihan1996/DNABERT-2-117M"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DNABERT2 layer search + CNN")
    p.add_argument("--data_roots", nargs="+", default=_DATA_ROOTS)
    p.add_argument("--output_dir", default=_OUTPUT_DIR)
    p.add_argument("--model_name", default=_MODEL_NAME)
    p.add_argument("--model_max_length", type=int, default=512)
    p.add_argument("--embed_batch_size", type=int, default=32)
    p.add_argument("--downstream_batch", type=int, default=256)
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--best_layer",
        type=int,
        default=None,
        help="Skip layer search and use this layer index directly.",
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
        help="Cap pilot-RBP samples used for layer search (speeds it up).",
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
        seq_col   = find_column(list(df.columns), ["sequence", "seq", "text", "input"])
        label_col = find_column(list(df.columns), ["label", "labels", "target", "y"])
        splits[split] = (df[seq_col].astype(str).tolist(),
                         df[label_col].to_numpy(dtype=np.int64))
    return splits


def discover_tasks(data_roots):
    """
    Walk each root for subdirectories with train/dev/test CSVs.
    Returns list of (task_id, rbp_name, source_tag, rbp_dir).
    """
    tasks = []
    seen  = set()
    for root in data_roots:
        if not os.path.isdir(root):
            print(f"[WARN] Data root not found, skipping: {root}")
            continue
        source_tag = os.path.basename(root)
        for rbp_dir in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(rbp_dir):
                continue
            rbp_name = os.path.basename(rbp_dir)
            task_id  = f"{source_tag}/{rbp_name}"
            if task_id in seen:
                continue
            if not os.path.exists(os.path.join(rbp_dir, "train.csv")):
                continue
            seen.add(task_id)
            tasks.append((task_id, rbp_name, source_tag, rbp_dir))
    return tasks


def completed_tasks(results_csv):
    if not os.path.exists(results_csv):
        return set()
    try:
        return set(pd.read_csv(results_csv)["task_id"].unique())
    except Exception:
        return set()


# ── DNABERT2 model ────────────────────────────────────────────────────────────
def load_dnabert2(model_name):
    print(f"\n[INFO] Loading DNABERT2: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        _fast_init=False,       # <-- this is the key fix; skips meta-device init
    )

    model.to(device)
    model.eval()
    print("[INFO] DNABERT2 loaded\n")
    return model, tokenizer


def extract_embeddings(seqs, model, tokenizer, model_max_length,
                       embed_batch_size, layer_idx=None, split_label=""):
    """
    Return list of (seq_len, hidden_dim) numpy arrays – one per sequence.

    If layer_idx is None, uses the final hidden state (out[0]).
    If layer_idx is an int, uses output_hidden_states=True and indexes into
    out.hidden_states[layer_idx].
    """
    all_embs = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(seqs), embed_batch_size),
            desc=f"  embed {split_label}",
            unit="batch",
        ):
            batch = seqs[i : i + embed_batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=model_max_length,
            ).to(device)

            if layer_idx is not None:
                hidden = _call_with_all_layers(model, dict(enc))[layer_idx]  # (B, seq_len, H)
            else:
                out = model(**enc)
                hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state

            all_embs.extend(hidden.cpu().numpy())
    return all_embs


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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    output_dir  = args.output_dir
    results_csv = os.path.join(output_dir, "DNABERT2_CNN_results.csv")
    cache_dir   = Path(output_dir) / "cache"
    plots_dir   = Path(output_dir) / "layer_search"
    os.makedirs(output_dir, exist_ok=True)

    tasks   = discover_tasks(args.data_roots)
    done    = completed_tasks(results_csv)
    pending = [t for t in tasks if t[0] not in done]

    print(f"Tasks total  : {len(tasks)}")
    print(f"Already done : {len(done)}")
    print(f"To process   : {len(pending)}\n")

    if not pending:
        print("Nothing to do.")
        return

    # ── step 1: layer search ─────────────────────────────────────────────────
    if args.best_layer is not None:
        best_layer = args.best_layer
        print(f"[INFO] Using forced best_layer={best_layer} (--best_layer flag)\n")
    else:
        # Pick pilot RBP
        if args.pilot_rbp:
            pilot_candidates = [t for t in tasks if t[1] == args.pilot_rbp]
            if not pilot_candidates:
                print(f"[WARN] --pilot_rbp '{args.pilot_rbp}' not found; using first task.")
                pilot_task = tasks[0]
            else:
                pilot_task = pilot_candidates[0]
        else:
            pilot_task = tasks[0]   # first alphabetically

        _, pilot_rbp_name, _, pilot_dir = pilot_task
        pilot_splits = load_splits(pilot_dir)
        # Combine train+dev for more signal
        pilot_seqs   = pilot_splits["train"][0] + pilot_splits["valid"][0]
        pilot_labels = np.concatenate([pilot_splits["train"][1],
                                       pilot_splits["valid"][1]])

        # Cap samples for speed
        if args.max_pilot_samples and len(pilot_seqs) > args.max_pilot_samples:
            rng          = np.random.default_rng(args.seed)
            idx          = rng.choice(len(pilot_seqs), args.max_pilot_samples, replace=False)
            pilot_seqs   = [pilot_seqs[i] for i in idx]
            pilot_labels = pilot_labels[idx]

        print(f"[INFO] Running layer search on pilot RBP: {pilot_rbp_name} "
              f"({len(pilot_seqs)} samples)")

        best_layer = run_layer_search(
            sequences          = pilot_seqs,
            labels             = pilot_labels,
            model_path         = args.model_name,
            fallback_tokenizer = args.model_name,
            fallback_model     = args.model_name,
            max_length         = args.model_max_length,
            batch_size         = args.embed_batch_size,
            device             = device,
            cache_dir          = cache_dir / "layer_search",
            output_dir         = plots_dir,
            rbp_name           = pilot_rbp_name,
            num_folds          = 5,
            seed               = args.seed,
        )
        print(f"\n[INFO] Best layer: {best_layer}\n")

    # ── step 2: load DNABERT2 and run CNN per RBP ────────────────────────────
    dnabert2, tokenizer = load_dnabert2(args.model_name)
    records = []

    for task_id, rbp_name, source_tag, rbp_dir in pending:
        print(f"\n{'='*60}")
        print(f"Task  : {task_id}")
        print(f"Layer : {best_layer}")

        splits = load_splits(rbp_dir)
        if splits is None:
            print("  [SKIP] missing CSV files")
            continue

        # ── extract per-token embeddings from best layer ──────────────────────
        dataset_tf = {}
        emb_shape  = None

        for split_name in ("train", "valid", "test"):
            seqs, labels = splits[split_name]
            print(f"  {split_name}: {len(seqs)} sequences")

            embs = extract_embeddings(
                seqs, dnabert2, tokenizer,
                model_max_length = args.model_max_length,
                embed_batch_size = args.embed_batch_size,
                layer_idx        = best_layer,
                split_label      = split_name,
            )

            if emb_shape is None:
                emb_shape = embs[0].shape

            with tf.device("CPU"):
                ds = (
                    tf.data.Dataset.from_tensor_slices((embs, labels))
                    .shuffle(args.downstream_batch * 4)
                    .batch(args.downstream_batch)
                )
            dataset_tf[split_name] = ds

        print(f"  Embedding shape : {emb_shape}  (layer {best_layer})")

        # ── train CNN ────────────────────────────────────────────────────────
        acc_list, auroc_list, aupr_list = [], [], []

        for rep in range(args.n_repeats):
            auroc_metric = tf.keras.metrics.AUC(curve="ROC", name="auroc")
            aupr_metric  = tf.keras.metrics.AUC(curve="PR",  name="aupr")

            cnn = chip_cnn(emb_shape, 1)
            cnn.compile(
                loss      = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics   = ["accuracy", auroc_metric, aupr_metric],
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

        # ── store result ──────────────────────────────────────────────────────
        rec = {
            "task_id"    : task_id,
            "rbp"        : rbp_name,
            "source"     : source_tag,
            "best_layer" : best_layer,
            "acc_mean"   : float(np.mean(acc_list)),
            "acc_std"    : float(np.std(acc_list)),
            "auroc_mean" : float(np.mean(auroc_list)),
            "auroc_std"  : float(np.std(auroc_list)),
            "aupr_mean"  : float(np.mean(aupr_list)),
            "aupr_std"   : float(np.std(aupr_list)),
        }
        records.append(rec)
        print(f"  → AUROC {rec['auroc_mean']:.4f}±{rec['auroc_std']:.4f}  "
              f"AUPR {rec['aupr_mean']:.4f}±{rec['aupr_std']:.4f}")

        # ── incremental save ──────────────────────────────────────────────────
        existing = []
        if os.path.exists(results_csv):
            try:
                existing = pd.read_csv(results_csv).to_dict("records")
            except Exception:
                pass
        pd.DataFrame(existing + records).to_csv(results_csv, index=False)
        print(f"  Results saved → {results_csv}")

    # ── final summary ─────────────────────────────────────────────────────────
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        print("\n" + "=" * 60)
        print("FINAL SUMMARY (mean per source)")
        print("=" * 60)
        print(df.groupby("source")[["auroc_mean", "aupr_mean", "acc_mean"]].mean().to_string())
        print(f"\nBest layer used: {best_layer}")
        print("\nOverall:")
        print(df[["auroc_mean", "aupr_mean", "acc_mean"]].mean().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
