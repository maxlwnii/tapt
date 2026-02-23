import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear probing for RBP binding prediction using frozen DNA embeddings"
    )

    p.add_argument(
        "--data_roots",
        nargs="+",
        default=[
            "/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/data",
            "/home/fr/fr_fr/fr_ml642/Thesis/data/finetune_data_koo",
        ],
        help="Root folders containing per-RBP subfolders with train/dev/test CSV files",
    )
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--base_model",
        type=str,
        default="zhihan1996/DNABERT-2-117M",
        help="HF id or local path for base DNABERT2",
    )
    p.add_argument(
        "--tapt_lamar_model",
        type=str,
        default="/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/src/pretrain/saving_model/tapt_lamar/checkpoint-98000",
        help="Path to TAPT LAMAR checkpoint",
    )
    p.add_argument(
        "--pretrained_lamar_model",
        type=str,
        default="/home/fr/fr_fr/fr_ml642/Thesis/LAMAR/weights",
        help="Path to pretrained LAMAR weights (checkpoint dir or single weights file)",
    )
    p.add_argument(
        "--fallback_tokenizer",
        type=str,
        default="zhihan1996/DNABERT-2-117M",
        help="Tokenizer fallback if local checkpoint lacks tokenizer files",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/evalEmbeddings/results/linear_probe_embedding_quality",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/evalEmbeddings/results/linear_probe_embedding_quality/cache",
    )
    p.add_argument("--max_points_vis", type=int, default=5000)
    p.add_argument("--max_points_cosine", type=int, default=1200)
    p.add_argument("--max_rbps", type=int, default=0, help="If >0, evaluate only first N RBPs")
    p.add_argument("--max_samples_per_rbp", type=int, default=0, help="If >0, cap samples per RBP")
    p.add_argument(
        "--embedding_models",
        nargs="+",
        default=["one_hot", "base_dnabert2", "tapt_lamar", "pretrained_lamar"],
        help="Subset of embedding models to run",
    )
    p.add_argument("--skip_plots", action="store_true", help="Skip plotting for fast smoke tests")

    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_column(columns: List[str], candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(f"Could not find column among candidates {candidates} in {columns}")


def load_rbp_tasks(data_roots: List[str]) -> Dict[str, pd.DataFrame]:
    tasks: Dict[str, List[pd.DataFrame]] = {}

    for root in data_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        for rbp_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
            split_files = [rbp_dir / s for s in ("train.csv", "dev.csv", "test.csv") if (rbp_dir / s).exists()]
            if not split_files:
                continue

            pieces = []
            for csv_path in split_files:
                df = pd.read_csv(csv_path)
                seq_col = find_column(list(df.columns), ["sequence", "seq", "text", "input", "x"])
                label_col = find_column(list(df.columns), ["label", "labels", "target", "y"])

                chunk = pd.DataFrame(
                    {
                        "sequence": df[seq_col].astype(str),
                        "label": df[label_col].astype(int),
                    }
                )
                chunk = chunk.dropna()
                pieces.append(chunk)

            if not pieces:
                continue

            task_name = rbp_dir.name
            merged = pd.concat(pieces, axis=0, ignore_index=True).drop_duplicates(subset=["sequence", "label"])
            if merged["label"].nunique() < 2:
                continue

            tasks.setdefault(task_name, []).append(merged)

    out: Dict[str, pd.DataFrame] = {}
    for rbp, dfs in tasks.items():
        merged = pd.concat(dfs, axis=0, ignore_index=True).drop_duplicates(subset=["sequence", "label"])
        if merged["label"].nunique() >= 2:
            out[rbp] = merged
    return out


def one_hot_embed(sequences: List[str], max_length: int) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    embs = np.zeros((len(sequences), max_length * 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        mat = np.zeros((max_length, 4), dtype=np.float32)
        for j, ch in enumerate(seq[:max_length]):
            idx = mapping.get(ch)
            if idx is not None:
                mat[j, idx] = 1.0
        embs[i] = mat.reshape(-1)
    return embs


def _model_cache_key(model_name: str, max_length: int) -> str:
    raw = f"{model_name}__L{max_length}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def get_tokenizer(model_path: str, fallback_tokenizer: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback_tokenizer, trust_remote_code=True)


def _extract_state_dict(raw_obj):
    if isinstance(raw_obj, dict):
        if "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
            return raw_obj["state_dict"]
        if "model" in raw_obj and isinstance(raw_obj["model"], dict):
            return raw_obj["model"]
    return raw_obj


def load_model_for_embeddings(model_path: str, fallback_model: str):
    model_file = Path(model_path)

    if model_file.is_file():
        model = AutoModel.from_pretrained(fallback_model, trust_remote_code=True)
        suffix = model_file.suffix.lower()

        if suffix == ".safetensors" or model_file.name == "weights":
            from safetensors.torch import load_file

            state = load_file(str(model_file))
        else:
            state = torch.load(str(model_file), map_location="cpu")
            state = _extract_state_dict(state)

        load_info = model.load_state_dict(state, strict=False)
        missing = len(getattr(load_info, "missing_keys", []))
        unexpected = len(getattr(load_info, "unexpected_keys", []))
        print(f"[INFO] Loaded file weights {model_file} (missing={missing}, unexpected={unexpected})")
        return model

    return AutoModel.from_pretrained(model_path, trust_remote_code=True)


def mean_pool_last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, input_ids: torch.Tensor, special_ids: List[int]) -> torch.Tensor:
    valid = attention_mask.bool()
    if special_ids:
        special_mask = torch.zeros_like(valid)
        for sid in special_ids:
            special_mask |= input_ids.eq(sid)
        valid = valid & (~special_mask)

    denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    pooled = (hidden * valid.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


def transformer_embeddings(
    sequences: List[str],
    model_path: str,
    fallback_tokenizer: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    fallback_model: str,
) -> np.ndarray:
    tokenizer = get_tokenizer(model_path, fallback_tokenizer)
    model = load_model_for_embeddings(model_path=model_path, fallback_model=fallback_model)
    model.to(device)
    model.eval()

    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            out = model(**tokens)
            hidden = out.last_hidden_state
            pooled = mean_pool_last_hidden(
                hidden,
                tokens["attention_mask"],
                tokens["input_ids"],
                tokenizer.all_special_ids,
            )
            all_vecs.append(pooled.detach().cpu().numpy().astype(np.float32))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(all_vecs, axis=0)


def get_or_build_embeddings(
    sequences: List[str],
    emb_name: str,
    rbp_name: str,
    cache_dir: Path,
    build_fn,
) -> np.ndarray:
    rbp_safe = rbp_name.replace("/", "_")
    emb_dir = cache_dir / emb_name
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_path = emb_dir / f"{rbp_safe}.npy"

    if emb_path.exists():
        return np.load(emb_path)

    X = build_fn(sequences)
    np.save(emb_path, X)
    return X


def evaluate_linear_probe(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    metrics = {"auroc": [], "accuracy": [], "f1": [], "auprc": []}

    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
        )
        clf.fit(X_tr, y_tr)

        prob = clf.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics["auroc"].append(roc_auc_score(y_te, prob))
        metrics["accuracy"].append(accuracy_score(y_te, pred))
        metrics["f1"].append(f1_score(y_te, pred))
        metrics["auprc"].append(average_precision_score(y_te, prob))

    out = {}
    for k, vals in metrics.items():
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def summarize_table(per_rbp_df: pd.DataFrame) -> pd.DataFrame:
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
            }
        )
    return pd.DataFrame(rows).sort_values("Mean AUROC", ascending=False)


def plot_boxplot(per_rbp_df: pd.DataFrame, out_path: Path) -> None:
    model_order = sorted(per_rbp_df["model"].unique())
    data = [per_rbp_df.loc[per_rbp_df["model"] == m, "auroc"].values for m in model_order]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=model_order, showmeans=True)
    plt.ylabel("AUROC")
    plt.title("AUROC distribution across RBPs")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_rbp_bars(per_rbp_df: pd.DataFrame, out_path: Path) -> None:
    pivot = per_rbp_df.pivot(index="rbp", columns="model", values="auroc")
    base_col = "base_dnabert2" if "base_dnabert2" in pivot.columns else pivot.columns[0]
    pivot = pivot.sort_values(base_col)

    ax = pivot.plot(kind="bar", figsize=(14, 6), width=0.85)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-RBP AUROC by embedding model")
    ax.legend(loc="lower right")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_delta(per_rbp_df: pd.DataFrame, out_path: Path) -> None:
    pivot = per_rbp_df.pivot(index="rbp", columns="model", values="auroc")
    if not {"base_dnabert2", "tapt_lamar"}.issubset(set(pivot.columns)):
        return

    delta = (pivot["tapt_lamar"] - pivot["base_dnabert2"]).sort_values()
    plt.figure(figsize=(12, 5))
    plt.bar(delta.index, delta.values)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("ΔAUROC (TAPT - Base)")
    plt.title("Per-RBP delta AUROC")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fit_projection(X: np.ndarray, seed: int):
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=seed)
        method = "umap"
    except Exception:
        reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
        method = "tsne"

    Z = reducer.fit_transform(X)
    return Z, method


def plot_embedding_projection(
    embeddings_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    rbp_names: Dict[int, str],
    out_dir: Path,
    seed: int,
    max_points: int,
) -> None:
    rng = np.random.default_rng(seed)

    for model_name, (X, rbp_idx) in embeddings_by_model.items():
        if len(X) > max_points:
            pick = rng.choice(len(X), size=max_points, replace=False)
            X_plot = X[pick]
            y_plot = rbp_idx[pick]
        else:
            X_plot = X
            y_plot = rbp_idx

        Z, method = fit_projection(X_plot, seed)

        plt.figure(figsize=(8, 6))
        uniq = np.unique(y_plot)
        for u in uniq:
            m = y_plot == u
            plt.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7, label=rbp_names[int(u)])
        plt.title(f"{model_name}: {method.upper()} projection")
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        if len(uniq) <= 15:
            plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(out_dir / f"{model_name}_{method}.png", dpi=220)
        plt.close()


def embedding_health_stats(embeddings_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]], max_points_cosine: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for model_name, (X, _) in embeddings_by_model.items():
        norms = np.linalg.norm(X, axis=1)

        if len(X) > max_points_cosine:
            idx = rng.choice(len(X), size=max_points_cosine, replace=False)
            Xc = X[idx]
        else:
            Xc = X

        C = cosine_similarity(Xc)
        iu = np.triu_indices_from(C, k=1)
        cos_vals = C[iu]

        rows.append(
            {
                "model": model_name,
                "norm_mean": float(np.mean(norms)),
                "norm_std": float(np.std(norms)),
                "cosine_mean": float(np.mean(cos_vals)),
                "cosine_std": float(np.std(cos_vals)),
            }
        )

    return pd.DataFrame(rows)


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

    tasks = load_rbp_tasks(args.data_roots)
    if not tasks:
        raise RuntimeError("No valid RBP tasks found in provided data roots.")

    tasks_items = sorted(tasks.items())
    if args.max_rbps > 0:
        tasks_items = tasks_items[: args.max_rbps]
        tasks = dict(tasks_items)

    print(f"Loaded {len(tasks)} RBP tasks")

    raw_model_specs = {
        "base_dnabert2": args.base_model,
        "tapt_lamar": args.tapt_lamar_model,
        "pretrained_lamar": args.pretrained_lamar_model,
    }
    model_specs = {}
    for name, model_ref in raw_model_specs.items():
        if not model_ref:
            continue
        is_local_path = model_ref.startswith("/") or model_ref.startswith(".")
        if is_local_path and not Path(model_ref).exists():
            print(f"[WARN] Skipping {name}: local path not found -> {model_ref}")
            continue
        model_specs[name] = model_ref

    selected_models = set(args.embedding_models)

    per_rbp_records = []
    embedding_pool: Dict[str, List[np.ndarray]] = {}
    rbp_pool: Dict[str, List[np.ndarray]] = {}
    rbp_to_int = {rbp: i for i, rbp in enumerate(sorted(tasks.keys()))}

    for rbp, df in sorted(tasks.items()):
        if args.max_samples_per_rbp > 0 and len(df) > args.max_samples_per_rbp:
            df = df.sample(n=args.max_samples_per_rbp, random_state=args.seed)

        sequences = df["sequence"].tolist()
        labels = df["label"].to_numpy(dtype=np.int64)

        print(f"[RBP] {rbp}: n={len(df)}")

        emb_builders = {}
        if "one_hot" in selected_models:
            emb_builders["one_hot"] = lambda seqs=sequences: one_hot_embed(seqs, args.max_length)

        for model_name, model_path in model_specs.items():
            if model_name not in selected_models:
                continue
            emb_builders[model_name] = lambda seqs=sequences, mp=model_path: transformer_embeddings(
                seqs,
                model_path=mp,
                fallback_tokenizer=args.fallback_tokenizer,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
                fallback_model=args.base_model,
            )

        for emb_name, builder in emb_builders.items():
            X = get_or_build_embeddings(sequences, emb_name, rbp, cache_dir, builder)
            y = labels

            metrics = evaluate_linear_probe(X, y, n_splits=args.num_folds, seed=args.seed)
            per_rbp_records.append(
                {
                    "rbp": rbp,
                    "model": emb_name,
                    "auroc": metrics["auroc_mean"],
                    "accuracy": metrics["accuracy_mean"],
                    "f1": metrics["f1_mean"],
                    "auprc": metrics["auprc_mean"],
                    "auroc_std_folds": metrics["auroc_std"],
                }
            )

            embedding_pool.setdefault(emb_name, []).append(X)
            rbp_pool.setdefault(emb_name, []).append(np.full(len(X), rbp_to_int[rbp], dtype=np.int32))

    per_rbp_df = pd.DataFrame(per_rbp_records)
    per_rbp_df.to_csv(output_dir / "per_rbp_metrics.csv", index=False)

    summary_df = summarize_table(per_rbp_df)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)

    print("\n=== Summary (mean ± std across RBPs) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not args.skip_plots:
        plot_boxplot(per_rbp_df, plots_dir / "auroc_boxplot.png")
        plot_per_rbp_bars(per_rbp_df, plots_dir / "per_rbp_auroc_bars.png")
        plot_delta(per_rbp_df, plots_dir / "delta_auroc_tapt_minus_base.png")

    embeddings_by_model = {}
    for model_name in embedding_pool:
        X_all = np.concatenate(embedding_pool[model_name], axis=0)
        y_all = np.concatenate(rbp_pool[model_name], axis=0)
        embeddings_by_model[model_name] = (X_all, y_all)

    if not args.skip_plots:
        plot_embedding_projection(
            embeddings_by_model=embeddings_by_model,
            rbp_names={v: k for k, v in rbp_to_int.items()},
            out_dir=plots_dir,
            seed=args.seed,
            max_points=args.max_points_vis,
        )

    health_df = embedding_health_stats(
        embeddings_by_model=embeddings_by_model,
        max_points_cosine=args.max_points_cosine,
        seed=args.seed,
    )
    health_df.to_csv(output_dir / "embedding_health_stats.csv", index=False)

    stat_results = {}
    pivot = per_rbp_df.pivot(index="rbp", columns="model", values="auroc")
    if {"base_dnabert2", "tapt_lamar"}.issubset(set(pivot.columns)):
        w = wilcoxon(pivot["tapt_lamar"], pivot["base_dnabert2"], alternative="two-sided")
        stat_results["wilcoxon_tapt_vs_base"] = {
            "statistic": float(w.statistic),
            "pvalue": float(w.pvalue),
            "n_rbps": int(len(pivot)),
            "mean_delta_auroc": float((pivot["tapt_lamar"] - pivot["base_dnabert2"]).mean()),
        }

    with open(output_dir / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2)

    print(f"\nSaved results to: {output_dir}")
    print(f"Saved plots to:   {plots_dir}")


if __name__ == "__main__":
    main()
