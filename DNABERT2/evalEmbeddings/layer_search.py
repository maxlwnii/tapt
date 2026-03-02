"""
layer_search.py
---------------
Finds the optimal intermediate transformer layer for DNA embedding tasks.
Drop this file into your project and call `run_layer_search()` from main().

Key design: extracts ALL layers in a SINGLE forward pass per batch (efficient).
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Tokenizer / model loading (mirrors your existing helpers)
# ---------------------------------------------------------------------------

def get_tokenizer(model_path: str, fallback: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)


def load_model(model_path: str, fallback_model: str):
    """
    Load model for layer search. Handles:
    - Single weights files (.safetensors / LAMAR 'weights' file)
    - HF checkpoint directories with config.json
    - Checkpoint directories WITHOUT config.json (LAMAR checkpoints)

    If fallback_model is a .json file, it is used as a standalone config
    (AutoConfig.from_pretrained + AutoModel.from_config).
    """
    from transformers import AutoConfig

    def _build_base() -> AutoModel:
        fb = Path(fallback_model)
        if fb.is_file() and fb.suffix == ".json":
            cfg = AutoConfig.from_pretrained(str(fb))
            return AutoModel.from_config(cfg)
        return AutoModel.from_pretrained(fallback_model, trust_remote_code=True, low_cpu_mem_usage=False)

    p = Path(model_path)

    # ── HuggingFace model ID (not a local path at all) ───────────────────────
    if not p.exists():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return AutoModel.from_pretrained(
            model_path, trust_remote_code=True, config=config, low_cpu_mem_usage=False
        )

    # ── Local weights file (.safetensors / LAMAR 'weights') ──────────────────
    if p.is_file():
        model = _build_base()
        suffix = p.suffix.lower()
        if suffix == ".safetensors" or p.name == "weights":
            from safetensors.torch import load_file
            state = load_file(str(p))
        else:
            state = torch.load(str(p), map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                state = state.get("state_dict", state.get("model", state))
        model.load_state_dict(state, strict=False)
        return model

    # ── Local directory with config.json ─────────────────────────────────────
    config_json = p / "config.json"
    if config_json.exists():
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return AutoModel.from_pretrained(model_path, trust_remote_code=True, config=config)

    # ── Local directory without config.json (LAMAR tapt checkpoint) ──────────
    model = _build_base()
    st_file = p / "model.safetensors"
    if not st_file.exists():
        raise FileNotFoundError(f"No config.json and no model.safetensors in: {model_path}")
    from safetensors.torch import load_file
    state = load_file(str(st_file))
    model.load_state_dict(state, strict=False)
    return model


# ---------------------------------------------------------------------------
# Core: mean-pool a single layer's hidden state
# ---------------------------------------------------------------------------

def _call_with_all_layers(model, kwargs: dict) -> List[torch.Tensor]:
    """
    Get per-layer hidden states as a list of (B, seq_len, H) tensors.

    Uses forward hooks on encoder.layer[i] — no special model flags needed.
    This avoids both output_hidden_states=True (broken for DNABERT-2's
    FlashAttention path → CUBLAS crash) and output_all_encoded_layers=True
    (also crashes in the same matmul).

    DNABERT-2 runs unpad_input before the encoder, so each BertLayer receives
    and emits unpadded (ntokens_unpad, H) tensors.  We scatter them back to
    (B, seq_len, H) using attention_mask.

    For standard HF models each layer emits (B, seq_len, H) directly.
    """
    import inspect as _inspect

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("Model has no 'encoder' attribute")

    layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
    if layers is None:
        raise ValueError("Cannot find encoder.layer / encoder.layers")

    # Detect DNABERT-2's unpadded architecture.
    try:
        _uses_unpadded = (
            "output_all_encoded_layers" in _inspect.signature(encoder.forward).parameters
        )
    except (ValueError, TypeError):
        _uses_unpadded = False

    # ── register forward hooks ────────────────────────────────────────────────
    captured: Dict[int, torch.Tensor] = {}
    hooks = []

    def _make_hook(idx: int):
        def _hook(_module, _input, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            captured[idx] = h
        return _hook

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(_make_hook(i)))
        model(**kwargs)          # plain forward — no special flags
    finally:
        for h in hooks:
            h.remove()

    ordered: List[torch.Tensor] = [captured[i] for i in range(len(layers))]

    # ── re-pad for DNABERT-2 ─────────────────────────────────────────────────
    if _uses_unpadded:
        attention_mask = kwargs["attention_mask"]      # (B, seq_len)
        batch_size, seq_len = attention_mask.shape
        hidden_size = ordered[0].shape[-1]
        flat_mask = attention_mask.bool().view(-1)     # (B * seq_len,)

        padded: List[torch.Tensor] = []
        for layer_out in ordered:
            buf = torch.zeros(
                batch_size, seq_len, hidden_size,
                device=layer_out.device, dtype=layer_out.dtype,
            )
            buf.view(-1, hidden_size)[flat_mask] = layer_out
            padded.append(buf)
        return padded

    return ordered


def mean_pool(
    hidden: torch.Tensor,         # (batch, seq_len, hidden)
    attention_mask: torch.Tensor, # (batch, seq_len)
    input_ids: torch.Tensor,      # (batch, seq_len)
    special_ids: List[int],
) -> torch.Tensor:                # (batch, hidden)
    """Mean pool hidden states, excluding padding and special tokens."""
    valid = attention_mask.bool()
    if special_ids:
        special_mask = torch.zeros_like(valid)
        for sid in special_ids:
            special_mask |= input_ids.eq(sid)
        valid = valid & ~special_mask
    denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (hidden * valid.unsqueeze(-1)).sum(dim=1) / denom


# ---------------------------------------------------------------------------
# Efficient: extract ALL layers in one pass, return dict {layer_idx: ndarray}
# ---------------------------------------------------------------------------

def extract_all_layers(
    sequences: List[str],
    model_path: str,
    fallback_tokenizer: str,
    fallback_model: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], int]:
    """
    Extract embeddings from all transformer layers in a single forward pass.

    Returns:
        embeddings : dict mapping layer_idx -> ndarray of shape (N, hidden_size)
        n_layers   : total number of hidden states (embedding + transformer blocks)
    """
    tokenizer = get_tokenizer(model_path, fallback_tokenizer)
    model = load_model(model_path, fallback_model)
    model.to(device)
    model.eval()

    # Discover layer count with a dummy pass
    with torch.no_grad():
        dummy = tokenizer(
            sequences[:min(2, len(sequences))], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        dummy = {k: v.to(device) for k, v in dummy.items()}
        n_layers = len(_call_with_all_layers(model, dummy))

    print(f"  [layer_search] Model has {n_layers} hidden states "
          f"(1 embedding + {n_layers - 1} transformer blocks)")

    # Accumulate per-layer pooled vectors
    layer_vecs: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_layers)}

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            all_hidden = _call_with_all_layers(model, tokens)
            for layer_idx, hidden in enumerate(all_hidden):
                pooled = mean_pool(
                    hidden, tokens["attention_mask"],
                    tokens["input_ids"], tokenizer.all_special_ids,
                )
                layer_vecs[layer_idx].append(
                    pooled.detach().cpu().numpy().astype(np.float32)
                )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings = {
        i: np.concatenate(vecs, axis=0) for i, vecs in layer_vecs.items()
    }
    return embeddings, n_layers


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _n_layers_from_cache(cache_dir: Path, model_path: str) -> Optional[int]:
    """Return the number of cached layers for a model, or None if not cached yet."""
    model_key = hashlib.md5(model_path.encode()).hexdigest()[:8]
    model_dir = cache_dir / f"model_{model_key}"
    if not model_dir.exists():
        return None
    layer_dirs = sorted(model_dir.glob("layer_*"))
    return len(layer_dirs) if layer_dirs else None


def _cache_path(cache_dir: Path, model_path: str, layer_idx: int, rbp: str) -> Path:
    """Determine cache path for a specific layer and RBP."""
    model_key = hashlib.md5(model_path.encode()).hexdigest()[:8]
    d = cache_dir / f"model_{model_key}" / f"layer_{layer_idx:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{rbp.replace('/', '_')}.npy"


def save_layer_embeddings(
    cache_dir: Path,
    model_path: str,
    rbp: str,
    embeddings: Dict[int, np.ndarray],
) -> None:
    """Save all layer embeddings to cache."""
    for layer_idx, X in embeddings.items():
        path = _cache_path(cache_dir, model_path, layer_idx, rbp)
        np.save(path, X)


def load_layer_embeddings(
    cache_dir: Path,
    model_path: str,
    rbp: str,
    n_layers: int,
) -> Optional[Dict[int, np.ndarray]]:
    """Load all layer embeddings from cache. Returns None if any layer is missing."""
    result = {}
    for i in range(n_layers):
        p = _cache_path(cache_dir, model_path, i, rbp)
        if not p.exists():
            return None
        result[i] = np.load(p)
    return result


# ---------------------------------------------------------------------------
# Linear probe evaluation (single layer)
# ---------------------------------------------------------------------------

def probe_layer(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Evaluate a single layer's embeddings via linear probing. Returns mean AUROC across stratified K-folds."""
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_auroc(layer_aurocs: Dict[int, float], out_path: Path) -> None:
    """Plot AUROC curve across layers."""
    layers = sorted(layer_aurocs)
    aurocs = [layer_aurocs[i] for i in layers]
    best = max(layer_aurocs, key=layer_aurocs.get)

    plt.figure(figsize=(10, 4))
    plt.plot(layers, aurocs, marker="o", linewidth=1.5)
    plt.axvline(best, linestyle="--", color="red", label=f"Best: layer {best}")
    plt.scatter([best], [layer_aurocs[best]], color="red", zorder=5)
    plt.xlabel("Layer index  (0 = token embeddings)")
    plt.ylabel("Mean AUROC (CV)")
    plt.title("Linear probe AUROC per transformer layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [layer_search] Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_layer_search(
    sequences: List[str],
    labels: np.ndarray,
    model_path: str,
    fallback_tokenizer: str,
    fallback_model: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    cache_dir: Path,
    output_dir: Path,
    rbp_name: str = "pilot",
    num_folds: int = 5,
    seed: int = 42,
) -> int:
    """
    Probes every transformer layer and returns the index with the best AUROC.

    Usage in main():
        best_layer = run_layer_search(
            sequences=pilot_seqs,
            labels=pilot_labels,
            model_path=args.tapt_lamar_model,
            fallback_tokenizer=args.fallback_tokenizer,
            fallback_model=args.base_model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
            cache_dir=cache_dir / "layer_search",
            output_dir=output_dir,
            rbp_name=pilot_rbp,
        )
        # Then use best_layer in your main embedding loop:
        #   transformer_embeddings(..., layer_idx=best_layer)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Layer Search | model={model_path} | RBP={rbp_name} ===")

    # --- Step 1: Get or build embeddings for all layers (single model load) ---
    # Try cache first without loading the model.
    layer_embeddings = None
    n_layers = _n_layers_from_cache(cache_dir, model_path)
    if n_layers is not None:
        cached = load_layer_embeddings(cache_dir, model_path, rbp_name, n_layers)
        if cached is not None:
            print(f"  [layer_search] Loaded all {n_layers} layers from cache")
            layer_embeddings = cached

    if layer_embeddings is None:
        print(f"  [layer_search] Extracting all layers (single forward pass)\u2026")
        layer_embeddings, n_layers = extract_all_layers(
            sequences, model_path, fallback_tokenizer,
            fallback_model, max_length, batch_size, device,
        )
        save_layer_embeddings(cache_dir, model_path, rbp_name, layer_embeddings)

    # --- Step 2: Probe each layer ---
    layer_aurocs: Dict[int, float] = {}
    for layer_idx in range(n_layers):
        X = layer_embeddings[layer_idx]

        # Skip degenerate layers (all-zero embeddings)
        if np.linalg.norm(X, axis=1).mean() < 1e-6:
            print(f"    Layer {layer_idx:2d}: SKIPPED (zero-norm embeddings)")
            continue

        auroc = probe_layer(X, labels, n_splits=num_folds, seed=seed)
        layer_aurocs[layer_idx] = auroc
        print(f"    Layer {layer_idx:2d}: AUROC = {auroc:.4f}")

    # --- Step 3: Report ---
    best_layer = max(layer_aurocs, key=layer_aurocs.get)
    best_auroc = layer_aurocs[best_layer]

    results = {
        "best_layer": best_layer,
        "best_auroc": best_auroc,
        "pilot_rbp": rbp_name,
        "model": model_path,
        "layer_aurocs": {str(k): v for k, v in layer_aurocs.items()},
    }
    out_json = output_dir / "layer_search.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    plot_layer_auroc(layer_aurocs, output_dir / "layer_auroc_curve.png")

    print(
        f"\n=== Layer Search Complete ===\n"
        f"  Model   : {model_path}\n"
        f"  Pilot   : {rbp_name}\n"
        f"  Layers  : {n_layers} total\n"
        f"  Best    : Layer {best_layer} → AUROC = {best_auroc:.4f}\n"
        f"  Results : {out_json}\n"
        f"  Plot    : {output_dir / 'layer_auroc_curve.png'}"
    )

    return best_layer
