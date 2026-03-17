"""
unsupervised_eval.py
====================
End-to-end unsupervised embedding-quality evaluation for all five model variants:

  lamar_pretrained   – LAMAR/weights (safetensors)          max_length=1024
  lamar_tapt_1024    – tapt_1024_standard_collator/ckpt-134000  max_length=1024
  lamar_tapt_512     – tapt_lamar/checkpoint-98000           max_length=512
  dnabert2_pretrained – zhihan1996/DNABERT-2-117M            max_length=512
  dnabert2_tapt      – dnabert2_standard_mlm/checkpoint-25652  max_length=512

For each model × RBP task:
  1.  Extract mean-pooled CLS-free hidden states (last transformer layer)
  2.  Cache embeddings as .npy  (skip re-extraction on reruns)
  3.  Compute IsoScore, RankMe, NESum, StableRank on pooled train+test matrix
  4.  Sensitivity to sequence length  (6 quantile bins, RankMe / NESum / StableRank)

Produces:
  - results/metrics.csv                (per-model × per-RBP table)
  - results/sensitivity.csv            (per-model × per-RBP × per-bin)
  - results/plots/*.png

Usage::

    python unsupervised_eval.py [options]

See ``--help`` for all flags.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – safe for HPC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ── locate the LAMAR package ─────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent   # …/p_eickhoff_isoscore/
_THESIS_ROOT = _SCRIPT_DIR.parent                # …/Thesis/
_LAMAR_PKG   = _THESIS_ROOT / "LAMAR"
if str(_LAMAR_PKG) not in sys.path:
    sys.path.insert(0, str(_LAMAR_PKG))

from LAMAR.modeling_nucESM2 import EsmForMaskedLM  # noqa: E402
from safetensors.torch import load_file            # noqa: E402

# ── IsoScore (from p_eickhoff_isoscore/IsoScore/) ───────────────────────────
_ISO_DIR = _SCRIPT_DIR / "IsoScore"
if str(_ISO_DIR) not in sys.path:
    sys.path.insert(0, str(_ISO_DIR))
from IsoScore import IsoScore as _IsoScore  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")
if torch.cuda.is_available():
    print(f"[GPU]    {torch.cuda.get_device_name(0)}")
else:
    print("[WARN] CUDA not available; running on CPU. Embedding extraction can be very slow and task progress may appear stuck without batch-level logs.")

# ─────────────────────────────────────────────────────────────────────────────
#  Model specs
# ─────────────────────────────────────────────────────────────────────────────

_BASE = str(_THESIS_ROOT)
_DB2_BASE = "/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2"

MODEL_SPECS: dict[str, dict] = {
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
    "lamar_tapt_512": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_lamar/checkpoint-98000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   512,
    },
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
    # ── New checkpoints requested for evaluation ──────────────────────────
    "lamar_tapt_512_std": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_512_standard_collator_1gpu/checkpoint-265000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   512,
    },
    "dnabert2_tapt_v3": {
        "type":         "dnabert2",
        "weights_path": f"{_BASE}/DNABERT2/pretrain/models"
                        "/dnabert2_tapt_v3/checkpoint-2566",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
    },
    # ── Layer-6 variants (6th transformer block output) ───────────────────
    "lamar_pretrained_layer6": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/weights",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   1024,
        "layer_idx":    6,
    },
    "lamar_tapt_1024_layer6": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   1024,
        "layer_idx":    6,
    },
    "lamar_tapt_512_layer6": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_lamar/checkpoint-98000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   512,
        "layer_idx":    6,
    },
    "lamar_tapt_512_std_layer6": {
        "type":         "lamar",
        "weights_path": f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_512_standard_collator_1gpu/checkpoint-265000",
        "tokenizer":    f"{_BASE}/LAMAR/src/pretrain/saving_model"
                        "/tapt_1024_standard_collator/checkpoint-134000",
        "max_length":   512,
        "layer_idx":    6,
    },
    "dnabert2_pretrained_layer6": {
        "type":         "dnabert2",
        "weights_path": "zhihan1996/DNABERT-2-117M",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
        "layer_idx":    6,
    },
    "dnabert2_tapt_layer6": {
        "type":         "dnabert2",
        "weights_path": f"{_DB2_BASE}/pretrain/models"
                        "/dnabert2_standard_mlm/checkpoint-25652",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
        "layer_idx":    6,
    },
    "dnabert2_tapt_v3_layer6": {
        "type":         "dnabert2",
        "weights_path": f"{_BASE}/DNABERT2/pretrain/models"
                        "/dnabert2_tapt_v3/checkpoint-2566",
        "tokenizer":    "zhihan1996/DNABERT-2-117M",
        "max_length":   512,
        "layer_idx":    6,
    },
}

_DATA_ROOTS = [
    f"{_BASE}/DNABERT2/data",
    f"{_BASE}/data/finetune_data_koo",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(columns: list[str], candidates: list[str]) -> str:
    lmap = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lmap:
            return lmap[cand]
    raise ValueError(f"None of {candidates} found in {columns}")


def discover_tasks(data_roots: list[str]) -> list[dict]:
    """Return list of task dicts with keys: task_id, rbp_name, source, rbp_dir."""
    tasks, seen = [], set()
    for root in data_roots:
        if not os.path.isdir(root):
            print(f"[WARN] data root not found: {root}")
            continue
        source = os.path.basename(root)
        for rbp_dir in sorted(Path(root).iterdir()):
            if not rbp_dir.is_dir():
                continue
            if not (rbp_dir / "train.csv").exists():
                continue
            tid = f"{source}/{rbp_dir.name}"
            if tid in seen:
                continue
            seen.add(tid)
            tasks.append({"task_id": tid,
                           "rbp_name": rbp_dir.name,
                           "source": source,
                           "rbp_dir": str(rbp_dir)})
    return tasks


def load_seqs_and_labels(rbp_dir: str, splits: list[str]) -> tuple[list[str], np.ndarray]:
    """Load sequences + labels from requested splits."""
    all_seqs, all_labels = [], []
    fname = {"train": "train.csv", "valid": "dev.csv", "test": "test.csv"}
    for split in splits:
        path = os.path.join(rbp_dir, fname[split])
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        seq_col   = _find_col(list(df.columns), ["sequence", "seq", "text", "input"])
        label_col = _find_col(list(df.columns), ["label", "labels", "target", "y"])
        seqs   = df[seq_col].astype(str).tolist()
        labels = df[label_col].to_numpy(dtype=np.int64)
        # Replace U with T (DNA models)
        seqs = [s.replace("U", "T").replace("u", "t") for s in seqs]
        all_seqs.extend(seqs)
        all_labels.append(labels)
    if not all_seqs:
        return [], np.array([], dtype=np.int64)
    return all_seqs, np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
#  LAMAR model  (loads EsmForMaskedLM with custom weights)
# ─────────────────────────────────────────────────────────────────────────────

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


def _remap_lamar_weights(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if k.startswith("esm.lm_head"):
            out[k[len("esm."):]] = v
        elif k.startswith("lm_head") or k.startswith("esm."):
            out[k] = v
        else:
            out["esm." + k] = v
    return out


def _load_weights(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        st = p / "model.safetensors"
        if st.exists():
            return load_file(str(st))
        pb = p / "pytorch_model.bin"
        if pb.exists():
            return torch.load(str(pb), map_location="cpu")
        raise FileNotFoundError(f"No weights file in {path}")
    # single file
    return load_file(str(p))


def load_lamar_model(spec: dict, tokenizer) -> EsmForMaskedLM:
    config = _build_lamar_config(tokenizer)
    model  = EsmForMaskedLM(config)
    raw    = _load_weights(spec["weights_path"])
    remapped = _remap_lamar_weights(raw)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    non_trivial = [k for k in missing if "lm_head" not in k]
    if non_trivial:
        print(f"  [WARN] missing (non-lm_head) keys: {non_trivial[:5]}")
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  DNABERT-2 model
# ─────────────────────────────────────────────────────────────────────────────

def _load_dnabert2_state_dict(path: str) -> dict:
    p = Path(path)
    state = None
    if p.is_dir():
        st = p / "model.safetensors"
        if st.exists():
            state = load_file(str(st))
        pb = p / "pytorch_model.bin"
        if state is None and pb.exists():
            state = torch.load(str(pb), map_location="cpu", weights_only=False)
            if isinstance(state, dict):
                state = state.get("state_dict", state.get("model", state))
        if state is not None:
            if any(k.startswith("bert.") for k in state):
                print("  [dnabert2] stripping 'bert.' prefix from state dict keys")
                state = {
                    (k[len("bert."):] if k.startswith("bert.") else k): v
                    for k, v in state.items()
                }
            if any(k.startswith("cls.") for k in state):
                print("  [dnabert2] dropping 'cls.' MLM-head keys for BertModel load")
                state = {k: v for k, v in state.items() if not k.startswith("cls.")}
            return state
        raise FileNotFoundError(f"No weights file in {path}")
    state = load_file(str(p))
    if any(k.startswith("bert.") for k in state):
        print("  [dnabert2] stripping 'bert.' prefix from state dict keys")
        state = {
            (k[len("bert."):] if k.startswith("bert.") else k): v
            for k, v in state.items()
        }
    if any(k.startswith("cls.") for k in state):
        print("  [dnabert2] dropping 'cls.' MLM-head keys for BertModel load")
        state = {k: v for k, v in state.items() if not k.startswith("cls.")}
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
            f"  [dnabert2] fallback to remote code for {weights_path}; missing local files: {missing_files}"
        )
        return True
    return False


def load_dnabert2_model(spec: dict, tokenizer) -> AutoModel:
    weights_path = spec["weights_path"]
    if _dnabert2_needs_remote_code_fallback(weights_path):
        config = AutoConfig.from_pretrained(spec["tokenizer"], trust_remote_code=True)
        with open(Path(weights_path) / "config.json") as f:
            local_cfg = json.load(f)
        for key, value in local_cfg.items():
            if key in {"auto_map", "_name_or_path", "transformers_version"}:
                continue
            try:
                object.__setattr__(config, key, value)
            except Exception:
                pass
        model = AutoModel.from_config(config, trust_remote_code=True)
        state_dict = _load_dnabert2_state_dict(weights_path)
        result = model.load_state_dict(state_dict, strict=False)
        non_trivial = [k for k in result.missing_keys if "position_ids" not in k]
        if non_trivial:
            print(f"  [WARN] missing keys after fallback load: {non_trivial[:5]}")
    else:
        model = AutoModel.from_pretrained(
            weights_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer loader
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(spec: dict):
    extra = {}
    if spec["type"] == "dnabert2":
        extra = {"use_fast": True, "trust_remote_code": True,
                 "padding_side": "right"}
    return AutoTokenizer.from_pretrained(
        spec["tokenizer"],
        model_max_length=spec["max_length"],
        **extra,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Embedding extraction  (last transformer layer, mean-pool over non-special tokens)
# ─────────────────────────────────────────────────────────────────────────────

def _mean_pool(hidden: torch.Tensor,
               attention_mask: torch.Tensor,
               input_ids: torch.Tensor,
               special_ids: set[int]) -> np.ndarray:
    """Mean-pool hidden states, masking out padding and special tokens."""
    valid = attention_mask.bool()
    if special_ids:
        for sid in special_ids:
            valid = valid & ~input_ids.eq(sid)
    denom  = valid.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
    pooled = (hidden.float() * valid.unsqueeze(-1).float()).sum(dim=1) / denom
    return pooled.cpu().numpy()


def _get_dnabert2_encoder_layer(model, layer_idx: int | None = None) -> torch.nn.Module:
    """Return BertLayer at layer_idx (None = last) of DNABERT2's encoder for hook attachment."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError("DNABERT2 model has no 'encoder' attribute")
    layers = getattr(encoder, "layer", None) or getattr(encoder, "layers", None)
    if layers is None:
        raise RuntimeError("Cannot find encoder.layer / encoder.layers on DNABERT2")
    idx = layer_idx if layer_idx is not None else len(layers) - 1
    return layers[idx], idx


def _uses_unpadded_arch(model) -> bool:
    """True if the model's encoder uses DNABERT2's unpadded FlashAttention path."""
    import inspect
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return False
    try:
        return "output_all_encoded_layers" in inspect.signature(encoder.forward).parameters
    except (ValueError, TypeError):
        return False


@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    seqs: list[str],
    model_type: str,
    max_length: int,
    batch_size: int = 32,
    layer_idx: int | None = None,
) -> np.ndarray:
    """Return float32 array [n_seqs, hidden_dim].

    layer_idx: transformer block index to pool (0 = embedding layer,
               1-12 = transformer blocks, None = last block).
    LAMAR: uses output_hidden_states=True on the ESM backbone.
    DNABERT2: uses a forward hook on the target BertLayer to capture hidden
              states — avoids the crash caused by output_hidden_states=True
              on DNABERT2's FlashAttention / unpadded architecture.
    """
    special_ids = set()
    for tid in (tokenizer.cls_token_id, tokenizer.sep_token_id,
                tokenizer.eos_token_id, tokenizer.bos_token_id):
        if tid is not None:
            special_ids.add(tid)

    # Pre-detect DNABERT2 unpadded architecture (do once, outside batch loop)
    unpadded = (model_type == "dnabert2") and _uses_unpadded_arch(model)

    if model_type == "dnabert2":
        target_layer, _resolved_idx = _get_dnabert2_encoder_layer(model, layer_idx)

    all_embs = []
    n_batches = (len(seqs) + batch_size - 1) // batch_size
    batch_iter = tqdm(
        range(0, len(seqs), batch_size),
        total=n_batches,
        desc=f"extract:{model_type}",
        unit="batch",
        leave=False,
    )
    for i in batch_iter:
        batch = seqs[i : i + batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        if model_type == "lamar":
            out = model.esm(**tokens, output_hidden_states=True)
            _li = layer_idx if layer_idx is not None else -1
            last_hidden = out.hidden_states[_li]   # (B, seq_len, H)

        else:  # dnabert2 — use a single hook on the target layer
            captured: list = []

            def _hook(_m, _inp, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                captured.append(h)

            handle = target_layer.register_forward_hook(_hook)
            try:
                model(**tokens)
            finally:
                handle.remove()

            raw = captured[0]  # (ntokens_unpad, H) or (B, seq_len, H)

            if unpadded:
                # Scatter unpadded tokens back into (B, seq_len, H)
                B, S = tokens["attention_mask"].shape
                H = raw.shape[-1]
                flat_mask = tokens["attention_mask"].bool().view(-1)
                buf = torch.zeros(B, S, H, device=raw.device, dtype=raw.dtype)
                buf.view(-1, H)[flat_mask] = raw
                last_hidden = buf
            else:
                last_hidden = raw  # already (B, seq_len, H)

        pooled = _mean_pool(
            last_hidden,
            tokens["attention_mask"],
            tokens["input_ids"],
            special_ids,
        )
        all_embs.append(pooled.astype(np.float32))

    return np.vstack(all_embs)


# ─────────────────────────────────────────────────────────────────────────────
#  Metric functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_rankme(M: np.ndarray, epsilon: float = 1e-7) -> float:
    _, sigma, _ = np.linalg.svd(M, full_matrices=False)
    pk = sigma / (sigma.sum() + epsilon)
    return float(np.exp(-np.sum(pk * np.log(pk + epsilon))))


def compute_stable_rank(M: np.ndarray) -> float:
    fro = float(np.linalg.norm(M, "fro"))
    two = float(np.linalg.norm(M, 2))
    return (fro ** 2) / (two ** 2) if two > 0 else 0.0

def compute_nesum(M: np.ndarray) -> float:
    C = np.cov(M, rowvar=False)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0, None)  # add this
    lam = np.sort(lam)[::-1]
    if lam[0] == 0:
        return 0.0
    return float(np.sum(lam / lam[0]))


def compute_isoscore(M: np.ndarray) -> float:
    """IsoScore.  Expects [n_seqs, hidden_dim].

    IsoScore internally calls PCA(n_components=n_features) on a
    [n_samples, n_features] matrix, requiring n_samples > n_features.
    When that condition is not met we PCA-reduce the feature space to
    (n_samples - 1) dimensions first, keeping n_samples > n_features_new.
    In production (thousands of sequences, 768 features) the fallback
    is never reached.
    """
    M = M.astype(np.float64)
    n_seqs, n_feat = M.shape
    if n_seqs <= n_feat:
        # Reduce to (n_seqs - 1) features so IsoScore's internal PCA is valid
        n_comp = max(2, n_seqs - 1)
        M = PCA(n_components=n_comp).fit_transform(M)  # [n_seqs, n_comp]
    pts = M.T  # [n_feat_or_reduced, n_seqs]
    try:
        return float(_IsoScore(pts))
    except Exception as e:
        print(f"    [IsoScore error] {e}")
        return float("nan")


def compute_pca_variance_ratio(M: np.ndarray, n: int = 2) -> tuple[float, ...]:
    pca = PCA(n_components=min(n, M.shape[1], M.shape[0]))
    pca.fit(M)
    return tuple(float(v) for v in pca.explained_variance_ratio_[:n])


def compute_all_metrics(M: np.ndarray) -> dict:
    # Center once here; compute_rankme and compute_stable_rank expect centered M.
    # compute_nesum uses np.cov (centers implicitly) and compute_isoscore handles
    # centering internally, so pass M unchanged to both.
    M_centered = M - M.mean(axis=0)
    t = {}
    t0 = time.time()
    t["RankMe"]    = compute_rankme(M_centered);           t["RankMe_sec"]      = time.time() - t0
    t0 = time.time()
    t["NESum"]     = compute_nesum(M);                    t["NESum_sec"]       = time.time() - t0
    t0 = time.time()
    t["StableRank"]= compute_stable_rank(M_centered);    t["StableRank_sec"]  = time.time() - t0
    t0 = time.time()
    t["IsoScore"]  = compute_isoscore(M);                t["IsoScore_sec"]    = time.time() - t0
    pca_vars = compute_pca_variance_ratio(M, n=2)
    t["PCA_Var1"] = pca_vars[0] if len(pca_vars) > 0 else float("nan")
    t["PCA_Var2"] = pca_vars[1] if len(pca_vars) > 1 else float("nan")
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _emb_cache_path(cache_dir: str, model_name: str, task_id: str) -> str:
    safe_task = task_id.replace("/", "__")
    return os.path.join(cache_dir, model_name, f"{safe_task}.npy")


def load_or_extract(
    cache_dir: str,
    model_name: str,
    task: dict,
    model,
    tokenizer,
    spec: dict,
    batch_size: int,
    force: bool,
    seqs: list[str] | None = None,
) -> np.ndarray:
    path = _emb_cache_path(cache_dir, model_name, task["task_id"])
    if not force and os.path.exists(path):
        return np.load(path)

    if seqs is None:
        seqs, _ = load_seqs_and_labels(
            task["rbp_dir"], ["train", "test"]
        )
    if not seqs:
        raise RuntimeError(f"No sequences for {task['task_id']}")

    layer_idx = spec.get("layer_idx", None)
    embs = extract_embeddings(
        model, tokenizer, seqs,
        model_type=spec["type"],
        max_length=spec["max_length"],
        batch_size=batch_size,
        layer_idx=layer_idx,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embs)
    return embs


# ─────────────────────────────────────────────────────────────────────────────
#  Sensitivity to sequence length
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    embeddings: np.ndarray,
    seqs: list[str],
    n_bins: int = 6,
) -> list[dict]:
    lengths = np.array([len(s) for s in seqs])
    df = pd.DataFrame({"length": lengths, "idx": np.arange(len(seqs))})
    try:
        df["bin"] = pd.qcut(df["length"], q=n_bins, duplicates="drop")
    except ValueError:
        return []

    rows = []
    for bin_label in df["bin"].cat.categories:
        idxs = df.index[df["bin"] == bin_label].tolist()
        if len(idxs) < 5:
            continue
        sub = embeddings[idxs]
        row = {
            "bin_label": str(bin_label),
            "n_seqs":    len(idxs),
            "length_mean": float(lengths[idxs].mean()),
        }
        row["RankMe"]     = compute_rankme(sub)
        row["NESum"]      = compute_nesum(sub)
        row["StableRank"] = compute_stable_rank(sub)
        row["IsoScore"]   = compute_isoscore(sub)
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

NICE_NAMES = {
    "lamar_pretrained":           "LAMAR pretrained",
    "lamar_tapt_1024":            "LAMAR TAPT-1024",
    "lamar_tapt_512":             "LAMAR TAPT-512",
    "dnabert2_pretrained":        "DNABERT2 pretrained",
    "dnabert2_tapt":              "DNABERT2 TAPT",
    "lamar_tapt_512_std":         "LAMAR TAPT-512-std",
    "dnabert2_tapt_v3":           "DNABERT2 TAPT-v3",
    # layer-6 variants
    "lamar_pretrained_layer6":    "LAMAR pretrained L6",
    "lamar_tapt_1024_layer6":     "LAMAR TAPT-1024 L6",
    "lamar_tapt_512_layer6":      "LAMAR TAPT-512 L6",
    "lamar_tapt_512_std_layer6":  "LAMAR TAPT-512-std L6",
    "dnabert2_pretrained_layer6": "DNABERT2 pretrained L6",
    "dnabert2_tapt_layer6":       "DNABERT2 TAPT L6",
    "dnabert2_tapt_v3_layer6":    "DNABERT2 TAPT-v3 L6",
}

METRIC_ORDER = ["IsoScore", "RankMe", "NESum", "StableRank"]


def _pivot(df: pd.DataFrame, metric: str):
    return df.pivot(index="model", columns="task_id", values=metric)


def _family_subset(df: pd.DataFrame, family_prefix: str) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df
    mask = df["model"].astype(str).str.startswith(family_prefix)
    return df[mask].copy()


def plot_heatmaps(df_metrics: pd.DataFrame, plots_dir: str, file_suffix: str = ""):
    """2×2 heatmap grid: IsoScore / RankMe / NESum / StableRank."""
    models = [m for m in NICE_NAMES if m in df_metrics["model"].unique()]
    tasks  = sorted(df_metrics["task_id"].unique())
    metrics = METRIC_ORDER

    fig, axes = plt.subplots(2, 2, figsize=(max(14, len(tasks) * 0.9 + 4), 9))
    cmaps = ["Blues", "YlGnBu", "YlOrRd", "PuBu"]

    for ax, metric, cmap in zip(axes.flat, metrics, cmaps):
        sub = df_metrics[df_metrics["model"].isin(models)].copy()
        sub["model_nice"] = sub["model"].map(NICE_NAMES)
        pivot = sub.pivot_table(index="model_nice", columns="task_id",
                                values=metric, aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                    linewidths=0.4, linecolor="white",
                    cbar_kws={"shrink": 0.8})
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.suptitle("Embedding Quality Metrics — Model × RBP Task", fontsize=14, y=1.01)
    plt.tight_layout()
    suffix = f"_{file_suffix}" if file_suffix else ""
    path = os.path.join(plots_dir, f"heatmaps_all_metrics{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


def plot_metric_boxplots(df_metrics: pd.DataFrame, plots_dir: str, file_suffix: str = ""):
    """Box/swarm plots: distribution across RBPs per model × metric."""
    models = [m for m in NICE_NAMES if m in df_metrics["model"].unique()]
    df = df_metrics[df_metrics["model"].isin(models)].copy()
    df["model_nice"] = df["model"].map(NICE_NAMES)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    palette = sns.color_palette("tab10", len(models))

    for ax, metric in zip(axes.flat, METRIC_ORDER):
        sns.boxplot(data=df, x="model_nice", y=metric, ax=ax,
                    palette=palette, width=0.5, linewidth=1.2)
        sns.stripplot(data=df, x="model_nice", y=metric, ax=ax,
                      color="black", alpha=0.55, size=3.5, jitter=True)
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    plt.suptitle("Embedding Quality Distribution Across RBP Tasks", fontsize=13)
    plt.tight_layout()
    suffix = f"_{file_suffix}" if file_suffix else ""
    path = os.path.join(plots_dir, f"boxplots_per_metric{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


def plot_radar(df_metrics: pd.DataFrame, plots_dir: str, file_suffix: str = ""):
    """Radar chart: mean-normalised metrics per model."""
    import math
    models = [m for m in NICE_NAMES if m in df_metrics["model"].unique()]
    metrics = METRIC_ORDER

    # Compute mean per model, normalise to [0,1] across models per metric
    pivot = (df_metrics[df_metrics["model"].isin(models)]
             .groupby("model")[metrics].mean())
    normed = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-12)

    N    = len(metrics)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7),
                           subplot_kw={"polar": True})
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for (model, row), col in zip(normed.iterrows(), colors):
        vals = row[metrics].tolist() + [row[metrics[0]]]
        ax.plot(angles, vals, linewidth=2, label=NICE_NAMES.get(model, model),
                color=col)
        ax.fill(angles, vals, alpha=0.1, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title("Embedding Quality Radar\n(min-max normalised across models)",
                 fontsize=12, pad=20)
    plt.tight_layout()
    suffix = f"_{file_suffix}" if file_suffix else ""
    path = os.path.join(plots_dir, f"radar_chart{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


def plot_sensitivity(df_sens: pd.DataFrame, plots_dir: str, file_suffix: str = ""):
    """Line plots: metric vs. mean sequence length bin per model."""
    if df_sens.empty:
        print("  [sensitivity] no data – skipping plot")
        return

    models = [m for m in NICE_NAMES if m in df_sens["model"].unique()]
    metrics_s = ["RankMe", "NESum", "StableRank", "IsoScore"]
    avail = [m for m in metrics_s if m in df_sens.columns]

    n_metrics = len(avail)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for ax, metric in zip(axes, avail):
        for model, col in zip(models, colors):
            sub = (df_sens[df_sens["model"] == model]
                   .sort_values("length_mean"))
            if sub.empty:
                continue
            ax.plot(sub["length_mean"], sub[metric], marker="o",
                    label=NICE_NAMES.get(model, model), color=col, linewidth=2)
        ax.set_xlabel("Mean sequence length (bin)", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"{metric} vs. Sequence Length", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Sensitivity to Sequence Length", fontsize=13)
    plt.tight_layout()
    suffix = f"_{file_suffix}" if file_suffix else ""
    path = os.path.join(plots_dir, f"sensitivity_seq_length{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


def plot_isoscore_bar(df_metrics: pd.DataFrame, plots_dir: str, file_suffix: str = ""):
    """Bar chart: mean IsoScore across all tasks per model."""
    models = [m for m in NICE_NAMES if m in df_metrics["model"].unique()]
    df = df_metrics[df_metrics["model"].isin(models)].copy()
    df["model_nice"] = df["model"].map(NICE_NAMES)

    agg = (df.groupby("model_nice")["IsoScore"]
             .agg(["mean", "std"])
             .reindex([NICE_NAMES[m] for m in models if m in NICE_NAMES]))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agg.index, agg["mean"],
                  yerr=agg["std"], capsize=5,
                  color=sns.color_palette("tab10", len(agg)),
                  width=0.6, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("IsoScore (mean ± std)", fontsize=11)
    ax.set_title("IsoScore per Model (averaged over all RBP tasks)",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.5,
               label="Perfect isotropy = 1")
    ax.legend(fontsize=9)
    plt.tight_layout()
    suffix = f"_{file_suffix}" if file_suffix else ""
    path = os.path.join(plots_dir, f"isoscore_bar{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


def plot_pca_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    task_id: str,
    plots_dir: str,
):
    """2-D PCA scatter coloured by label for one (model, task) pair."""
    pca  = PCA(n_components=2)
    X2d  = pca.fit_transform(embeddings)
    ev   = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl, color in zip([0, 1], ["steelblue", "tomato"]):
        mask = labels == lbl
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=color, s=6, alpha=0.4,
                   label=f"class {lbl} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=10)
    ax.set_title(f"PCA — {NICE_NAMES.get(model_name, model_name)}\n{task_id}",
                 fontsize=10)
    ax.legend(fontsize=8, markerscale=3)
    plt.tight_layout()
    safe_task = task_id.replace("/", "__")
    path = os.path.join(plots_dir, "pca_scatters",
                        f"{model_name}__{safe_task}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Unsupervised embedding-quality evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_roots",     nargs="+", default=_DATA_ROOTS)
    p.add_argument("--output_dir",     type=str,
                   default=str(_SCRIPT_DIR / "results" / "unsupervised_eval"))
    p.add_argument("--models",         nargs="+", default=list(MODEL_SPECS.keys()),
                   choices=list(MODEL_SPECS.keys()),
                   help="Which models to evaluate")
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--sensitivity_task", type=str, default=None,
                   help="task_id to use for seq-length sensitivity (default: first task)")
    p.add_argument("--n_sensitivity_tasks", type=int, default=3,
                   help="Number of tasks to run sensitivity on (if --sensitivity_task not set)")
    p.add_argument("--n_bins",         type=int, default=6)
    p.add_argument("--pca_scatter",    action="store_true",
                   help="Save per-(model, task) PCA scatter plots (many files)")
    p.add_argument("--force_reextract", action="store_true",
                   help="Re-extract embeddings even if cache exists")
    p.add_argument("--plots_only", action="store_true",
                   help="Skip evaluation and generate plots from existing metrics.csv/sensitivity.csv")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_dir    = Path(args.output_dir)
    cache_dir  = out_dir / "embedding_cache"
    plots_dir  = out_dir / "plots"
    for d in (out_dir, cache_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        metrics_csv = out_dir / "metrics.csv"
        sens_csv = out_dir / "sensitivity.csv"

        if not metrics_csv.exists():
            sys.exit(f"[ERROR] --plots_only requires existing file: {metrics_csv}")

        print(f"[plots_only] loading {metrics_csv}")
        try:
            df_metrics = pd.read_csv(metrics_csv)
        except pd.errors.EmptyDataError:
            sys.exit(f"[ERROR] metrics.csv is empty: {metrics_csv}")

        if sens_csv.exists():
            print(f"[plots_only] loading {sens_csv}")
            try:
                df_sens = pd.read_csv(sens_csv)
            except pd.errors.EmptyDataError:
                print(f"[plots_only] sensitivity.csv is empty at {sens_csv} — skipping sensitivity plots")
                df_sens = pd.DataFrame()
        else:
            print(f"[plots_only] no sensitivity.csv found at {sens_csv} — skipping sensitivity plots")
            df_sens = pd.DataFrame()

        print("\n[Plotting …]")
        if not df_metrics.empty:
            plot_heatmaps(df_metrics, str(plots_dir))
            plot_metric_boxplots(df_metrics, str(plots_dir))
            plot_radar(df_metrics, str(plots_dir))
            plot_isoscore_bar(df_metrics, str(plots_dir))

            for family_prefix, family_label in (("dnabert2", "DNABERT2"), ("lamar", "LAMAR")):
                df_metrics_family = _family_subset(df_metrics, family_prefix)
                if df_metrics_family.empty:
                    continue
                print(f"[Plotting … {family_label} variants only]")
                plot_heatmaps(df_metrics_family, str(plots_dir), file_suffix=f"{family_prefix}_only")
                plot_metric_boxplots(df_metrics_family, str(plots_dir), file_suffix=f"{family_prefix}_only")
                plot_radar(df_metrics_family, str(plots_dir), file_suffix=f"{family_prefix}_only")
                plot_isoscore_bar(df_metrics_family, str(plots_dir), file_suffix=f"{family_prefix}_only")

        if not df_sens.empty:
            plot_sensitivity(df_sens, str(plots_dir))
            for family_prefix in ("dnabert2", "lamar"):
                df_sens_family = _family_subset(df_sens, family_prefix)
                if not df_sens_family.empty:
                    plot_sensitivity(df_sens_family, str(plots_dir), file_suffix=f"{family_prefix}_only")

        print("\n[Done] Plotting finished from existing CSVs.")
        return

    tasks = discover_tasks(args.data_roots)
    print(f"\n[tasks] found {len(tasks)} RBP tasks across {len(args.data_roots)} roots")
    for t in tasks:
        print(f"  {t['task_id']}")

    if not tasks:
        sys.exit("[ERROR] No tasks found. Check --data_roots.")

    # ── Result containers ────────────────────────────────────────────────────
    all_metrics:  list[dict] = []
    all_sens:     list[dict] = []

    # ── Per-model loop ───────────────────────────────────────────────────────
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        print(f"\n{'='*65}")
        print(f"  Model: {model_name}  [{spec['type']}]  max_len={spec['max_length']}")
        print(f"{'='*65}")

        # Load tokenizer + model
        tokenizer = load_tokenizer(spec)
        if spec["type"] == "lamar":
            model = load_lamar_model(spec, tokenizer)
        else:
            model = load_dnabert2_model(spec, tokenizer)

        # Sensitivity tasks: pick first N that have enough length variance
        sensitivity_tasks = []
        if args.sensitivity_task:
            matched = [t for t in tasks if t["task_id"] == args.sensitivity_task]
            sensitivity_tasks = matched[:1]
        else:
            sensitivity_tasks = tasks[:args.n_sensitivity_tasks]
        # Build id-set once outside the task loop (fix: was comparing task dicts)
        sens_task_ids = {t["task_id"] for t in sensitivity_tasks}

        for task in tqdm(tasks, desc=model_name, unit="task"):
            print(f"\n  ── {task['task_id']}")

            # Load sequences once; reuse for extraction, sensitivity, and PCA.
            try:
                seqs, labels = load_seqs_and_labels(
                    task["rbp_dir"], ["train", "test"]
                )
            except Exception as e:
                print(f"    [ERROR] loading sequences: {e}")
                continue

            try:
                embs = load_or_extract(
                    str(cache_dir), model_name, task,
                    model, tokenizer, spec,
                    batch_size=args.batch_size,
                    force=args.force_reextract,
                    seqs=seqs,
                )
            except Exception as e:
                print(f"    [ERROR] embedding: {e}")
                continue

            # Sanity-check embeddings
            if np.any(np.isnan(embs)):
                print(f"    [WARN] embeddings contain NaN – skipping task")
                continue
            emb_norms = np.linalg.norm(embs, axis=1)
            print(
                f"    [emb] shape={embs.shape}  "
                f"norm mean={emb_norms.mean():.4f}  "
                f"min={emb_norms.min():.4f}  max={emb_norms.max():.4f}"
            )
            if emb_norms.mean() < 0.01:
                print(f"    [WARN] mean embedding norm < 0.01 – embeddings may be collapsed")

            # Global metrics
            try:
                m = compute_all_metrics(embs)
            except Exception as e:
                print(f"    [ERROR] metrics: {e}")
                continue

            m["model"]   = model_name
            m["task_id"] = task["task_id"]
            m["n_seqs"]  = len(embs)
            all_metrics.append(m)
            print(
                f"    IsoScore={m['IsoScore']:.3f}  "
                f"RankMe={m['RankMe']:.1f}  "
                f"NESum={m['NESum']:.3f}  "
                f"StableRank={m['StableRank']:.3f}"
            )

            # PCA scatter (optional) – reuse already-loaded seqs/labels
            if args.pca_scatter:
                if len(labels) == len(embs):
                    plot_pca_scatter(
                        embs, labels, model_name, task["task_id"],
                        str(plots_dir)
                    )

            # Sensitivity – reuse already-loaded seqs
            if task["task_id"] in sens_task_ids:
                rows = sensitivity_analysis(embs, seqs, n_bins=args.n_bins)
                for r in rows:
                    r["model"]   = model_name
                    r["task_id"] = task["task_id"]
                all_sens.extend(rows)

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── Save raw results ─────────────────────────────────────────────────────
    df_metrics = pd.DataFrame(all_metrics)
    metrics_csv = str(out_dir / "metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"\n[saved] metrics.csv  ({len(df_metrics)} rows)")

    df_sens = pd.DataFrame(all_sens)
    sens_csv = str(out_dir / "sensitivity.csv")
    df_sens.to_csv(sens_csv, index=False)
    print(f"[saved] sensitivity.csv  ({len(df_sens)} rows)")

    # ── Summary table ────────────────────────────────────────────────────────
    if not df_metrics.empty:
        summary = (df_metrics
                   .groupby("model")[METRIC_ORDER]
                   .agg(["mean", "std"])
                   .round(4))
        summary.to_csv(str(out_dir / "summary.csv"))
        print("\n[Summary — mean across all RBPs]\n")
        print(summary.to_string())

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n[Plotting …]")
    if not df_metrics.empty:
        plot_heatmaps(df_metrics, str(plots_dir))
        plot_metric_boxplots(df_metrics, str(plots_dir))
        plot_radar(df_metrics, str(plots_dir))
        plot_isoscore_bar(df_metrics, str(plots_dir))

        for family_prefix, family_label in (("dnabert2", "DNABERT2"), ("lamar", "LAMAR")):
            df_metrics_family = _family_subset(df_metrics, family_prefix)
            if df_metrics_family.empty:
                continue
            print(f"[Plotting … {family_label} variants only]")
            plot_heatmaps(df_metrics_family, str(plots_dir), file_suffix=f"{family_prefix}_only")
            plot_metric_boxplots(df_metrics_family, str(plots_dir), file_suffix=f"{family_prefix}_only")
            plot_radar(df_metrics_family, str(plots_dir), file_suffix=f"{family_prefix}_only")
            plot_isoscore_bar(df_metrics_family, str(plots_dir), file_suffix=f"{family_prefix}_only")
    if not df_sens.empty:
        plot_sensitivity(df_sens, str(plots_dir))
        for family_prefix in ("dnabert2", "lamar"):
            df_sens_family = _family_subset(df_sens, family_prefix)
            if not df_sens_family.empty:
                plot_sensitivity(df_sens_family, str(plots_dir), file_suffix=f"{family_prefix}_only")

    print("\n[Done] All results written to:", out_dir)
    print("  metrics.csv    – per-model × per-task scores")
    print("  summary.csv    – mean ± std per model")
    print("  sensitivity.csv– sequence-length sensitivity")
    print("  plots/         – PNG figures")


if __name__ == "__main__":
    main()
