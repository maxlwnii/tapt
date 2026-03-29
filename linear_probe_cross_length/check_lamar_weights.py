#!/usr/bin/env python3
"""Check LAMAR weights loading and report missing/unexpected keys.

Usage: python check_lamar_weights.py
"""
from pathlib import Path
import importlib.util
import sys

repo_root = Path(__file__).resolve().parents[1]
module_path = repo_root / "linear_probe_cross_length" / "linear_probe_cross_length.py"
spec = importlib.util.spec_from_file_location("lpc", str(module_path))
lpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpc)

from transformers import AutoTokenizer
import torch

# collect LAMAR model specs from the module
model_defaults = getattr(lpc, "_MODEL_DEFAULTS", None)
if model_defaults is None:
    print("Cannot find _MODEL_DEFAULTS in linear_probe_cross_length.py")
    raise SystemExit(1)

lamar_models = {k: v for k, v in model_defaults.items() if v.get("type") == "lamar"}
print(f"Found {len(lamar_models)} LAMAR model specs")

for name, specm in lamar_models.items():
    wpath = specm.get("weights_path")
    print("\n" + "-" * 60)
    print(f"Model: {name}")
    print(f"Weights path: {wpath}")
    if wpath == "__random__":
        print("  [info] random init sentinel, skipping weight file check")
        continue

    # Prefer module helper; it handles directory checkpoints correctly.
    raw = None
    wpath_obj = Path(wpath)
    used_helper = False
    try:
        loader = getattr(lpc, "_load_weights_file", None)
        if loader:
            raw = loader(wpath)
            used_helper = True
            print("  loaded raw weights using module helper")
    except Exception as e:
        print("  helper _load_weights_file failed:", e)

    if raw is None:
        if wpath_obj.is_dir():
            print("  [ERROR] weights path is a directory; helper must succeed for this format")
            continue
        # try safetensors
        try:
            from safetensors.torch import load_file as st_load
            raw = st_load(wpath)
            print("  loaded raw weights with safetensors.torch.load_file")
        except Exception:
            try:
                raw = torch.load(wpath, map_location="cpu")
                print("  loaded raw weights with torch.load")
            except Exception as e:
                print("  failed to load weights file:", e)
                continue

    total_keys = len(raw.keys())
    print(f"  [raw] total keys in file: {total_keys}")
    sample_keys = list(raw.keys())[:15]
    print("  [sample keys]", sample_keys)

    # remap if helper exists
    remap = getattr(lpc, "_remap_lamar_weights", None)
    if remap:
        try:
            remapped = remap(raw)
            print(f"  remapped keys -> {len(remapped.keys())} keys")
        except Exception as e:
            print("  remap failed:", e)
            remapped = raw
    else:
        remapped = raw

    # try to build config and model
    try:
        # need tokenizer
        tokenizer = AutoTokenizer.from_pretrained(specm.get("tokenizer"), use_fast=True)
        cfg = lpc._build_lamar_config(tokenizer)
        model = lpc.EsmForMaskedLM(cfg)
        # load state dict
        # convert to torch tensors if necessary
        state_dict = {k: (torch.as_tensor(v) if not hasattr(v, "dtype") else v) for k, v in remapped.items()}
        result = model.load_state_dict(state_dict, strict=False)
        if hasattr(result, "missing_keys"):
            missing = result.missing_keys
        elif isinstance(result, dict):
            missing = result.get("missing_keys", [])
        else:
            missing = []

        if hasattr(result, "unexpected_keys"):
            unexpected = result.unexpected_keys
        elif isinstance(result, dict):
            unexpected = result.get("unexpected_keys", [])
        else:
            unexpected = []
        model_keys = len(model.state_dict())
        loaded_keys = model_keys - len(missing)
        print(f"  [weight check] {loaded_keys}/{model_keys} model keys received weights")
        print(f"  [weight check] {len(missing)} missing, {len(unexpected)} unexpected")
        print(f"  [weight check] {len(unexpected)} unexpected keys in file (not used)")
        non_trivial = [k for k in missing if "lm_head" not in k and "classifier" not in k]
        if non_trivial:
            print("  [WARN] ALL non-trivial missing:")
            for k in non_trivial:
                print("    ", k)
        if unexpected:
            print("  [WARN] unexpected keys sample:", unexpected[:15])
    except Exception as e:
        print("  Failed to instantiate model and load state_dict:", e)

print("\nDone.")
