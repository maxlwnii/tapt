# Batch Job Scripts - Quick Start

## Available Scripts

| Script | Purpose | Time | When to use |
|--------|---------|------|------------|
| `run_layer_search.slurm` | Find best layer for TAPT LAMAR | 6h | 🟢 **START HERE** |
| `run_layer_search_test.slurm` | Quick test (1 RBP, 300 samples) | 30m | Debug/verify setup |
| `run_full_evaluation_with_best_layer.slurm` | Evaluate all models with best layer | 20h | After layer search completes |
| `run_embedding_comparison.slurm` | Compare all embedding models | 24h | Full model comparison |
| `run_layer_search_workflow.sh` | Automated workflow (experimental) | — | Auto-chain jobs |

## Quick Start (Recommended Path)

### 1️⃣ Test Your Setup (Optional but recommended)
```bash
cd /home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/evalEmbeddings
sbatch run_layer_search_test.slurm
# Wait ~30 minutes
tail -f logs/layer_search_test_*.out
```

### 2️⃣ Find the Best Layer ⭐
```bash
sbatch run_layer_search.slurm
# Wait ~6 hours
tail -f logs/layer_search_tapt_*.out
```

### 3️⃣ Check Results
After layer search completes:
```bash
# Find the output directory
ls -lt results/linear_probe_embedding_quality/ | head -1

# View the best layer
cat results/linear_probe_embedding_quality/layer_search_tapt_*/layer_search.json

# Note the "best_layer" value (e.g., 6)
```

### 4️⃣ Run Full Evaluation
Edit `run_full_evaluation_with_best_layer.slurm`:
```bash
nano run_full_evaluation_with_best_layer.slurm
```

Find ~line 25 and update:
```bash
BEST_LAYER=6    # <-- Change this to your best layer value
```

Then submit:
```bash
sbatch run_full_evaluation_with_best_layer.slurm
# Wait ~20 hours
tail -f logs/emb_probe_all_*.out
```

### 5️⃣ View Results
```bash
cat results/.../summary_metrics.csv
# Shows: Model, Mean AUROC, Std AUROC, Mean F1, Std F1, Mean AUPRC, Std AUPRC
```

---

## Important Notes

### ⚠️ Before running:
- Ensure environment is set up: `conda activate /home/fr/fr_fr/fr_ml642/.conda/envs/dnabert2`
- Check GPU availability: `nvidia-smi`
- Verify data paths exist in scripts

### 📝 Editing scripts:
- Use `nano` or `vi` to edit
- Only change `BEST_LAYER` value in full evaluation script
- Do NOT modify SLURM headers unless you know what you're doing

### 🗂️ Output structure:
```
results/linear_probe_embedding_quality/
├── layer_search_tapt_<timestamp>_<JOBID>/     ← Layer search output
│   ├── layer_search.json                      ← Extract best_layer here
│   ├── layer_auroc_curve.png
│   └── per_rbp_metrics.csv
│
├── emb_probe_all_layer<N>_<timestamp>_<JOBID>/  ← Full evaluation output
│   ├── summary_metrics.csv
│   ├── per_rbp_metrics.csv
│   ├── plots/
│   └── statistical_tests.json
│
└── cache/
    └── layer_search/                          ← Cached embeddings
        └── model_<hash>/
            ├── layer_00/
            ├── layer_01/
            └── ...
```

---

## One-Liner Commands

### Submit test run
```bash
sbatch run_layer_search_test.slurm
```

### Submit layer search
```bash
sbatch run_layer_search.slurm
```

### Get best layer from results
```bash
grep best_layer results/*/layer_search.json | head -1
```

### Edit & submit full evaluation (in one command)
```bash
BEST_LAYER=6 && sed -i "s/BEST_LAYER=.*/BEST_LAYER=$BEST_LAYER/" run_full_evaluation_with_best_layer.slurm && sbatch run_full_evaluation_with_best_layer.slurm
```

### Monitor running job
```bash
watch -n 10 squeue | grep $USER
```

### Check results summary
```bash
tail results/emb_probe_all_*/summary_metrics.csv
```

---

## Troubleshooting

**"Job fails immediately with Python error"**
→ Try test script first: `sbatch run_layer_search_test.slurm`

**"CUDA out of memory"**
→ Reduce batch_size in script (line with `--batch_size`)

**"Can't find output directory"**
→ Check logs: `ls -la logs/layer_search_tapt_*.err`

**"Module not found"**
→ Verify conda activation in terminal

**"Better documentation needed"**
→ See: `SLURM_JOBS_README.md`, `LAYER_SEARCH_README.md`, `LAYER_SEARCH_QUICK_REF.md`

---

## File Reference

All scripts located in: `/home/fr/fr_fr/fr_ml642/Thesis/DNABERT2/evalEmbeddings/`

### Batch scripts:
- `run_layer_search.slurm` - Main layer search
- `run_layer_search_test.slurm` - Quick test  
- `run_full_evaluation_with_best_layer.slurm` - Full evaluation
- `run_embedding_comparison.slurm` - Model comparison
- `run_layer_search_workflow.sh` - Workflow automation

### Documentation:
- `SLURM_JOBS_README.md` - Complete SLURM documentation
- `LAYER_SEARCH_README.md` - Layer search details
- `LAYER_SEARCH_QUICK_REF.md` - Quick reference
- `THIS_FILE` - Quick start guide

### Code:
- `layer_search.py` - Layer search implementation
- `linear_probe_embedding_quality.py` - Main evaluation code

---

## Example Output

After full evaluation completes, you should see:

```
========== Summary Metrics ==========
Model                Mean AUROC  Std AUROC  Mean F1   Std F1    Mean AUPRC  Std AUPRC
tapt_lamar           0.8471      0.0892     0.7563    0.1124    0.6789      0.1456
pretrained_lamar     0.8234      0.0945     0.7234    0.1189    0.6456      0.1523
base_dnabert2        0.7650      0.1023     0.6234    0.1342    0.5123      0.1678
one_hot              0.6234      0.1456     0.4567    0.1789    0.3456      0.1945
```

✅ TAPT LAMAR with best layer (layer 6) outperforms other methods!

---

## Next Steps After Evaluation

1. **Visualize results**: Check `plots/` directory for AUROC by RBP, boxplots, embeddings
2. **Compare RBPs**: Identify which RBPs benefit most from each model
3. **Statistical analysis**: Check `statistical_tests.json` for significance
4. **Fine-tune**: Consider re-running with different RBPs or hyperparameters if needed

---

For detailed documentation, see the other README files in this directory.
