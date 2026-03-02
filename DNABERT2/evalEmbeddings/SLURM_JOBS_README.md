# SLURM Batch Scripts for Layer Search

This directory contains SLURM batch scripts for running the layer search and evaluation pipeline on HPC clusters.

## Available Scripts

### 1. `run_layer_search.slurm` - Discover Best Layer ⭐ START HERE
**Purpose**: Find the optimal transformer layer for TAPT LAMAR model via linear probing.

**Time**: ~6 hours (includes all hidden layers)  
**GPU**: 1× A40 (64GB memory)  
**Settings**: Full dataset, all RBPs

**Submit**:
```bash
sbatch run_layer_search.slurm
```

**Output**:
- `layer_search.json` - Best layer index and per-layer AUROC scores
- `layer_auroc_curve.png` - Visualization of AUROC across layers

**Next Step**: Extract the `best_layer` value from `layer_search.json` and use it in the next script.

---

### 2. `run_full_evaluation_with_best_layer.slurm` - Full Evaluation
**Purpose**: Evaluate all embedding models (one_hot, base_dnabert2, tapt_lamar, pretrained_lamar) using the best layer found in Step 1.

**Time**: ~20 hours (all RBPs, all models)  
**GPU**: 1× A40 (64GB memory)  
**Settings**: Full dataset, all embedding models

**Prerequisite**: Must run `run_layer_search.slurm` first!

**Setup**:
1. Extract best layer from `layer_search.json`:
   ```bash
   grep best_layer results/linear_probe_embedding_quality/*/layer_search.json
   ```

2. Edit `run_full_evaluation_with_best_layer.slurm` line ~25:
   ```bash
   # OLD:
   BEST_LAYER=6
   
   # NEW (example - use your actual best layer):
   BEST_LAYER=7
   ```

3. Submit:
   ```bash
   sbatch run_full_evaluation_with_best_layer.slurm
   ```

**Output**:
- `per_rbp_metrics.csv` - Per-RBP results for all models
- `summary_metrics.csv` - Summary statistics
- `auroc_boxplot.png` - Distribution plot
- `per_rbp_auroc_bars.png` - Per-RBP comparison
- `embedding_health_stats.csv` - Embedding quality metrics

---

### 3. `run_layer_search_test.slurm` - Quick Test Run
**Purpose**: Quick smoke test on 1 RBP with limited samples to verify setup.

**Time**: ~30 minutes (1 RBP only, 300 samples, 3-fold CV)  
**GPU**: 1× A40 (32GB memory)  
**Settings**: Reduced dataset for faster iteration

**Submit**:
```bash
sbatch run_layer_search_test.slurm
```

**Use case**: Debugging, verifying CUDA setup, testing environment before full runs.

---

### 4. `run_layer_search_workflow.sh` - Automated Workflow (Experimental)
**Purpose**: Automate the full workflow from layer search to evaluation.

**Usage**:
```bash
bash run_layer_search_workflow.sh
```

**What it does**:
1. Submits layer search job
2. Waits for completion
3. Extracts best layer from results
4. Prompts you to run full evaluation with that layer
5. (Optional) Auto-submits full evaluation job

**Note**: Requires interactive editing or uncomment automation lines.

---

## Recommended Workflow

```
┌─────────────────────────────────┐
│  run_layer_search.slurm         │ (6 hours)
│  ↓ Outputs: layer_search.json ↓ │
├─────────────────────────────────┤
│                                 │
│  Extract best_layer value       │
│  Edit: run_full_evaluation...   │
│  ↓                              │
├─────────────────────────────────┤
│ run_full_evaluation_...slurm    │ (20 hours)
│  ↓ Outputs: summary metrics ↓   │
├─────────────────────────────────┤
│  Analyze results & plots        │
└─────────────────────────────────┘
```

## Command Reference

### Submit a job
```bash
sbatch run_layer_search.slurm
```

### Check job status
```bash
squeue -u $USER
# or for specific job:
squeue -j <SLURM_JOB_ID>
```

### View job output in real-time
```bash
tail -f logs/layer_search_tapt_*.out
```

### Cancel a job
```bash
scancel <SLURM_JOB_ID>
```

### View completed job info
```bash
sinfo  # show node status
sacct # show completed jobs
```

### Chain jobs (run one after another)
```bash
# Submit layer search
JOB1=$(sbatch run_layer_search.slurm | awk '{print $4}')

# Submit full evaluation after layer search completes
# (this is what run_layer_search_workflow.sh does)
sbatch --dependency=afterok:$JOB1 run_full_evaluation_with_best_layer.slurm
```

## Output Directory Structure

Results are organized by timestamp and job ID:

```
DNABERT2/evalEmbeddings/results/linear_probe_embedding_quality/
├── layer_search_tapt_20260226_143521_12345678/
│   ├── layer_search.json                 ← Extract best_layer from here
│   ├── layer_auroc_curve.png
│   ├── per_rbp_metrics.csv
│   └── plots/
│
├── emb_probe_all_layer6_20260226_145000_12345679/
│   ├── per_rbp_metrics.csv
│   ├── summary_metrics.csv
│   ├── embedding_health_stats.csv
│   ├── plots/
│   │   ├── auroc_boxplot.png
│   │   ├── per_rbp_auroc_bars.png
│   │   └── ...
│   └── statistical_tests.json
│
└── cache/
    └── layer_search/
        └── model_<hash>/
            ├── layer_00/
            ├── layer_01/
            └── ... (cached embeddings for fast re-runs)
```

## Tips & Tricks

### Find best layer quickly
```bash
find results -name "layer_search.json" -exec grep best_layer {} \;
```

### Compare results across runs
```bash
ls -ldt results/linear_probe_embedding_quality/emb_probe_all*
```

### Clean up old results (careful!)
```bash
# Show what would be deleted
find results -type d -name "*layer_search*" -mtime +30

# Actually delete (older than 30 days)
find results -type d -name "*layer_search*" -mtime +30 -exec rm -rf {} \;
```

### Monitor long-running job
```bash
# Check GPU usage
watch -n 2 'squeue -j <JOB_ID> && nvidia-smi'

# Or via srun if job is running
srun --jobid=<JOB_ID> nvidia-smi
```

### Re-run with different settings
Edit the script and change:
- `--max_rbps` (number of RBPs to evaluate)
- `--max_samples_per_rbp` (samples per RBP)
- `--batch_size` (latency/throughput tradeoff)
- `--num_folds` (more folds = slower but more robust CV)

## Troubleshooting

**Job starts but Python crashes immediately**
- Check conda environment: `conda info --envs`
- Verify CUDA: `nvidia-smi`
- Run test script first: `sbatch run_layer_search_test.slurm`

**"ModuleNotFoundError: No module named"**
- Activate conda environment manually: `source activate /path/to/env`
- Install missing packages in that environment

**CUDA out of memory**
- Reduce `--batch_size` to 32 or 16
- Reduce `--max_length` if appropriate

**Job kills before completion**
- Increase `--time` parameter (e.g., 20:00:00 = 20 hours)
- Check memory: `--mem=64gb` should be sufficient

**Slow layer extraction**
- Normal for full dataset; layer search extracts all layers in one pass
- Test run script uses 10% of data for faster iteration

## Environment Variables

The scripts set:
```bash
export PYTHONPATH="$BASE:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false    # Avoid parallelism warnings
export OMP_NUM_THREADS=8                # Set thread count explicitly
```

These are pre-configured; usually no changes needed.

## Getting Help

### View script content
```bash
cat run_layer_search.slurm
```

### Check what job is about to run
```bash
sbatch --test-only run_layer_search.slurm
```

### For detailed documentation
See `LAYER_SEARCH_README.md` and `LAYER_SEARCH_QUICK_REF.md` in this directory.
