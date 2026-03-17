#!/bin/bash
# Run the AUROC plot comparing OneHot (x-axis) vs LAMAR variants (y-axis)

RESULTS_DIR="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/results_clip_data"
PLOT_SCRIPT="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/plots/plot_auroc_onehot_vs_lamar.py"

python "$PLOT_SCRIPT" \
    --onehot "$RESULTS_DIR/OneHot_CNN_clip_results.csv" \
    --random "$RESULTS_DIR/LAMAR_Random_L11_results.csv" \
    --tapt "$RESULTS_DIR/LAMAR_TAPT_L11_results.csv" \
    --pretrained "$RESULTS_DIR/LAMAR_Pretrained_L11_results.csv" \
    --metric AUROC \
    --out "$RESULTS_DIR/auroc_onehot_vs_lamar_L11.png" \
    --figsize 10,10 \
    --dataset-label "CLIP"

echo "Plot saved to: $RESULTS_DIR/auroc_onehot_vs_lamar_L11.png"

# Also generate AUPR plot
python "$PLOT_SCRIPT" \
    --onehot "$RESULTS_DIR/OneHot_CNN_clip_results.csv" \
    --random "$RESULTS_DIR/LAMAR_Random_L11_results.csv" \
    --tapt "$RESULTS_DIR/LAMAR_TAPT_L11_results.csv" \
    --pretrained "$RESULTS_DIR/LAMAR_Pretrained_L11_results.csv" \
    --metric AUPR \
    --out "$RESULTS_DIR/aupr_onehot_vs_lamar_L11.png" \
    --figsize 10,10 \
    --dataset-label "CLIP"

echo "Plot saved to: $RESULTS_DIR/aupr_onehot_vs_lamar_L11.png"

# Also generate Accuracy plot
python "$PLOT_SCRIPT" \
    --onehot "$RESULTS_DIR/OneHot_CNN_clip_results.csv" \
    --random "$RESULTS_DIR/LAMAR_Random_L11_results.csv" \
    --tapt "$RESULTS_DIR/LAMAR_TAPT_L11_results.csv" \
    --pretrained "$RESULTS_DIR/LAMAR_Pretrained_L11_results.csv" \
    --metric Accuracy \
    --out "$RESULTS_DIR/accuracy_onehot_vs_lamar_L11.png" \
    --figsize 10,10 \
    --dataset-label "CLIP"

echo "Plot saved to: $RESULTS_DIR/accuracy_onehot_vs_lamar_L11.png"

# =========================
# Koo dataset (5 repeats per TF; averages handled in plot script)
# =========================
KOO_DIR="/gpfs/bwfor/work/ws/fr_ml642-thesis_work/Thesis/LAMAR/evalEmbeddings/results/results_koo_data"

python "$PLOT_SCRIPT" \
    --onehot "$KOO_DIR/OneHot_perf.csv" \
    --random "$KOO_DIR/LAMAR_layer11_random_perf.csv" \
    --tapt "$KOO_DIR/LAMAR_tapt_layer11_perf.csv" \
    --pretrained "$KOO_DIR/LAMAR_layer11_perf.csv" \
    --metric AUROC \
    --out "$KOO_DIR/auroc_onehot_vs_lamar_L11_koo.png" \
    --figsize 10,10 \
    --dataset-label "Koo"

echo "Plot saved to: $KOO_DIR/auroc_onehot_vs_lamar_L11_koo.png"

python "$PLOT_SCRIPT" \
    --onehot "$KOO_DIR/OneHot_perf.csv" \
    --random "$KOO_DIR/LAMAR_layer11_random_perf.csv" \
    --tapt "$KOO_DIR/LAMAR_tapt_layer11_perf.csv" \
    --pretrained "$KOO_DIR/LAMAR_layer11_perf.csv" \
    --metric AUPR \
    --out "$KOO_DIR/aupr_onehot_vs_lamar_L11_koo.png" \
    --figsize 10,10 \
    --dataset-label "Koo"

echo "Plot saved to: $KOO_DIR/aupr_onehot_vs_lamar_L11_koo.png"

python "$PLOT_SCRIPT" \
    --onehot "$KOO_DIR/OneHot_perf.csv" \
    --random "$KOO_DIR/LAMAR_layer11_random_perf.csv" \
    --tapt "$KOO_DIR/LAMAR_tapt_layer11_perf.csv" \
    --pretrained "$KOO_DIR/LAMAR_layer11_perf.csv" \
    --metric Accuracy \
    --out "$KOO_DIR/accuracy_onehot_vs_lamar_L11_koo.png" \
    --figsize 10,10 \
    --dataset-label "Koo"

echo "Plot saved to: $KOO_DIR/accuracy_onehot_vs_lamar_L11_koo.png"
