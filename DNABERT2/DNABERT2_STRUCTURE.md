# DNABERT2 Directory Structure Documentation

## Overview
The `DNABERT2/` directory contains scripts and resources for fine-tuning and evaluating DNABERT2 models on RNA-binding protein (RBP) binding site classification tasks. DNABERT2 is a transformer-based language model pre-trained on genomic sequences.

---

## Directory Structure

```
DNABERT2/
├── train.py                    # Main training script
├── convert_fasta_to_csv.py     # Single dataset converter
├── convert_datasets_to_csv.py  # Batch converter
├── visualisation.py            # Interpretation methods
├── plot_eval_scatter.py        # Results visualization
├── finetune.sh                 # Training wrapper
├── generate_dnabert2_jobs.sh   # SLURM job generator
├── environment.yml             # Conda environment
├── README.md                   # Documentation
│
├── data/                       # Training data (CSV format)
│   ├── GTF2F1_K562_IDR/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   └── ... (8 RBPs total)
│
├── output/                     # Fine-tuned models
│   ├── dnabert2_GTF2F1_K562_IDR__fold5_fold5/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── eval_results.json
│   │   └── training_args.bin
│   ├── dnabert2_GTF2F1_K562_IDR__fold5_random_fold5/
│   └── ... (16 total: 8 RBPs × 2 conditions)
│
├── logs_dnabert2/             # SLURM job scripts
│   ├── dnabert2_GTF2F1_K562_IDR_fold5.slurm
│   ├── dnabert2_GTF2F1_K562_IDR_fold5_random.slurm
│   └── ... (16 total scripts)
│
├── images/                    # Visualization outputs
└── pretrain/                  # Pre-training configs
    ├── dnabert2_pretrain.yml
    ├── dnabert2_requirements.txt
    └── scripts/
```

---

## Core Components

### 1. **Training & Fine-tuning**

#### **train.py** - Main training script (411 lines)

**Key Classes:**

```python
@dataclass
class ModelArguments:
    model_name_or_path: str = "facebook/opt-125m"
    use_lora: bool = False              # LoRA fine-tuning
    use_random_init: bool = False       # Random initialization baseline
    lora_r: int = 8                     # LoRA rank
    lora_alpha: int = 32               
    lora_dropout: float = 0.05
    lora_target_modules: str = "query,value"

@dataclass
class DataArguments:
    data_path: str = None
    kmer: int = -1                      # K-mer tokenization (-1 = no k-mer)
    fold: int = -1                      # Cross-validation fold number
    cv_folds: int = 5                   # Number of CV folds

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = 512
    per_device_train_batch_size: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    early_stopping_patience: int = 3
    metric_for_best_model: str = "eval_accuracy"
```

**Key Functions:**

```python
def init_weights(module):
    """WOLF random initialization for fair baseline comparison"""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    # ... LayerNorm, Embedding initialization

def calculate_metric_with_sklearn(predictions, labels, probabilities):
    """Compute comprehensive metrics"""
    metrics = {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro"),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro"),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro"),
    }
    # Binary classification AUC/AUPRC
    if probabilities is not None and probabilities.shape[1] == 2:
        pos_prob = probabilities[:, 1]
        metrics["auc"] = sklearn.metrics.roc_auc_score(valid_labels, pos_prob)
        metrics["auprc"] = sklearn.metrics.average_precision_score(valid_labels, pos_prob)
    return metrics

def compute_metrics(eval_pred):
    """Metric computation for Trainer API"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    return calculate_metric_with_sklearn(predictions, labels, probabilities)
```

**Cross-Validation Support:**
```python
# K-fold stratified cross-validation
if data_args.fold != -1:
    skf = StratifiedKFold(n_splits=data_args.cv_folds, shuffle=True, random_state=42)
    folds = list(skf.split(all_texts, all_labels))
    
    # Fold assignment: test, val, train
    test_fold = data_args.fold - 1
    val_fold = data_args.fold % data_args.cv_folds
    train_folds = [i for i in range(cv_folds) if i not in [test_fold, val_fold]]
```

**Dataset Classes:**
```python
class SupervisedDataset(Dataset):
    """Dataset for sequence classification"""
    
    def __init__(self, data_path, tokenizer, kmer=-1):
        # Load CSV data
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]  # Skip header
        
        # Extract sequences and labels
        texts = [d[0] for d in data]
        labels = [int(d[1]) for d in data]
        
        # Optional k-mer tokenization
        if kmer != -1:
            texts = load_or_generate_kmer(data_path, texts, kmer)
        
        # Tokenize
        output = tokenizer(texts, padding="longest", 
                          max_length=tokenizer.model_max_length,
                          truncation=True)
        self.input_ids = output["input_ids"]
        self.labels = labels
```

---

### 2. **Data Preprocessing**

#### **convert_fasta_to_csv.py** - FASTA to CSV converter

```python
def parse_fasta_file(filepath, label):
    """
    Parse FASTA file and assign labels
    
    Args:
        filepath: Path to .fa file
        label: 0 (negative) or 1 (positive)
    
    Returns:
        List of (sequence, label) tuples
    """
    sequences = []
    current_sequence = ""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>'):  # Header
            if current_sequence:
                # RNA to DNA conversion
                sequences.append((current_sequence.replace('U', 'T'), label))
                current_sequence = ""
            i += 1
            # Read sequence lines
            while i < len(lines) and not lines[i].strip().startswith('>'):
                current_sequence += lines[i].strip()
                i += 1
        else:
            i += 1
    return sequences

def create_csv_files():
    """Create train/dev/test CSVs with 70/15/15 split"""
    negative_sequences = parse_fasta_file('data/negatives.fa', 0)
    positive_sequences = parse_fasta_file('data/positives.fa', 1)
    
    all_sequences = negative_sequences + positive_sequences
    df = pd.DataFrame(all_sequences, columns=['sequence', 'label'])
    
    # Stratified split
    train_df, temp_df = train_test_split(df, test_size=0.3, 
                                         random_state=42, 
                                         stratify=df['label'])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                       random_state=42, 
                                       stratify=temp_df['label'])
    
    # Save CSVs
    train_df.to_csv('data/train.csv', index=False)
    dev_df.to_csv('data/dev.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
```

#### **convert_datasets_to_csv.py** - Batch conversion for multiple RBPs

Converts all RBP datasets from `clip_training_data_uhl/` to CSV format in `data/` subdirectories.

---

### 3. **Visualization & Interpretation**

#### **visualisation.py** - Model interpretation methods

**1. In-Silico Mutagenesis**
```python
def in_silico_mutagenesis(self, sequence):
    """
    Test effect of single nucleotide mutations
    
    Returns:
        DataFrame (L x 4) with mutation effects
    """
    seq = sequence.upper()
    L = len(seq)
    effect = pd.DataFrame(0.0, index=range(L), columns=BASES)
    orig_prob = self.get_prob(seq)
    
    for pos in range(L):
        orig_nt = seq[pos]
        for new_nt in BASES:
            if new_nt != orig_nt:
                mutated_seq = seq[:pos] + new_nt + seq[pos+1:]
                new_prob = self.get_prob(mutated_seq)
                effect.loc[pos, new_nt] = new_prob - orig_prob
    return effect
```

**2. Gradient-Based Saliency**
```python
def gradient_saliency(self, sequence):
    """
    Compute gradients of prediction w.r.t. embeddings
    
    Returns:
        Saliency scores for each position
    """
    inputs = self._encode(sequence)
    inputs = _to_device(inputs, self.device)
    
    # Get embeddings and enable gradient tracking
    embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
    embeddings.requires_grad = True
    
    # Forward pass
    logits = self.model(inputs_embeds=embeddings, 
                       attention_mask=inputs['attention_mask']).logits
    prob = torch.softmax(logits, dim=-1)[0, 1]  # Positive class
    
    # Backward pass
    prob.backward()
    
    # L2 norm of gradients
    saliency = torch.norm(embeddings.grad, dim=-1).squeeze().detach().cpu().numpy()
    return saliency
```

**3. Integrated Gradients**
```python
def integrated_gradients(self, sequence, n_steps=50):
    """
    Path integral from baseline to input
    More accurate attribution than simple gradients
    """
    # Baseline: zero embeddings
    baseline = torch.zeros_like(embeddings)
    
    # Interpolate between baseline and input
    integrated_grads = 0
    for step in range(n_steps):
        alpha = step / n_steps
        interpolated = baseline + alpha * (embeddings - baseline)
        interpolated.requires_grad = True
        
        # Compute gradient at this point
        logits = self.model(inputs_embeds=interpolated, ...).logits
        prob = torch.softmax(logits, dim=-1)[0, 1]
        prob.backward()
        
        integrated_grads += interpolated.grad
    
    # Average and multiply by (input - baseline)
    integrated_grads = integrated_grads / n_steps * (embeddings - baseline)
    return integrated_grads
```

**4. Sliding Window Mutagenesis**
```python
def sliding_window_mutagenesis(self, sequence, window_size=5):
    """
    Mutate contiguous windows to find important regions
    """
    L = len(sequence)
    window_effects = []
    
    for start in range(L - window_size + 1):
        # Mutate entire window
        mutated_seq = (sequence[:start] + 
                      'N' * window_size + 
                      sequence[start+window_size:])
        effect = self.get_prob(mutated_seq) - orig_prob
        window_effects.append((start, effect))
    
    return window_effects
```

---

### 4. **Job Management & Scheduling**

#### **generate_dnabert2_jobs.sh** - SLURM job generator

```bash
# RBP configuration
declare -A SEQ_LENGTHS=(
    ["GTF2F1_K562_IDR"]=101
    ["HNRNPL_K562_IDR"]=101
    # ... 8 RBPs total
)

# SLURM template
TEMPLATE="#!/bin/bash
#SBATCH --job-name=dnabert2_RBP_NAME_SUFFIX
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:A40:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G

# Activate environment
conda activate dnabert2

# Set variables
export DATA_PATH=./data/RBP_NAME
export MAX_LENGTH=MAX_LEN
export LR=3e-5

# Run training
python train.py \\
    --model_name_or_path zhihan1996/DNABERT-2-117M \\
    --data_path \$DATA_PATH \\
    --kmer -1 \\
    --run_name dnabert2_RBP_NAME_SUFFIX \\
    --model_max_length \$MAX_LENGTH \\
    --per_device_train_batch_size 8 \\
    --per_device_eval_batch_size 16 \\
    --num_train_epochs 5 \\
    --learning_rate \$LR \\
    --output_dir ./output/dnabert2_RBP_NAME_SUFFIX \\
    --early_stopping_patience 3 \\
    --metric_for_best_model eval_auprc
"
```

**Generates:**
- Normal fine-tuning jobs (pre-trained initialization)
- Random initialization jobs (baseline comparison)
- One SLURM script per RBP per condition

#### **finetune.sh** - Simple training wrapper
```bash
#!/bin/bash
python train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path ./data/GTF2F1_K562_IDR \
    --run_name dnabert2_GTF2F1 \
    --model_max_length 101 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --learning_rate 3e-5 \
    --output_dir ./output/dnabert2_GTF2F1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --load_best_model_at_end True
```

---

### 5. **Results Analysis**

#### **plot_eval_scatter.py** - Performance visualization

```python
def parse_results(paths):
    """Extract metrics from eval_results.json files"""
    normal_auc = {}
    random_auc = {}
    normal_auprc = {}
    random_auprc = {}
    
    for p in paths:
        with open(p, 'r') as f:
            data = json.load(f)
        
        auc = data.get('eval_auc') or data.get('eval_roc_auc')
        auprc = data.get('eval_auprc')
        
        # Parse model name from path
        is_random = 'random' in model_dir
        rbp = model_dir.replace('dnabert2_', '').split('__fold5')[0]
        
        if is_random:
            random_auc[rbp] = float(auc)
            random_auprc[rbp] = float(auprc)
        else:
            normal_auc[rbp] = float(auc)
            normal_auprc[rbp] = float(auprc)
    
    return normal_auc, random_auc, normal_auprc, random_auprc

def plot_scatter(metric_name, normal, random, outpath):
    """Create scatter plot: random vs pretrained performance"""
    keys = sorted(set(normal.keys()) & set(random.keys()))
    
    xs = [random[k] for k in keys]  # Random init
    ys = [normal[k] for k in keys]  # Pre-trained
    
    plt.figure(figsize=(6,6))
    for k in keys:
        plt.scatter(random[k], normal[k], s=100)
        plt.text(random[k], normal[k], k.split('_')[0], fontsize=8)
    
    # Diagonal line (y=x)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel(f'Random Init {metric_name}')
    plt.ylabel(f'Pre-trained {metric_name}')
    plt.title(f'{metric_name}: Pre-trained vs Random')
    plt.savefig(outpath)
```

**Output:** Scatter plots comparing:
- AUC: Pre-trained vs Random
- AUPRC: Pre-trained vs Random

---

## Usage Workflow

### 1. Data Preparation
```bash
# Convert FASTA to CSV for all RBPs
python convert_datasets_to_csv.py
```

### 2. Generate SLURM Jobs
```bash
# Create training scripts for all RBPs
bash generate_dnabert2_jobs.sh
```

### 3. Submit Training Jobs
```bash
# Submit to SLURM cluster
for job in logs_dnabert2/*.slurm; do
    sbatch $job
done
```

### 4. Evaluate Results
```bash
# Plot performance comparisons
python plot_eval_scatter.py
```

### 5. Interpret Models
```bash
# Analyze trained model
python visualisation.py \
    --model_path ./output/dnabert2_GTF2F1_fold5/ \
    --sequence AUGCAUGCAUGC \
    --method integrated_gradients
```

---

## Key Features

**Training:**
- LoRA fine-tuning support
- Cross-validation (5-fold)
- Early stopping
- Random initialization baselines
- K-mer tokenization option

**Metrics:**
- Accuracy, F1, Precision, Recall
- Matthews Correlation Coefficient
- AUC-ROC, AUC-PRC (primary metric)

**Interpretation:**
- In-silico mutagenesis
- Gradient-based saliency
- Integrated gradients
- Sliding window analysis

**Infrastructure:**
- SLURM cluster support
- GPU (A40) optimization
- Conda environment management
- Automated job generation

---

## Model Architecture

**DNABERT-2-117M:**
- Transformer encoder (12 layers)
- Hidden size: 768
- Attention heads: 12
- Parameters: 117M
- Pre-trained on human genome (GRCh38)
- Tokenization: BPE on k-mers

**Fine-tuning Head:**
```python
# Sequence classification
DNABERT-2 Encoder (117M params)
    ↓
[CLS] Token Representation (768-dim)
    ↓
Dropout (0.1)
    ↓
Linear Layer (768 → 2)
    ↓
Softmax (Binary Classification)
```

---

## Configuration Files

**environment.yml:**
```yaml
name: dnabert2
dependencies:
  - python=3.9
  - pytorch=2.0
  - transformers=4.30
  - scikit-learn
  - pandas
  - matplotlib
  - logomaker
```

**Training defaults:**
- Batch size: 8 (train), 16 (eval)
- Learning rate: 3e-5
- Epochs: 5 (with early stopping)
- Max sequence length: 101-512bp
- Optimizer: AdamW
- Warmup steps: 50
