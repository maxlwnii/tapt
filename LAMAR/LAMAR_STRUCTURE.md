# LAMAR Directory Structure Documentation

## Overview
The `LAMAR/` directory contains a nucleotide-specific ESM2-based language model for RNA/DNA sequence analysis. LAMAR (Language Model for Analyzing RNA) is adapted from ESM-2 and includes training, fine-tuning, evaluation, and interpretation tools for RNA-binding protein (RBP) binding site prediction.

---

## Core Architecture

### LAMAR Model Design

**Base:** ESM-2 architecture adapted for nucleotide sequences

```python
# Configuration (150M parameter model)
EsmConfig(
    vocab_size=tokenizer_vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    positional_embedding_type="rotary",  # Rotary position embeddings
    token_dropout=False,
    pad_token_id=tokenizer.pad_token_id,
    mask_token_id=tokenizer.mask_token_id
)
```

**Key Features:**
- Rotary position embeddings (RoPE)
- 768-dimensional hidden states
- 12 transformer layers
- 12 attention heads
- ~150M parameters

---

## Directory Structure

```
LAMAR/
├── LAMAR/                          # Core model implementation
│   ├── modeling_nucESM2.py        # Main model architecture
│   ├── sequence_classification_patch.py
│   ├── data_collator_patch.py
│   └── flash_attn_patch.py
│
├── evalEmbeddings/                 # Evaluation & benchmarking
│   ├── LAMAR_CNN_clip_data.py     # LAMAR embedding + CNN
│   ├── LAMAR_CNN_tapt.py          # TAPT variant evaluation
│   ├── LAMAR_CNN_random.py        # Random baseline
│   ├── OneHot_CNN_clip_data.py    # One-hot encoding baseline
│   ├── plot_results.py            # Result visualization
│   ├── depMaps/                   # Dependency map analysis
│   │   └── compute_dep_maps.py
│   └── isoscore/                  # Embedding quality metrics
│       ├── compute_isoscore_clip.py
│       └── compute_isoscore_finetuned_clip.py
│
├── finetune_scripts/               # Fine-tuning for RBP tasks
│   └── finetune_rbp.py
│
├── preprocess/                     # Data preprocessing
│   ├── preprocess.py              # CLIP data processing
│   ├── train_val_split.py
│   └── run_prep.sh
│
├── models/                         # Trained model checkpoints
│   ├── finetuned_full/
│   ├── finetuned_limited/
│   ├── finetuned_early_stopping_full/
│   └── finetuned_test/
│
├── visualisation/                  # Model interpretation
│   └── saliency_analyzer.py
│
├── weights                         # Pre-trained LAMAR weights
├── environment_finetune_fixed.yml  # Conda environment
└── dep_map.ipynb                   # Dependency analysis notebook
```

---

## 1. Core Model Implementation

### **LAMAR/modeling_nucESM2.py** (1192 lines)

**Rotary Position Embeddings:**
```python
class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings for relative positional encoding.
    Query and keys are transformed by rotation matrices 
    depending on their relative positions.
    """
    def __init__(self, dim: int):
        super().__init__()
        # Frequency scaling: 1 / (10000^(2i/d))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # Generate cos/sin tables
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k)
        # Apply rotation
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)
        )

def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding"""
    return (x * cos) + (rotate_half(x) * sin)

def rotate_half(x):
    """Split and rotate for complex multiplication"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
```

**Custom Activation Functions:**
```python
def gelu(x):
    """ESM-specific GELU implementation"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

**Model Architecture:**
- `EsmEmbeddings`: Token + position embeddings
- `EsmEncoder`: 12 transformer layers
- `EsmForMaskedLM`: Pre-training head
- `EsmForSequenceClassification`: Fine-tuning head

---

## 2. Evaluation Framework

### **evalEmbeddings/LAMAR_CNN_clip_data.py** (357 lines)

**Embedding Extraction + CNN Classifier Approach**

```python
# Model variants for comparison
variants = {
    "Pretrained": "/path/to/pretrained/weights",
    "TAPT": "/path/to/tapt/checkpoint-100000/model.safetensors",
    "Random": None  # Random initialization
}

# Multi-layer embedding extraction
target_layers = [11, 5]  # Layer 11 (top) and Layer 5 (middle)

def load_data(rbp_name):
    """Load positive and negative FASTA sequences"""
    pos_file = f"{data_dir}/{rbp_name}.positives.fa"
    neg_file = f"{data_dir}/{rbp_name}.negatives.fa"
    
    seqs, labels = [], []
    
    # Load positives (label=1)
    for record in SeqIO.parse(pos_file, "fasta"):
        seq = str(record.seq).upper().replace("U", "T")
        seqs.append(seq)
        labels.append(1)
    
    # Load negatives (label=0)
    for record in SeqIO.parse(neg_file, "fasta"):
        seq = str(record.seq).upper().replace("U", "T")
        seqs.append(seq)
        labels.append(0)
    
    return np.array(seqs), np.array(labels)
```

**CNN Architecture for Downstream Classification:**
```python
def chip_cnn(input_shape, output_shape):
    """
    CNN classifier on top of LAMAR embeddings
    
    Architecture:
    - BatchNorm + 1x1 Conv (512 filters) for dimension reduction
    - Conv1D (64 filters, kernel=7) + BatchNorm + ReLU + MaxPool + Dropout
    - Conv1D (96 filters, kernel=5) + BatchNorm + ReLU + MaxPool + Dropout
    - Conv1D (128 filters, kernel=5) + BatchNorm + ReLU + MaxPool + Dropout
    - Flatten + Dense(256) + BatchNorm + ReLU + Dropout
    - Dense(1) + Sigmoid
    """
    initializer = tf.keras.initializers.HeUniform(seed=42)
    input = keras.Input(shape=input_shape)
    
    # Dimension reduction
    nn = keras.layers.BatchNormalization()(input)
    nn = keras.layers.Conv1D(filters=512, kernel_size=1, 
                             kernel_initializer=initializer)(nn)
    
    # First conv block
    nn = keras.layers.Conv1D(filters=64, kernel_size=7, 
                             padding='same', 
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    
    # Second conv block
    nn = keras.layers.Conv1D(filters=96, kernel_size=5, 
                             padding='same', 
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    
    # Third conv block
    nn = keras.layers.Conv1D(filters=128, kernel_size=5, 
                             padding='same', 
                             kernel_initializer=initializer)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(2)(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    
    # Dense layers
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(256)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.Dropout(0.5)(nn)
    
    # Output
    logits = keras.layers.Dense(output_shape)(nn)
    output = keras.layers.Activation('sigmoid')(logits)
    
    model = keras.Model(inputs=input, outputs=output)
    return model
```

**Model Loading and Weight Initialization:**
```python
def get_lamar_model(weights_path):
    """Load LAMAR with proper configuration"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, 
        model_max_length=model_max_length
    )
    
    # Create config
    config = EsmConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        token_dropout=False,
        positional_embedding_type="rotary",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_labels=2
    )
    
    # Initialize model
    model = EsmForMaskedLM(config)
    
    # Load weights
    if weights_path:
        if weights_path.endswith('.safetensors'):
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
        
        # Clean weight names (handle esm. prefix)
        weight_dict = {}
        for k, v in state_dict.items():
            if k.startswith("esm."):
                weight_dict[k] = v
            else:
                weight_dict["esm." + k] = v
        
        model.load_state_dict(weight_dict, strict=False)
    
    return model, tokenizer

def init_weights(module):
    """WOLF initialization for random baseline"""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)
```

**Embedding Extraction Pipeline:**
```python
# Extract embeddings from specific layers
for i in range(0, len(seqs), extraction_batch_size):
    batch_seqs = seqs[i:i+extraction_batch_size]
    
    # Tokenize
    inputs = tokenizer(batch_seqs, 
                      return_tensors="pt", 
                      padding=True, 
                      truncation=True, 
                      max_length=model_max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model.esm(**inputs, output_hidden_states=True)
        
        # Extract from target layers
        for layer_idx in target_layers:
            hidden_state = outputs.hidden_states[layer_idx]
            
            # Mean pooling
            mask = inputs['attention_mask'].unsqueeze(-1)
            pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            
            embeddings[layer_idx].append(pooled.cpu().numpy())
```

---

### **evalEmbeddings/OneHot_CNN_clip_data.py** (289 lines)

**Baseline: One-Hot Encoding + CNN**

```python
def one_hot_encode(seq, max_len=512):
    """
    One-hot encode DNA sequence
    Returns: (max_len, 4) array for A, C, G, T
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((max_len, 4), dtype=np.float32)
    
    seq = seq[:max_len].upper()
    for i, nucleotide in enumerate(seq):
        if nucleotide in mapping:
            one_hot[i, mapping[nucleotide]] = 1.0
    
    return one_hot

def get_baseline_cnn_onehot(input_shape, output_shape=1):
    """
    Same CNN architecture as LAMAR_CNN but on one-hot inputs
    """
    initializer = tf.keras.initializers.HeUniform(seed=42)
    input_layer = keras.Input(shape=input_shape)
    
    # Conv Layer 1
    nn = keras.layers.Conv1D(filters=64, kernel_size=7, 
                             padding='same', 
                             kernel_initializer=initializer)(input_layer)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation('relu')(nn)
    nn = keras.layers.MaxPooling1D(4)(nn)
    nn = keras.layers.Dropout(0.2)(nn)
    
    # ... (similar structure to LAMAR CNN)
    
    return model
```

---

## 3. Fine-tuning Framework

### **finetune_scripts/finetune_rbp.py** (793 lines)

**Complete fine-tuning pipeline with cross-validation**

```python
def compute_metrics(p):
    """Comprehensive metrics for classification and regression"""
    predictions, labels = p
    
    # Classification branch
    if predictions.ndim > 1 and predictions.shape[-1] > 1:
        probs = torch.nn.functional.softmax(
            torch.tensor(predictions), dim=-1
        ).numpy()
        pred_labels = np.argmax(predictions, axis=1)
        
        # Standard metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='binary', zero_division=0
        )
        acc = accuracy_score(labels, pred_labels)
        auc = roc_auc_score(labels, probs[:, 1])
        auprc = average_precision_score(labels, probs[:, 1])
        
        return {
            'accuracy': float(acc),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc),
            'auprc': float(auprc),
        }
    
    # Regression branch
    else:
        preds_1d = predictions.reshape(-1)
        labs_1d = labels.reshape(-1)
        mse = float(np.mean((preds_1d - labs_1d) ** 2))
        
        df = pd.DataFrame({'pred': preds_1d, 'label': labs_1d})
        corr_pearson = float(df.corr(method='pearson').iloc[0, 1])
        corr_spearman = float(df.corr(method='spearman').iloc[0, 1])
        
        return {
            'mse': mse,
            'corr_coef_pearson': corr_pearson,
            'corr_coef_spearman': corr_spearman,
        }

def load_encoder_weights(model, weights_path):
    """
    Load ONLY encoder weights, leave classifier random
    Important for proper transfer learning
    """
    state_dict = load_file(weights_path)
    
    # Filter to keep ONLY encoder weights (esm.*)
    encoder_weights = {}
    for k, v in state_dict.items():
        # Skip language model head and classifier
        if 'lm_head' in k or 'classifier' in k:
            continue
        
        # Ensure proper esm. prefix
        if k.startswith("esm."):
            encoder_weights[k] = v
        else:
            encoder_weights["esm." + k] = v
    
    # Load with strict=False (classifier will remain random)
    missing_keys, unexpected_keys = model.load_state_dict(
        encoder_weights, strict=False
    )
    
    encoder_loaded = [k for k in missing_keys if k.startswith('esm.')]
    classifier_missing = [k for k in missing_keys if 'classifier' in k]
    
    print(f"✓ Encoder weights loaded: {len(encoder_weights)} tensors")
    print(f"✓ Classifier randomly initialized: {len(classifier_missing)} tensors")
    
    return model

def freeze_encoder(model, freeze=True):
    """Freeze encoder, keep classifier trainable"""
    for name, param in model.named_parameters():
        if name.startswith('esm.'):
            param.requires_grad = not freeze
        else:  # classifier layers
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

**Training Arguments:**
```python
parser.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder, train only classifier")
parser.add_argument("--warmup_epochs", type=int, default=0,
                   help="Epochs to train only classifier before unfreezing")
parser.add_argument("--early_stopping_patience", type=int, default=None)
parser.add_argument("--subsample_pos", type=int, default=None,
                   help="Subsample positive samples (limited data)")
parser.add_argument("--cv_folds", type=int, default=5,
                   help="Cross-validation folds")
parser.add_argument("--nlabels", type=int, default=2,
                   help="2 for classification, 1 for regression")
parser.add_argument("--fp16", action='store_true',
                   help="Enable mixed precision training")
```

**Cross-Validation Support:**
```python
# K-fold CV on combined dataset
if args.cv_folds > 1:
    kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_fold = dataset.select(train_idx)
        val_fold = dataset.select(val_idx)
        
        # Train on this fold
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(patience=args.early_stopping_patience)]
        )
        trainer.train()
```

---

## 4. Data Preprocessing

### **preprocess/preprocess.py** (698 lines)

**eCLIP Peak Processing**

```python
def read_fasta_file(fasta_file):
    """
    Parse FASTA with genomic coordinates
    
    Expected format:
    >chr1:100-110
    AGCAGTCGAT
    """
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        chrom, rest = record.id.split(':')
        if not is_standard_chromosome(chrom):
            continue
        
        start, end = rest.split('-')
        sequence = str(record.seq).upper()
        
        sequences.append({
            'chrom': chrom,
            'seq_id': record.id,
            'start': int(start),
            'end': int(end),
            'sequence': sequence,
            'seq_len': len(sequence)
        })
    
    return sequences

def is_standard_chromosome(chrom):
    """Filter to standard chromosomes only"""
    standard_chroms = {f'chr{i}' for i in range(1, 23)}.union({'chrX', 'chrY'})
    return chrom in standard_chroms

def read_bed_file(bed_file):
    """
    Parse BED file with peak information
    
    Format: chr start end name score
    score = number of overlapping binding sites
    """
    peaks = defaultdict(list)
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            chrom = parts[0]
            if not is_standard_chromosome(chrom):
                continue
            
            peaks[chrom].append({
                'start': int(parts[1]),
                'end': int(parts[2]),
                'num_binding_sites': int(parts[4])
            })
    
    return peaks
```

**Sampling Strategies:**
```python
def sample_sequences(sequence, max_len):
    """
    Sample fixed-length windows from variable-length sequences
    
    Strategy:
    1) seq_len <= max_len: Take entire sequence (pad later)
    2) seq_len > max_len: Sample overlapping windows (stride)
    3) Very long sequences: Sample sqrt(seq_len/max_len) random windows
    """
    seq_len = len(sequence)
    
    if seq_len <= max_len:
        return [sequence]
    
    # For moderately long sequences: overlapping windows
    if seq_len < max_len * 4:
        stride = max_len // 2
        windows = []
        for start in range(0, seq_len - max_len + 1, stride):
            windows.append(sequence[start:start+max_len])
        return windows
    
    # For very long sequences: random sampling
    n_samples = int(np.sqrt(seq_len / max_len))
    windows = []
    for _ in range(n_samples):
        start = np.random.randint(0, seq_len - max_len + 1)
        windows.append(sequence[start:start+max_len])
    
    return windows
```

---

## 5. Interpretation & Analysis

### **evalEmbeddings/depMaps/compute_dep_maps.py** (673 lines)

**Dependency Map Analysis**

```python
class EsmSequenceClassifier(torch.nn.Module):
    """Wrapper for computing attribution scores"""
    
    def __init__(self, esm_model, hidden_size=768, num_labels=2):
        super().__init__()
        self.esm = esm_model.esm  # ESM encoder
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.esm(input_ids=input_ids, 
                          attention_mask=attention_mask)
        
        # [CLS] token for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), 
                           labels.view(-1))
        
        return type('Output', (), {
            'logits': logits, 
            'loss': loss,
            'hidden_states': outputs.hidden_states
        })()
```

**Interpretation Methods:**
1. **In-silico mutagenesis** - Single nucleotide changes
2. **Sliding window mutagenesis** - Regional importance
3. **Gradient-based saliency** - Embedding-space attribution
4. **Integrated gradients** - Path-integral attribution

---

### **evalEmbeddings/isoscore/compute_isoscore_finetuned_clip.py** (163 lines)

**IsoScore: Embedding Quality Metric**

```python
def compute_isoscore(embeddings):
    """
    Measure embedding isotropy (uniform distribution in space)
    
    Higher IsoScore = more isotropic = better representation
    """
    # Center embeddings
    embeddings = embeddings - embeddings.mean(axis=0)
    
    # SVD decomposition
    _, S, _ = np.linalg.svd(embeddings, full_matrices=False)
    
    # Normalize singular values
    S_normalized = S / S.sum()
    
    # Cumulative variance
    cumsum = np.cumsum(S_normalized)
    
    # Dimensions needed for 90% variance
    k_90 = np.searchsorted(cumsum, 0.9) + 1
    
    # IsoScore: normalized dimensionality
    isoscore = k_90 / len(S)
    
    return isoscore

def get_embeddings(model, tokenizer, sequences):
    """Extract embeddings with mean pooling"""
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        
        inputs = tokenizer(batch_seqs, 
                          return_tensors="pt", 
                          padding=True, 
                          truncation=True, 
                          max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.esm(**inputs)
            
            # Mean pooling over sequence
            mask = inputs['attention_mask'].unsqueeze(-1)
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            embeddings.append(pooled.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)
```

---

## Usage Workflows

### 1. **Embedding Extraction + CNN Evaluation**

```bash
# Evaluate LAMAR on all RBPs
python evalEmbeddings/LAMAR_CNN_clip_data.py

# Evaluate one-hot baseline
python evalEmbeddings/OneHot_CNN_clip_data.py

# Compare results
python evalEmbeddings/plot_results.py
```

### 2. **Fine-tuning on RBP Tasks**

```bash
# Fine-tune with pre-trained weights
python finetune_scripts/finetune_rbp.py \
    --rbp_name GTF2F1_K562_IDR \
    --data_path /path/to/finetune_data_koo/GTF2F1_K562_IDR \
    --output_dir ./models/finetuned/GTF2F1 \
    --pretrain_path /path/to/tapt/checkpoint-100000/model.safetensors \
    --epochs 10 \
    --batch_size 4 \
    --lr 3e-5 \
    --freeze_encoder \
    --warmup_epochs 2 \
    --early_stopping_patience 3 \
    --cv_folds 5

# Fine-tune with random initialization (baseline)
python finetune_scripts/finetune_rbp.py \
    --rbp_name GTF2F1_K562_IDR \
    --data_path /path/to/finetune_data_koo/GTF2F1_K562_IDR \
    --output_dir ./models/random/GTF2F1 \
    --pretrain_path ""  # Empty = random init \
    --epochs 10 \
    --batch_size 4
```

### 3. **Compute IsoScore**

```bash
# Analyze embedding quality
python evalEmbeddings/isoscore/compute_isoscore_finetuned_clip.py
```

### 4. **Dependency Map Analysis**

```bash
# Compute attribution maps
python evalEmbeddings/depMaps/compute_dep_maps.py \
    --model_path ./models/finetuned/GTF2F1 \
    --sequence_file test_sequences.fa \
    --output_dir ./depMaps_output
```

---

## Key Configurations

### Model Variants

**Pretrained:** Base LAMAR model trained on genomic sequences
**TAPT (Task-Adaptive Pre-Training):** Additional pre-training on CLIP sequences
**Random:** Randomly initialized baseline

### Training Defaults

```python
# Fine-tuning
batch_size = 4
learning_rate = 3e-5
epochs = 10
warmup_ratio = 0.05
gradient_clipping = 1.0
fp16 = True  # Mixed precision

# Evaluation
extraction_batch_size = 32
downstream_batch_size = 256
model_max_length = 512
```

### CNN Training

```python
# TensorFlow/Keras
batch_size = 256
epochs = 30
optimizer = Adam(learning_rate=0.001)
loss = binary_crossentropy
callbacks = [
    EarlyStopping(patience=5),
    ReduceLROnPlateau(patience=3)
]
```

---

## Performance Expectations

**Embedding + CNN (AUC-PRC):**
- LAMAR (Pretrained): 0.70-0.85
- LAMAR (TAPT): 0.75-0.90
- LAMAR (Random): 0.60-0.75
- One-Hot Baseline: 0.55-0.70

**Fine-tuned Models (AUC-PRC):**
- Full fine-tuning: 0.80-0.95
- Frozen encoder: 0.75-0.90
- Limited data (100 samples): 0.65-0.80

**IsoScore:**
- Pre-trained: 0.60-0.75
- TAPT: 0.65-0.80
- Random: 0.40-0.55

Higher IsoScore indicates more uniform embedding space.

---

## Environment Setup

```yaml
# environment_finetune_fixed.yml
name: lamar
dependencies:
  - python=3.9
  - pytorch=2.0
  - transformers=4.30
  - safetensors
  - tensorflow=2.12
  - scikit-learn
  - pandas
  - numpy
  - biopython
  - matplotlib
  - logomaker
  - tqdm
```

---

## Model Checkpoints

```
models/
├── finetuned_full/          # Full fine-tuning (encoder + classifier)
│   ├── Pretrained/
│   ├── TAPT/
│   └── Random/
├── finetuned_limited/       # Limited data (100 pos + 100 neg)
│   ├── Pretrained/
│   ├── TAPT/
│   └── Random/
├── finetuned_early_stopping_full/  # With early stopping
└── finetuned_test/          # Development experiments
```

Each checkpoint contains:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `training_args.bin` - Training hyperparameters
- `eval_results.json` - Performance metrics

---

## Key Differences from DNABERT2

| Feature | LAMAR | DNABERT2 |
|---------|-------|----------|
| Base Model | ESM-2 | BERT |
| Position Embeddings | Rotary (RoPE) | Learned Absolute |
| Tokenization | Character-level | BPE k-mers |
| Pre-training | Masked Language Model | Masked Language Model |
| Parameters | 150M | 117M |
| Hidden Size | 768 | 768 |
| Layers | 12 | 12 |
| Domain | RNA/DNA | DNA |

---

## Advanced Features

**Flash Attention:** Optional optimization for longer sequences
**Gradient Checkpointing:** Memory-efficient training
**Mixed Precision (FP16):** Faster training with lower memory
**LoRA:** Parameter-efficient fine-tuning (not implemented)
**Cross-Validation:** Built-in k-fold support
**Warmup Training:** Freeze encoder, train classifier first

---

## Troubleshooting

**Common Issues:**

1. **Weight loading errors:** Check esm. prefix in state_dict keys
2. **CUDA OOM:** Reduce batch size or use gradient accumulation
3. **Poor performance:** Check frozen layers, learning rate
4. **Embedding dimension mismatch:** Verify config matches tokenizer

**Debug Commands:**
```python
# Check trainable parameters
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Verify weight loading
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing: {missing}")
print(f"Unexpected: {unexpected}")
```
