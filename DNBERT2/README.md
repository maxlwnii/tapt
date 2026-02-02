# Visualisation Genome Foundation Model Predictions.

This repo contains the necessary scripts to finetune DNABERT2 on RBP-seq data and evaluaten its predicitive capabilities to confirm specific binding motifs.
It contains a small dataset of PUM2 sequences.

# Setup
```bash
conda env create -f environment.yml
conda activate dnabert2
```
## Running the Script.

1.  **Convert the FASTA data to CSV:**
    This will create `train.csv`, `test.csv`, and `dev.csv` in the `data` directory.
    ```bash
    python3 convert_fasta_to_csv.py
    ```
    This will create the required csv files for finetuning.The files are in the format described in the DNABERT2 paper: with a sequence and its corresponding label.

2.  **Run the finetuning script:**
 
    ```bash
    export TOKENIZERS_PARALLELISM=false
    export DATA_PATH=./data
    export MAX_LENGTH=20
    export LR=3e-5

    python train.py \
        --model_name_or_path zhihan1996/DNABERT-2-117M \
        --data_path ${DATA_PATH} \
        --kmer -1 \
        --run_name DNABERT2_${DATA_PATH} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${LR} \
        --num_train_epochs 5 \
        --fp16 \
        --save_steps 200 \
        --eval_steps 200 \
        --output_dir output/dnabert2 \
        --warmup_steps 50 \
        --logging_steps 100 \
        --overwrite_output_dir \
        --log_level info
    ```

    Or just run:
    ```bash
    bash finetune.sh
    ```
    (Uninstalling triton helped resolve type errors.)
    
    ```bash
    pip uninstall -y triton
    ```

3. **Visualisation** 
    After finetuning, the model will be saved in the output/dnabert2/checkpoint-X directory.
    ```bash
    python3 visualisation.py
    ```
    You will be asked to enter 1) the model checkpoint, 2) a DNA sequence and 3) an output directory
    Furthermore, you can adjust paramaters: 
    ```bash
    Sliding window size [1]:
    Sliding window step size [1]: 
    Integrated gradients steps [120]:
    ```
    The resulting plots will be saved as a png file in the specified outut directory.
    
    ![Visualization example](images/vis.png)
    ```
