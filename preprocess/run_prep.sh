python3 preprocess.py \
    --fasta ../data/rel_data/overlap.fa \
    --bed ../data/rel_data/overlap.bed \
    --eclip ../data/rel_data/combined_sorted_idr.bed \
    --output preprocessed_data_1024 \
    --max_len 1024 \
    --stride 512 \
    --merge_peaks