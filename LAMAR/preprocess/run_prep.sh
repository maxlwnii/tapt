python3 preprocess.py \
    --fasta ../data/overlap.fa \
    --bed ../data/rel_data/overlap.bed \
    --eclip ../data/rel_data/combined_sorted_idr.bed \
    --output preprocessed_data \
    --max_len 512 \
    --stride 256 \
    --merge_peaks