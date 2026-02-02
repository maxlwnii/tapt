"""
Methods:
1. In-silico mutagenesis.
2. Sliding window mutagenesis.
3. Gradient-based saliency (embedding-space).
4. Integrated gradients saliency (embedding-space)
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logomaker

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, AutoModelForMaskedLM

BASES = ['A', 'C', 'G', 'T']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

class SaliencyAnalyzer:
    def __init__(self, model_path, device=None):
        self.device = device or DEVICE
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        print(f"Loading model from: {self.model_path}")
        print(f"Using device: {self.device}")
        try:
            config = BertConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                config=config
            ).to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _encode(self, seq, with_offsets=False):
        return self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=with_offsets
        )

    def get_prob(self, seq):
        inputs = self._encode(seq)
        inputs = _to_device(inputs, self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item()
        

    def in_silico_mutagenesis(self, sequence):
        """
        """
        print("Computing in-silico mutagenesis...")
        seq = sequence.upper()
        L = len(seq)
        effect = pd.DataFrame(0.0, index=range(L), columns=BASES)
        orig_prob = self.get_prob(seq)

        for pos in range(L):
            orig_nt = seq[pos]
            for nt in BASES:
                if nt == orig_nt:
                    continue
                mutated = seq[:pos] + nt + seq[pos + 1:]
                mut_prob = self.get_prob(mutated)
                effect.at[pos, nt] = mut_prob - orig_prob

        print("In-silico mutagenesis completed!")
        return effect

    def sliding_window_mutagenesis(self, sequence, window_size=1, step_size=1):
        print(f"Computing sliding window mutagenesis (window size: {window_size}, step size: {step_size})...")
        seq = sequence.upper()
        L = len(seq)
        importance_scores = pd.DataFrame(0.0, index=range(L), columns=['Importance'])
        orig_prob = self.get_prob(seq)

        for start in range(0, L - window_size + 1, step_size):
            end = start + window_size
            mutated_seq = seq[:start] + (self.tokenizer.mask_token * window_size) + seq[end:]
            mut_prob = self.get_prob(mutated_seq)
            change = mut_prob - orig_prob
            for pos in range(start, end):
                importance_scores.at[pos, 'Importance'] += change

        # Normalize by coverage
        coverage = np.zeros(L)
        for start in range(0, L - window_size + 1, step_size):
            end = start + window_size
            coverage[start:end] += 1

        for pos in range(L):
            if coverage[pos] > 0:
                importance_scores.at[pos, 'Importance'] /= coverage[pos]

        print(" Sliding window mutagenesis completed!")
        return importance_scores

    def extract_offset_mapping(self, sequence):
        inputs = self._encode(sequence, with_offsets=True)
        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        token_ids = inputs['input_ids'][0].tolist()
        offset_mapping = inputs['offset_mapping'][0].tolist()

        filtered_tokens = []
        filtered_token_ids = []
        filtered_offset_mapping = []

        # skip special tokens with (0,0)
        for token, tid, (start, end) in zip(all_tokens, token_ids, offset_mapping):
            if not (start == 0 and end == 0):
                filtered_tokens.append(token)
                filtered_token_ids.append(tid)
                filtered_offset_mapping.append((start, end))

        return filtered_tokens, filtered_token_ids, filtered_offset_mapping

    def compute_gradient_saliency(self, sequence, target_class=None):
        """
        Gradient saliency via embedding-layer hook:
        d(logit[target]) / d(embedding_output)
        """
        print("Computing gradient-based saliency...")
        enc = self._encode(sequence)
        enc = _to_device(enc, self.device)

        input_ids = enc['input_ids']
        attention_mask = enc.get('attention_mask', None)
        token_type_ids = enc.get('token_type_ids', None)

        embedding_layer = self.model.get_input_embeddings()
        saved_grads = []

        def fwd_hook(module, inp, out):
            # no-op; keep to ensure grad hook attaches to correct tensor
            return None

        def bwd_hook(module, grad_input, grad_output):
            # grad_output[0]: dL/d(embedding_output) with shape [B, T, H]
            saved_grads.append(grad_output[0].detach())

        h1 = embedding_layer.register_forward_hook(fwd_hook)
        h2 = embedding_layer.register_full_backward_hook(bwd_hook)

        self.model.zero_grad(set_to_none=True)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=-1).item())

        loss = logits[0, target_class]
        loss.backward()

        # cleanup hooks
        h1.remove(); h2.remove()

        grads = saved_grads[-1]  # [1, T, H]
        token_scores = grads.norm(dim=-1).squeeze(0).cpu().numpy()  # [T]

        # Map token-level scores back to nucleotides via offsets
        filtered_tokens, _, filtered_offsets = self.extract_offset_mapping(sequence)
        L = len(sequence)
        saliency_df = pd.DataFrame(0.0, index=range(L), columns=BASES)

        enc_offsets = self._encode(sequence, with_offsets=True)['offset_mapping'][0].tolist()
        keep_mask = [not (s == 0 and e == 0) for (s, e) in enc_offsets]
        token_scores_kept = np.array([ts for ts, keep in zip(token_scores, keep_mask) if keep])

        for i, (start, end) in enumerate(filtered_offsets):
            if i >= len(token_scores_kept):
                break
            score = float(token_scores_kept[i])
          #  score = score / (end - start) # normalization over token length
            for pos in range(start, min(end, L)):
                base = sequence[pos].upper()
                if base in BASES:
                    saliency_df.at[pos, base] += score

        print(" Gradient saliency completed!")
        return saliency_df, filtered_tokens, token_scores_kept.tolist(), filtered_offsets

    def compute_integrated_gradients_saliency(self, sequence, target_class=None, n_steps=120):
        """
        Integrated gradients approximated in embedding space using an embedding-output hook.
        Baseline is zero embeddings; path implemented by interpolating between baseline and input embeddings.
        """
        print(f"Computing integrated gradients with {n_steps} steps...")
        enc = self._encode(sequence)
        enc = _to_device(enc, self.device)
                
        input_ids = enc['input_ids']
        attention_mask = enc.get('attention_mask', None)
        token_type_ids = enc.get('token_type_ids', None)

        embedding_layer = self.model.get_input_embeddings()
        with torch.no_grad():
            orig_embeds = embedding_layer(input_ids).detach()
       # pad_id = self.tokenizer.pad_token_id  ---> leads to considerable worse performance.
        mask_id = self.tokenizer.mask_token_id
        mask_ids = torch.full_like(input_ids, mask_id)
        baseline_embeds = embedding_layer(mask_ids).detach()
        
        # Storage for interpolated embeddings and gradients
        current_embeds = None
        saved_grads = []
        
        def fwd_hook(module, inp, out):
            # Replace the embedding output with  interpolated embeddings
            if current_embeds is not None:
                return current_embeds
            return out
        
        def bwd_hook(module, grad_input, grad_output):
            # Capture gradients w.r.t. the interpolated embeddings
            saved_grads.append(grad_output[0].detach().clone())

        h1 = embedding_layer.register_forward_hook(fwd_hook)
        h2 = embedding_layer.register_full_backward_hook(bwd_hook)

        alphas = torch.linspace(0.0, 1.0, n_steps, device=self.device)
        integrated_grads = torch.zeros_like(orig_embeds)

        for i, alpha in enumerate(alphas):
            # Create interpolated embeddings: baseline + alpha * (input - baseline)
            with torch.no_grad():
                current_embeds = baseline_embeds + alpha * (orig_embeds - baseline_embeds)
            
            # Ensure we capture gradients for this step
            current_embeds.requires_grad_(True)
            saved_grads.clear()
            
            self.model.zero_grad(set_to_none=True)
            
            # Forward pass with interpolated embeddings
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = out.logits

            if target_class is None:
                tgt = int(torch.argmax(logits, dim=-1).item())
            else:
                tgt = target_class

            logit = logits[0, tgt]
            logit.backward()

            # Accumulate gradients for integration (Riemann sum approximation)
            if saved_grads:
                integrated_grads += saved_grads[-1]
            
            if (i + 1) % max(1, n_steps // 5) == 0:
                print(f"  Step {i+1}/{n_steps} completed")

        h1.remove()
        h2.remove()

        # Compute final attributions: (input - baseline) * average_gradients
        avg_grads = integrated_grads / n_steps
        attributions = (orig_embeds - baseline_embeds) * avg_grads
        token_attr = attributions.norm(dim=-1).squeeze(0).cpu().numpy()  # [T]
    
        filtered_tokens, _, filtered_offsets = self.extract_offset_mapping(sequence)
        L = len(sequence)
        saliency_df = pd.DataFrame(0.0, index=range(L), columns=BASES)

        enc_offsets = self._encode(sequence, with_offsets=True)['offset_mapping'][0].tolist()
        keep_mask = [not (s == 0 and e == 0) for (s, e) in enc_offsets]
        token_attr_kept = np.array([ts for ts, keep in zip(token_attr, keep_mask) if keep])

        for i, (start, end) in enumerate(filtered_offsets):
            if i >= len(token_attr_kept):
                break
            score = float(token_attr_kept[i])
            for pos in range(start, min(end, L)):
                base = sequence[pos].upper()
                if base in BASES:
                    saliency_df.at[pos, base] += score

        print(" Integrated gradients completed!")
        return saliency_df, filtered_tokens, token_attr_kept.tolist(), filtered_offsets

    def create_sliding_window_logo_df(self, sequence, importance_scores):
        seq = sequence.upper()
        L = len(seq)
        logo_df = pd.DataFrame(0.0, index=range(L), columns=BASES)
        for pos in range(L):
            base = seq[pos]
            if base in BASES:
                logo_df.at[pos, base] = float(importance_scores.at[pos, 'Importance'])
        return logo_df

    def analyze_sequence(self, sequence, output_dir="saliency_output", window_size=1, step_size=1, n_steps=120):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        seq = sequence.upper()
        orig_prob = self.get_prob(seq)
        prediction = "Positive" if orig_prob > 0.5 else "Negative"

        print(f"\n=== SEQUENCE ANALYSIS ===")
        print(f"Sequence length: {len(seq)}")
        print(f"Prediction: {prediction} (probability: {orig_prob:.3f})")
        print(f"Output directory: {output_dir}")

        # Analyses
        in_silico_df = self.in_silico_mutagenesis(seq)
        window_importance = self.sliding_window_mutagenesis(seq, window_size=window_size, step_size=step_size)
        gradient_df, grad_tokens, grad_scores, grad_offsets = self.compute_gradient_saliency(seq)                    # Scores used to be written in a csv file.
        ig_df, ig_tokens, ig_scores, ig_offsets = self.compute_integrated_gradients_saliency(seq, n_steps=n_steps)

        # Logos
        seq_df = logomaker.alignment_to_matrix([seq])
        sliding_window_df = self.create_sliding_window_logo_df(seq, window_importance)

        fig, axs = plt.subplots(5, 1, figsize=(max(12, len(seq) // 2), 15),
                                gridspec_kw={'height_ratios': [1, 2, 2, 2, 2]})

        # Sequence logo
        logomaker.Logo(seq_df, ax=axs[0])
        axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[0].set_ylabel('Sequence')
        axs[0].set_title(f"Prediction: {prediction} (prob: {orig_prob:.3f})", fontsize=14, loc='left')

        # In-silico
        logomaker.Logo(in_silico_df, ax=axs[1])
        axs[1].axhline(0, color='black', linewidth=1)
        axs[1].set_ylabel('In-silico mutations'); axs[1].set_xlabel('')

        # Sliding window
        logomaker.Logo(sliding_window_df, ax=axs[2])
        axs[2].axhline(0, color='black', linewidth=1)
        axs[2].set_ylabel(f'Sliding window (size={window_size})'); axs[2].set_xlabel('')

        # Gradient saliency
        logomaker.Logo(gradient_df, ax=axs[3])
        axs[3].axhline(0, color='black', linewidth=1)
        axs[3].set_ylabel('Gradient saliency'); axs[3].set_xlabel('')

        # Integrated gradients
        logomaker.Logo(ig_df, ax=axs[4])
        axs[4].axhline(0, color='black', linewidth=1)
        axs[4].set_ylabel('Integrated gradients'); axs[4].set_xlabel('Position')

        plt.tight_layout()

        plot_filename = f"saliency_analysis_{timestamp}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f" Plot saved: {plot_path}")
        plt.show(); plt.close(fig)

        results_df = pd.DataFrame({
            'Position': range(len(seq)),
            'Nucleotide': list(seq)
        })

        for pos in range(len(seq)):
            base = seq[pos]
            if base in BASES:
                results_df.loc[pos, 'InSilico_Score'] = float(in_silico_df.at[pos, base])
                results_df.loc[pos, 'SlidingWindow_Score'] = float(sliding_window_df.at[pos, base])
                results_df.loc[pos, 'Gradient_Score'] = float(gradient_df.at[pos, base])
                results_df.loc[pos, 'IntegratedGrad_Score'] = float(ig_df.at[pos, base])
        return {
            'in_silico': in_silico_df,
            'sliding_window': sliding_window_df,
            'gradient': gradient_df,
            'integrated_gradients': ig_df,
            'prediction': prediction,
            'probability': orig_prob,
            'plot_path': plot_path,
        }


def get_user_input():
    print("=== DNA Sequence Saliency Analysis Tool ===")
    model_path = input("Enter the path to your trained model directory [output/dnabert2/checkpoint-1000]: ").strip()
    if not model_path:
        model_path = "output/dnabert2/checkpoint-1000"

    print("\nChoose sequence input method:")
    print("Enter sequence manually")

    sequence = input("Enter DNA sequence (A/T/G/C/U only): ").strip().upper().replace("U", "T")

    # Validate sequence
    valid_bases = set('ATGCU')
    if not sequence or not all(base in valid_bases for base in sequence):
        raise ValueError("Sequence contains invalid characters. Only A, T, G, C, U are allowed.")

    print("\nOptional parameters (press Enter for defaults):")
    output_dir = input("Output directory [saliency_output]: ").strip() or "saliency_output"
    window_size = input("Sliding window size [1]: ").strip()
    window_size = int(window_size) if window_size else 1
    step_size   = input("Sliding window step size [1]: ").strip()
    step_size = int(step_size) if step_size else 1
    n_steps = input("Integrated gradients steps [120]: ").strip()
    n_steps = int(n_steps) if n_steps else 120

    return model_path, sequence, output_dir, window_size, step_size, n_steps


def main():
    try:
        model_path, sequence, output_dir, window_size, step_size, n_steps = get_user_input()
        print("\nInitializing analyzer...")
        analyzer = SaliencyAnalyzer(model_path)
        print("\nStarting analysis...")
        results = analyzer.analyze_sequence(
            sequence=sequence,
            output_dir=output_dir,
            window_size=window_size,
            step_size=step_size,
            n_steps=n_steps
        )
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Prediction: {results['prediction']}")
        print(f"Probability: {results['probability']:.3f}")
        print(f"Plot saved: {results['plot_path']}")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
