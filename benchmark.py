import os
import sys
import argparse
import time
import torch
import sentencepiece as spm
from torch.serialization import default_restore_location
from seq2seq.models import Seq2SeqModel
# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq import models, utils
from seq2seq.decode import beam_search_decode #The new version  

def original_beam_search(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):
    """Beam Search decoding compatible with Transformer-based Seq2Seq models."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    # __QUESTION 1: what does this line set up and why is the beam represented this way?
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]
                # __QUESTION 2: Why do we need to create trg_pad_mask here and how does it affect the model's predictions?
                trg_pad_mask = (seq == PAD)[:, None, None, :]
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                # __QUESTION 3: Explain the purpose of applying log_softmax and selecting top-k tokens here.
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                # __QUESTION 4: explain the tensor shapes and the logic when creating new_seq and new_score below. Is any broadcasting or indexing issue possible?
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        # __QUESTION 5: Why do we check for EOS here and what does it imply for beam search?
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break
    best_seq, _ = beams[0]
    # __QUESTION 6: What is returned, and why are we squeezing, converting to list and wrapping in another list here?
    return [best_seq.squeeze(0).tolist()]

def get_args():
    parser = argparse.ArgumentParser('Benchmark Script')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input', required=True)
    parser.add_argument('--src-tokenizer', required=True)
    parser.add_argument('--tgt-tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--max-len', default=128, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    
    # 1. Load Model
    print("Loading model...")
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    
    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)
    
    model = models.build_model(args_loaded, src_tokenizer, tgt_tokenizer)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    
    DEVICE = 'cuda' if args.cuda else 'cpu'
    
    # ==========================================
    # 2. DATA PREPARATION (50 Sentences)
    # ==========================================
    print("Reading input data...")
    with open(args.input, encoding="utf-8") as f:
        # Take the first 50 lines
        src_lines = [line.strip() for line in f if line.strip()][:50]

    print(f"Preparing {len(src_lines)} individual sentences...")
    
    # We create a list of separate (token, mask) tuples.
    # We do NOT create one giant batch because 'original_beam_search' usually 
    # crashes if batch_size > 1 due to scalar .item() calls.
    benchmark_data = []
    
    # Helper to generate masks
    make_batch = utils.make_batch_input(device=DEVICE, pad=src_tokenizer.pad_id(), max_seq_len=args.max_len)

    for line in src_lines:
        # Encode to IDs
        encoded_ids = src_tokenizer.Encode(line, out_type=int, add_eos=True)
        
        # Convert to Tensor and add Batch Dimension: (Seq_Len) -> (1, Seq_Len)
        src_tensor = torch.tensor(encoded_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        
        # Create Masks using make_batch
        dummy_y = torch.full_like(src_tensor, fill_value=src_tokenizer.pad_id())
        src_tokens, _, _, src_pad_mask, _ = make_batch(src_tensor, dummy_y)
        
        benchmark_data.append((src_tokens, src_pad_mask))

    print(f"\n--- Starting Benchmark on {len(benchmark_data)} sentences ---")
    
    # ==========================================
    # 3. BENCHMARK LOOP
    # ==========================================
    def run_benchmark(beam_k):
        print(f"\nTesting Beam Size: {beam_k}")
        
        # Warmup: Run once on the very first sentence to initialize caches
        try:
            tokens_0, mask_0 = benchmark_data[0]
            _ = original_beam_search(model, tokens_0, mask_0, 30, tgt_tokenizer, args_loaded, DEVICE, beam_k)
        except Exception as e:
            print(f"Original function failed during warmup: {e}")
            return

        # Measure Original: Loop through all 50 sentences
        start = time.time()
        for src_tokens, src_pad_mask in benchmark_data:
            _ = original_beam_search(model, src_tokens, src_pad_mask, 30, tgt_tokenizer, args_loaded, DEVICE, beam_k)
        original_total_time = time.time() - start
        
        # Measure Optimized: Loop through all 50 sentences
        start = time.time()
        for src_tokens, src_pad_mask in benchmark_data:
            # NOTE: Ensure you fixed the argument order in decode.py as discussed previously!
            _ = beam_search_decode(model, src_tokens, src_pad_mask, 30, tgt_tokenizer, args_loaded, DEVICE, beam_k)
        opt_total_time = time.time() - start
        
        # Calculate Averages
        avg_orig = original_total_time / len(benchmark_data)
        avg_opt = opt_total_time / len(benchmark_data)
        
        print(f"  > Original Total Time:  {original_total_time:.4f}s (Avg: {avg_orig:.4f}s/sent)")
        print(f"  > Optimized Total Time: {opt_total_time:.4f}s (Avg: {avg_opt:.4f}s/sent)")
        print(f"  > Speedup:              {original_total_time/opt_total_time:.2f}x FASTER")

    # Run for k=1, 3, 5
    for k in [1, 3, 5]:
        run_benchmark(k)

if __name__ == '__main__':
    main()
