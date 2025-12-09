import os
import sys
import argparse
import time
import torch
import sentencepiece as spm
from torch.serialization import default_restore_location
import sacrebleu

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.models import Seq2SeqModel
from seq2seq import models, utils

# IMPORT YOUR PRUNING FUNCTIONS HERE
from seq2seq.decode import beam_search_decode, beam_search_decode_relative_pruning, beam_search_decode_absolute_pruning

def get_args():
    parser = argparse.ArgumentParser('Benchmark Script')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input', required=True)
    parser.add_argument('--src-tokenizer', required=True)
    parser.add_argument('--tgt-tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--max-len', default=128, type=int)
    parser.add_argument('--reference', type=str, help='Path to reference file (required for BLEU)')
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
    PAD, BOS, EOS = src_tokenizer.pad_id(), tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id()

    # 2. Data Preparation (50 sentences)
    print("Reading input data...")
    with open(args.input, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()][:50]
        
    benchmark_data = []
    make_batch = utils.make_batch_input(device=DEVICE, pad=PAD, max_seq_len=args.max_len)

    for line in src_lines:
        encoded_ids = src_tokenizer.Encode(line, out_type=int, add_eos=True)
        src_tensor = torch.tensor(encoded_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        dummy_y = torch.full_like(src_tensor, fill_value=PAD)
        src_tokens, _, _, src_pad_mask, _ = make_batch(src_tensor, dummy_y)
        benchmark_data.append((src_tokens, src_pad_mask))

    print(f"\n--- Starting Pruning Strategy Benchmark (50 sentences) ---")

    # Helper function for post-processing
    def decode_sentence(ids):
        if isinstance(ids, torch.Tensor): ids = ids.tolist()
        if len(ids) > 0 and ids[0] == BOS: ids = ids[1:] # Remove BOS
        if EOS in ids: ids = ids[:ids.index(EOS)]        # Truncate at EOS
        return tgt_tokenizer.Decode(ids)

    # 3. Universal Benchmark Runner
    def run_method(name, decode_func, **kwargs):
        print(f"\nMethod: {name}")
        translations = []
        total_tokens = 0
        
        start = time.time()
        for src_tokens, src_pad_mask in benchmark_data:
            # Call the specific decoding function
            # We fix alpha=0.0 to isolate the pruning effect
            output = decode_func(
                model, src_tokens, src_pad_mask, 30, tgt_tokenizer, 
                args_loaded, DEVICE, beam_size=5, **kwargs
            )
            # Process output
            if not output: # Safety check if pruning killed everything
                best_ids = []
            else:
                best_ids = output[0]
                
            total_tokens += len(best_ids)
            translations.append(decode_sentence(best_ids))
            
        total_time = time.time() - start
        avg_time = total_time / len(benchmark_data)
        avg_len = total_tokens / len(benchmark_data)
        
        print(f"  > Avg Time:   {avg_time:.4f}s")
        print(f"  > Avg Length: {avg_len:.2f}")

        # BLEU
        if args.reference:
            with open(args.reference, encoding='utf-8') as ref_file:
                refs = [line.strip() for line in ref_file if line.strip()][:50]
            bleu = sacrebleu.corpus_bleu(translations, [refs])
            print(f"  > BLEU:       {bleu.score:.2f}")
            
        # Return first translation to check for differences
        return translations[0]

    # --- RUN COMPARISONS ---
    
    # 1. Baseline (Standard Beam Search)
    t1 = run_method("Baseline (No Pruning)", beam_search_decode)

    # 2. Relative Threshold Pruning (rp=1.5 implies strict window around max score)
    #    Since scores are negative, rp > 1 makes the threshold lower (more negative).
    t2 = run_method("Relative Pruning (rp=1.5)", beam_search_decode_relative_pruning, rp=1.5)

    # 3. Absolute Threshold Pruning (ap=5.0 means discard if score < max - 5)
    t3 = run_method("Absolute Pruning (ap=5.0)", beam_search_decode_absolute_pruning, ap=5.0)

    print("\n--- Example Translation Comparison (Sentence 1) ---")
    print(f"Baseline:   {t1}")
    print(f"Relative:   {t2}")
    print(f"Absolute:   {t3}")

if __name__ == '__main__':
    main()
