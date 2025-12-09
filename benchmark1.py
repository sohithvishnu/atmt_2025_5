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
from seq2seq.decode import beam_search_decode_len_pen  # The new version

def get_args():
    parser = argparse.ArgumentParser('Benchmark Script')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--input', required=True)
    parser.add_argument('--src-tokenizer', required=True)
    parser.add_argument('--tgt-tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--max-len', default=128, type=int)
    parser.add_argument('--reference', type=str,
                        help='Path to reference file (required for BLEU)')
    parser.add_argument('--output', type=str,
                        help='Write translations to file')
    return parser.parse_args()

def main():
    args = get_args()
    torch.manual_seed(args.seed)

    print("Loading model...")
    state_dict = torch.load(args.checkpoint_path,
                            map_location=lambda s, l: default_restore_location(s, 'cpu'))

    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})

    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)

    model = models.build_model(args_loaded, src_tokenizer, tgt_tokenizer)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])

    DEVICE = 'cuda' if args.cuda else 'cpu'
    PAD = src_tokenizer.pad_id()
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    print("Reading input data...")
    with open(args.input, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()][:50]
    def postprocess_ids(ids, pad, bos, eos):
        """Remove leading BOS, truncate at first EOS, remove PADs."""
        if isinstance(ids, torch.Tensor):
           ids = ids.tolist()
        # remove leading BOS if present
        if len(ids) > 0 and ids[0] == bos:
           ids = ids[1:]
     # truncate at EOS (do not include EOS)
        if eos in ids:
           ids = ids[:ids.index(eos)]
        # remove PAD tokens (typically trailing, but remove any)
        ids = [i for i in ids if i != pad]
        return ids
    def decode_sentence(tokenizer: spm.SentencePieceProcessor, sentence_ids):
        """Convert token ids to a detokenized string using the target tokenizer."""
        ids = postprocess_ids(sentence_ids, PAD, BOS, EOS)
        # Use tokenizer.Decode to produce properly detokenized text
        return tokenizer.Decode(ids)	
    print(f"Preparing {len(src_lines)} individual sentences...")

    benchmark_data = []

    make_batch = utils.make_batch_input(device=DEVICE,
                                        pad=src_tokenizer.pad_id(),
                                        max_seq_len=args.max_len)

    for line in src_lines:
        encoded_ids = src_tokenizer.Encode(line, out_type=int, add_eos=True)
        src_tensor = torch.tensor(encoded_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        dummy_y = torch.full_like(src_tensor, fill_value=src_tokenizer.pad_id())
        src_tokens, _, _, src_pad_mask, _ = make_batch(src_tensor, dummy_y)
        benchmark_data.append((src_tokens, src_pad_mask))

    print(f"\n--- Starting Benchmark on {len(benchmark_data)} sentences ---")

    def run_benchmark(alpha):
        print(f"\nTesting Alpha: {alpha}")
        beam_k = 5
        translations = []
        total_tokens = 0
        start_time = time.time()
        for src_tokens, src_pad_mask in benchmark_data:
            predictions = beam_search_decode_len_pen(model, src_tokens, src_pad_mask,
                               30, tgt_tokenizer, args_loaded, DEVICE, beam_k,alpha)
            best_seq_ids = predictions[0]
            total_tokens += len(best_seq_ids)
            for sent in predictions:
               translation = decode_sentence(tgt_tokenizer, sent)
               translations.append(translation)
        total_time = time.time() - start_time
        avg_time = total_time / len(benchmark_data)
        avg_len = total_tokens / len(benchmark_data)
        
        print(f"  > Avg Time:   {avg_time:.4f}s / sentence")
        print(f"  > Avg Length: {avg_len:.2f} tokens")
        # BLEU COMPUTATION
        if args.reference:
            with open(args.reference, encoding='utf-8') as ref_file:
                references = [line.strip() for line in ref_file if line.strip()][:50]

            if len(references) != len(translations):
                raise ValueError(
                    f"Reference ({len(references)}) and hypothesis ({len(translations)}) line counts do not match."
                )

            bleu = sacrebleu.corpus_bleu(translations, [references])
            print(f"BLEU score: {bleu.score:.2f}")

    for k in [0.0,0.3,0.7,1.0]:
        run_benchmark(k)


if __name__ == '__main__':
    main()
