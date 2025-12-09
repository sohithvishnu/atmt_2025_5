import os
import pickle
import logging
import argparse
import sentencepiece as spm

from seq2seq.data.tokenizer import BPETokenizer

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess text data for training.")
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')

    # File paths
    parser.add_argument("--raw-data", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--dest-dir", type=str, default="./", help="Directory to save the processed data.")
    parser.add_argument("--model-dir", type=str, default="./models", help="Directory to save the trained tokenization model and vocab.")
    parser.add_argument("--src-model", type=str, default=None, help="Path to the Source Language SentencePiece tokenization model. If none, creates a tokenization model from the training-split.")
    parser.add_argument("--tgt-model", type=str, default=None, help="Path to the Target Language SentencePiece tokenization model. If none, creates a tokenization model from the training-split.")
    parser.add_argument("--force-train", action="store_true", help="Force training even if a model already exists.")

    # File prefixes (optional)
    parser.add_argument('--train-prefix', default=None, metavar='FP', help='raw train file prefix (without .lang extension)')
    parser.add_argument('--tiny-train-prefix', default=None, metavar='FP', help='raw tiny train file prefix (without .lang extension)')
    parser.add_argument('--valid-prefix', default=None, metavar='FP', help='raw valid file prefix (without .lang extension)')
    parser.add_argument('--test-prefix', default=None, metavar='FP', help='raw test file prefix (without .lang extension)')
    parser.add_argument("--ignore-existing", action="store_true", help="Skip processing of raw-files if the output file already exists. Useful for resuming.")

    parser.add_argument("--src-vocab-size", type=int, default=32000, help="Vocabulary size for Source Language SentencePiece.")
    parser.add_argument("--tgt-vocab-size", type=int, default=32000, help="Vocabulary size for Target Language SentencePiece.")
    
    parser.add_argument("--quiet", action="store_true", help="Suppress logging output.")
    parser.add_argument("--joint-bpe", action="store_true", help="Train a single joint BPE model on source + target.")

    # Special tokens (names for logging; IDs are fixed below)
    parser.add_argument("--eos-token", type=str, default="</s>", help="End of sentence token.")
    parser.add_argument("--bos-token", type=str, default="<s>", help="Beginning of sentence token.")
    parser.add_argument("--pad-token", type=str, default="<pad>", help="Padding token.")
    parser.add_argument("--unk-token", type=str, default="<unk>", help="Unknown token.")
    
    return parser.parse_args()


def make_binary_dataset(input_file, output_file, preprocessor: BPETokenizer, append_eos=True, ignore_existing=False, quiet=False):
    if os.path.exists(output_file) and not ignore_existing:
        logging.info(f"File {output_file} already exists, skipping...")
        return
    
    nsent, ntok = 0, 0
    unk_counter = 0

    def unk_consumer(idx):
        nonlocal unk_counter
        if idx == preprocessor.tokenizer.unk_id():
            unk_counter += 1

    tokens_list = []
    with open(input_file, 'r', encoding='utf-8') as inf:
        for line in inf:
            tokens = preprocessor.encode_to_tensor(line.strip(), append_eos, consumer=unk_consumer)
            nsent += 1
            ntok += len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.DEFAULT_PROTOCOL)
        if not quiet:
            unk_pct = 0.0 if ntok == 0 else (100.0 * unk_counter / ntok)
            logging.info(
                "Built a binary dataset for %s: %d sentences, %d tokens, %.3f%% <unk> (unk_id=%d)",
                input_file, nsent, ntok, unk_pct, preprocessor.tokenizer.unk_id()
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    # If user passed the SAME model path for src/tgt, they likely intend joint BPE.
    if (not args.joint_bpe) and args.src_model and args.tgt_model and os.path.abspath(args.src_model) == os.path.abspath(args.tgt_model):
        logging.info("Detected identical --src-model and --tgt-model. Enabling --joint-bpe.")
        args.joint_bpe = True

    # Resolve model file paths if not provided
    tgt_tokenizer_model = args.tgt_model if args.tgt_model \
        else os.path.join(args.model_dir, f"{args.target_lang}-bpe-{args.tgt_vocab_size}.model")
    src_tokenizer_model = args.src_model if args.src_model \
        else os.path.join(args.model_dir, f"{args.source_lang}-bpe-{args.src_vocab_size}.model")
    
    os.makedirs(args.dest_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Instantiate tokenizers
    src_processor = BPETokenizer(
        language=args.source_lang,
        vocab_size=args.src_vocab_size,
        eos=args.eos_token,
        bos=args.bos_token,
        pad=args.pad_token,
        unk=args.unk_token
    )
    tgt_processor = BPETokenizer(
        language=args.target_lang,
        vocab_size=args.tgt_vocab_size,
        eos=args.eos_token,
        bos=args.bos_token,
        pad=args.pad_token,
        unk=args.unk_token
    )

    # ===========================
    # JOINT BPE (safe special IDs)
    # ===========================
    if args.joint_bpe:
        joint_model_prefix = os.path.join(args.model_dir, f"joint-bpe-{args.src_vocab_size}")
        joint_model_file   = joint_model_prefix + ".model"
        joint_vocab_file   = joint_model_prefix + ".vocab"
        combined_file      = os.path.join(args.model_dir, "joint_train.txt")

        # Build combined training file
        if not os.path.exists(combined_file) or args.force_train:
            if args.train_prefix is None:
                raise ValueError("--train-prefix is required for joint BPE training.")
            src_file = os.path.join(args.raw_data, f"{args.train_prefix}.{args.source_lang}")
            tgt_file = os.path.join(args.raw_data, f"{args.train_prefix}.{args.target_lang}")
            with open(combined_file, "w", encoding="utf-8") as out, \
                 open(src_file, "r", encoding="utf-8") as s, \
                 open(tgt_file, "r", encoding="utf-8") as t:
                for src_line, tgt_line in zip(s, t):
                    out.write(src_line.strip() + "\n")
                    out.write(tgt_line.strip() + "\n")
            logging.info(f"Created combined training file: {combined_file}")

        # Train with EXPLICIT special IDs: unk=0, bos=1, eos=2, pad=3 (NO -1!)
        if not os.path.exists(joint_model_file) or args.force_train:
            spm.SentencePieceTrainer.train(
                input=combined_file,
                model_prefix=joint_model_prefix,
                vocab_size=args.src_vocab_size,
                character_coverage=1.0,
                model_type="bpe",
                unk_id=0,
                bos_id=1,
                eos_id=2,
                pad_id=3,
                unk_piece=args.unk_token,
                bos_piece=args.bos_token,
                eos_piece=args.eos_token,
                pad_piece=args.pad_token,
            )
            logging.info(f"✅ Trained joint BPE model: {joint_model_file}")
        else:
            logging.info(f"Using existing joint model: {joint_model_file}")

        # Load joint model for both
        src_processor.load(model_path=joint_model_file)
        tgt_processor.load(model_path=joint_model_file)

        # Log special IDs (sanity)
        logging.info("[JOINT] ids: unk=%d bos=%d eos=%d pad=%d",
                     src_processor.tokenizer.unk_id(),
                     src_processor.tokenizer.bos_id(),
                     src_processor.tokenizer.eos_id(),
                     src_processor.tokenizer.pad_id())

        # Save vocabs for consistency
        src_processor.save_vocab(args.model_dir)
        tgt_processor.save_vocab(args.model_dir)

        logging.info("Loaded shared Joint BPE tokenizer for both source and target languages.")

    # ===========================
    # SEPARATE BPE
    # ===========================
    else:
        # Source
        if (not os.path.exists(src_tokenizer_model)) or args.force_train:
            if args.train_prefix is None:
                raise ValueError("No training data for source tokenizer. Provide --train-prefix.")
            src_processor.train_tokenizer(
                training_data=os.path.join(args.raw_data, f"{args.train_prefix}.{args.source_lang}"),
                model_dir=args.model_dir
            )
            logging.info(f"Trained SentencePiece model for {args.source_lang}")
        else:
            src_processor.load(model_path=src_tokenizer_model)
            logging.info(f"Loaded SentencePiece model for {args.source_lang}")
        src_processor.save_vocab(args.model_dir)

        # Target
        if (not os.path.exists(tgt_tokenizer_model)) or args.force_train:
            if args.train_prefix is None:
                raise ValueError("No training data for target tokenizer. Provide --train-prefix.")
            tgt_processor.train_tokenizer(
                training_data=os.path.join(args.raw_data, f"{args.train_prefix}.{args.target_lang}"),
                model_dir=args.model_dir
            )
            logging.info(f"Trained SentencePiece model for {args.target_lang}")
        else:
            tgt_processor.load(model_path=tgt_tokenizer_model)
            logging.info(f"Loaded SentencePiece model for {args.target_lang}")
        tgt_processor.save_vocab(args.model_dir)

        logging.info("[SRC] ids: unk=%d bos=%d eos=%d pad=%d",
                     src_processor.tokenizer.unk_id(),
                     src_processor.tokenizer.bos_id(),
                     src_processor.tokenizer.eos_id(),
                     src_processor.tokenizer.pad_id())
        logging.info("[TGT] ids: unk=%d bos=%d eos=%d pad=%d",
                     tgt_processor.tokenizer.unk_id(),
                     tgt_processor.tokenizer.bos_id(),
                     tgt_processor.tokenizer.eos_id(),
                     tgt_processor.tokenizer.pad_id())

    # ===========================
    # Build binary splits
    # ===========================
    def make_split_datasets(lang, pre_processor):
        if args.train_prefix is not None:
            make_binary_dataset(
                input_file=os.path.join(args.raw_data, f"{args.train_prefix}.{lang}"),
                output_file=os.path.join(args.dest_dir, f"train.{lang}"),
                preprocessor=pre_processor,
                ignore_existing=args.ignore_existing,
                quiet=args.quiet
            )
        if args.tiny_train_prefix is not None:
            make_binary_dataset(
                input_file=os.path.join(args.raw_data, f"{args.tiny_train_prefix}.{lang}"),
                output_file=os.path.join(args.dest_dir, f"tiny_train.{lang}"),
                preprocessor=pre_processor,
                ignore_existing=args.ignore_existing,
                quiet=args.quiet
            )
        if args.valid_prefix is not None:
            make_binary_dataset(
                input_file=os.path.join(args.raw_data, f"{args.valid_prefix}.{lang}"),
                output_file=os.path.join(args.dest_dir, f"valid.{lang}"),
                preprocessor=pre_processor,
                ignore_existing=args.ignore_existing,
                quiet=args.quiet
            )
        if args.test_prefix is not None:
            make_binary_dataset(
                input_file=os.path.join(args.raw_data, f"{args.test_prefix}.{lang}"),
                output_file=os.path.join(args.dest_dir, f"{args.test_prefix}.{lang}"),
                preprocessor=pre_processor,
                ignore_existing=args.ignore_existing,
                quiet=args.quiet
            )
    
    make_split_datasets(args.source_lang, src_processor)
    make_split_datasets(args.target_lang, tgt_processor)

    logging.info('Data processing complete!')
