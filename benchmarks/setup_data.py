#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "numpy",
#   "h5py",
#   "datasets",
# ]
# ///

# Redirect all HuggingFace and temp downloads to data partition before any imports
import os
_data_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '.hf_cache')
os.makedirs(_data_cache, exist_ok=True)
os.environ["HF_HOME"] = _data_cache
os.environ["HF_DATASETS_CACHE"] = _data_cache
os.environ["TMPDIR"] = _data_cache
os.environ["TEMP"] = _data_cache
os.environ["TMP"] = _data_cache

import argparse
import numpy as np
import sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from bench_utils import (
    DATASET_PARAMS,
    DATASET_HDF5_NAMES,
    ANGULAR_DATASETS,
    DATA_DIR,
    l2_normalize,
    read_hdf5_data,
)
np.random.seed(42)

RAW_DATA = './data/' # '../../PDX/benchmarks/datasets/downloaded/'

def setup_hdf5_dataset(dataset_id, raw_data_path):
    if dataset_id not in DATASET_PARAMS:
        print(f"Error: Unknown dataset '{dataset_id}'")
        print(f"Available datasets: {', '.join(sorted(DATASET_PARAMS.keys()))}")
        sys.exit(1)

    if dataset_id not in DATASET_HDF5_NAMES:
        print(f"Error: Dataset '{dataset_id}' does not have HDF5 mapping")
        sys.exit(1)

    n, d = DATASET_PARAMS[dataset_id]
    hdf5_name = DATASET_HDF5_NAMES[dataset_id]
    needs_normalization = dataset_id in ANGULAR_DATASETS

    print(f"Processing dataset: {dataset_id}")
    print(f"  HDF5 file: {hdf5_name}.hdf5")
    print(f"  Expected shape: ({n:,}, {d})")
    print(f"  L2 normalization: {'Yes' if needs_normalization else 'No'}")

    train, test = read_hdf5_data(raw_data_path, hdf5_name)

    if needs_normalization:
        train = l2_normalize(train.astype(np.float32))
        test = l2_normalize(test.astype(np.float32))
    else:
        train = train.astype(np.float32)
        test = test.astype(np.float32)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / f"data_{dataset_id}.bin"
    test_path = DATA_DIR / f"data_{dataset_id}_test.bin"

    train.tofile(train_path)
    test.tofile(test_path)

    print(f"  Saved train: {len(train):,} vectors -> {train_path}")
    print(f"  Saved test:  {len(test):,} vectors -> {test_path}")
    print(f"  Actual dims: {len(train[0])}")


def setup_cohere_dataset(full=False):
    """Download and process Cohere Wikipedia embeddings from HuggingFace."""
    from datasets import load_dataset

    USE_SEPARATE_QUERIES = True  # Use separate queries subset
    MAX_TRAIN_SAMPLES = 50_000_000 if full else 10_000_000
    MAX_TEST_SAMPLES = 1000

    embedding_dim = 1024
    # Process in chunks to avoid PyArrow list offset overflow on large streams
    chunk_size = 10_000_000
    cache_dir = str(DATA_DIR / ".hf_cache")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_file = DATA_DIR / "data_cohere50m.bin" if full else DATA_DIR / "data_cohere.bin"
    test_file = DATA_DIR / "data_cohere50m_test.bin" if full else DATA_DIR / "data_cohere_test.bin"

    # Resume support: check existing file size
    start_count = 0
    if train_file.exists():
        file_size = train_file.stat().st_size
        expected_row_bytes = embedding_dim * np.dtype(np.float32).itemsize
        start_count = file_size // expected_row_bytes
        if file_size % expected_row_bytes != 0:
            print(f"WARNING: File size not aligned to row size, truncating to {start_count:,} complete rows")
            with open(train_file, 'r+b') as f:
                f.truncate(start_count * expected_row_bytes)

    print("Downloading Cohere Wikipedia embeddings from HuggingFace...")
    print(f"  Max train samples: {MAX_TRAIN_SAMPLES:,}")
    print(f"  Max test samples:  {MAX_TEST_SAMPLES:,}")
    if start_count > 0:
        print(f"  Resuming from:     {start_count:,}")

    if USE_SEPARATE_QUERIES:
        print(f"\nLoading train embeddings (max {MAX_TRAIN_SAMPLES:,})...")

        file_mode = 'ab' if start_count > 0 else 'wb'
        total_count = start_count

        with open(train_file, file_mode) as f_train:
            for chunk_start in range(start_count, MAX_TRAIN_SAMPLES, chunk_size):
                chunk_end = min(chunk_start + chunk_size, MAX_TRAIN_SAMPLES)
                print(f"\nLoading chunk [{chunk_start:,}:{chunk_end:,}]...")

                ds = load_dataset(
                    "Cohere/msmarco-v2.1-embed-english-v3", "passages",
                    split=f"train[{chunk_start}:{chunk_end}]",
                    cache_dir=cache_dir,
                )

                embeddings = np.array(ds['emb'], dtype=np.float32)
                embeddings.tofile(f_train)
                f_train.flush()
                total_count += len(embeddings)
                print(f"Wrote {total_count:,} / {MAX_TRAIN_SAMPLES:,} train samples")
                del ds, embeddings

        train_count = total_count
        print(f"Train complete: {train_count:,} samples")

        print(f"\nLoading test queries from 'queries' subset (max {MAX_TEST_SAMPLES})...")
        dataset_test = load_dataset("Cohere/msmarco-v2.1-embed-english-v3", "queries", split="test", cache_dir=cache_dir)

        test_embeddings = np.array(dataset_test['emb'], dtype=np.float32)
        test_idx = min(MAX_TEST_SAMPLES, len(test_embeddings))

        print(f"Downloaded {len(test_embeddings)} queries, using first {test_idx}")

        test_embeddings[:test_idx].tofile(test_file)
        print(f"Test complete: {test_idx} samples")

    else:
        print(f"Using old behavior: taking test from train samples (max train: {MAX_TRAIN_SAMPLES:,})")
        dataset = load_dataset("Cohere/msmarco-v2.1-embed-english-v3", "passages", split="train", streaming=True)

        write_buffer = np.empty((write_chunk_size, embedding_dim), dtype=np.float32)
        buffer_idx = 0
        total_count = 0
        test_size = MAX_TEST_SAMPLES

        test_buffer = np.empty((test_size, embedding_dim), dtype=np.float32)
        test_idx = 0

        with open(train_file, 'wb') as f_train:
            for batch in dataset.iter(batch_size=batch_size):
                print(f'Receiving batch (total: {total_count:,}/{MAX_TRAIN_SAMPLES:,})')
                batch_embeddings = np.array(batch['emb'], dtype=np.float32)
                batch_len = len(batch_embeddings)

                for i in range(batch_len):
                    if total_count >= MAX_TRAIN_SAMPLES:
                        break

                    write_buffer[buffer_idx] = batch_embeddings[i]
                    buffer_idx += 1
                    total_count += 1

                    if buffer_idx >= write_chunk_size:
                        write_buffer.tofile(f_train)
                        print(f"Processed and wrote {total_count:,} samples...")
                        buffer_idx = 0

                if total_count >= MAX_TRAIN_SAMPLES:
                    break

            remaining = buffer_idx

            if remaining > test_size:
                write_buffer[:remaining - test_size].tofile(f_train)
                test_buffer[:test_size] = write_buffer[remaining - test_size:remaining]
                test_idx = test_size
            else:
                test_buffer[:remaining] = write_buffer[:remaining]
                test_idx = remaining

        test_buffer[:test_idx].tofile(test_file)

        train_count = total_count - test_idx
        print(f"Done! Wrote {train_count:,} train samples and {test_idx} test samples")

    print(f'\ncohere: train={train_count:,}, test={test_idx}, dim={embedding_dim}')
    print(f"Saved train: {train_file}")
    print(f"Saved test:  {test_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process datasets for SuperKMeans benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
{chr(10).join(f"  - {name:15s} ({n:>10,} x {d:>5,})" for name, (n, d) in sorted(DATASET_PARAMS.items()))}

Examples:
  %(prog)s mxbai
  %(prog)s clip
  %(prog)s cohere
  %(prog)s --data-dir /path/to/hdf5/files sift
        """
    )

    parser.add_argument(
        "dataset",
        help="Dataset identifier (e.g., mxbai, clip, cohere, sift)"
    )

    parser.add_argument(
        "--data-dir",
        default=RAW_DATA,
        help=f"Path to directory containing raw HDF5 files (default: {RAW_DATA})"
    )

    args = parser.parse_args()

    if args.dataset == "cohere":
        setup_cohere_dataset()
    elif args.dataset == "cohere50m":
        setup_cohere_dataset(True)
    else:
        setup_hdf5_dataset(args.dataset, args.data_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()
