import numpy as np
import os
import h5py

np.random.seed(42)

RAW_DATA = '../../PDX/benchmarks/datasets/downloaded/'

def l2_normalize(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)

def read_hdf5_data(dataset):
    hdf5_file_name = os.path.join(RAW_DATA, dataset + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32), np.array(hdf5_file["test"], dtype=np.float32)


if __name__ == "__main__":
    # num_dimensions = 987
    # num_dimensions = 1024
    # num_vectors = 262144
    # num_centroids = 2048
    #
    # print(f'Bench gen: \n- D={num_dimensions}\n- num_centroids={num_centroids}\n- dataset=RANDOM')
    # data, _ = sklearn.datasets.make_blobs(n_samples=num_vectors, n_features=num_dimensions, centers=num_centroids, random_state=1)
    # data = data.astype(np.float32)
    #
    # data.tofile("./data/data_random.bin")

    # train, test = read_hdf5_data("glove-200-angular")
    # train = l2_normalize(train.astype(np.float32))
    # train.tofile("./data/data_glove200.bin")
    # test = l2_normalize(test.astype(np.float32))
    # test.tofile("./data/data_glove200_test.bin")
    # print('glove200', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("glove-100-angular")
    # train = train.astype(np.float32)
    # train.tofile("./data/data_glove100.bin")
    # test = test.astype(np.float32)
    # test.tofile("./data/data_glove100_test.bin")
    # print('glove100', len(train), len(test), len(train[0]))
    #
    # train, test = read_hdf5_data("glove-50-angular")
    # train = train.astype(np.float32)
    # train.tofile("./data/data_glove50.bin")
    # test = test.astype(np.float32)
    # test.tofile("./data/data_glove50_test.bin")
    # print('glove50', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("imagenet-clip-512-normalized")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_clip.bin")
    # test.tofile("./data/data_clip_test.bin")
    # print('clip', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("yahoo-minilm-384-normalized")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_yahoo.bin")
    # test.tofile("./data/data_yahoo_test.bin")
    # print('yahoo', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("llama-128-ip")
    # train = l2_normalize(train.astype(np.float32))
    # test = l2_normalize(test.astype(np.float32))
    # train.tofile("./data/data_llama.bin")
    # test.tofile("./data/data_llama_test.bin")
    # print('llama', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("yi-128-ip")
    # train = l2_normalize(train.astype(np.float32))
    # test = l2_normalize(test.astype(np.float32))
    # train.tofile("./data/data_yi.bin")
    # test.tofile("./data/data_yi_test.bin")
    # print('yi', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("yandex-200-cosine")
    # train = l2_normalize(train.astype(np.float32))
    # test = l2_normalize(test.astype(np.float32))
    # train.tofile("./data/data_yandex.bin")
    # test.tofile("./data/data_yandex_test.bin")
    # print('yandex', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("agnews-mxbai-1024-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_mxbai.bin")
    # test.tofile("./data/data_mxbai_test.bin")
    # print('mxbai', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("openai-1536-angular")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_openai.bin")
    # test.tofile("./data/data_openai_test.bin")
    # print('openai', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("instructorxl-arxiv-768")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_arxiv.bin")
    # test.tofile("./data/data_arxiv_test.bin")
    # print('arxiv', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("sift-128-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_sift.bin")
    # test.tofile("./data/data_sift_test.bin")
    # print('sift', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("fashion-mnist-784-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_fmnist.bin")
    # test.tofile("./data/data_fmnist_test.bin")
    # print('fmnist', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("gist-960-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_gist.bin")
    # test.tofile("./data/data_gist_test.bin")
    # print('gist', len(train), len(test), len(train[0]))

    train, test = read_hdf5_data("contriever-768")
    train = l2_normalize(train.astype(np.float32))
    test = l2_normalize(test.astype(np.float32))
    train.tofile("./data/data_contriever.bin")
    test.tofile("./data/data_contriever_test.bin")
    print('contriever', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("simplewiki-openai-3072-normalized")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_wiki.bin")
    # test.tofile("./data/data_wiki_test.bin")
    # print('wiki', len(train), len(test), len(train[0]))

    # Download and process Cohere Wikipedia embeddings from HuggingFace
    # from datasets import load_dataset
    #
    # # Configuration flags
    # USE_SEPARATE_QUERIES = True  # If True, load queries from separate "queries" subset with split="test"
    #                               # If False, use old behavior (take test from last samples of train)
    # MAX_TRAIN_SAMPLES = 10_000_000  # Limit train to 10M samples
    # MAX_TEST_SAMPLES = 1000  # Limit test/queries to 1000 samples
    #
    # print("Downloading Cohere Wikipedia embeddings from HuggingFace...")
    #
    # batch_size = 10000  # Fetch 10k embeddings per batch from HuggingFace
    # write_chunk_size = 100000  # Write to disk every 100k embeddings
    # embedding_dim = 1024
    #
    # train_file = "./data/data_cohere.bin"
    # test_file = "./data/data_cohere_test.bin"
    #
    # if USE_SEPARATE_QUERIES:
    #     print(f"Loading train embeddings (max {MAX_TRAIN_SAMPLES:,})...")
    #     dataset_train = load_dataset("Cohere/msmarco-v2.1-embed-english-v3", "passages", split="train", streaming=True)
    #
    #     # Process train embeddings
    #     write_buffer = np.empty((write_chunk_size, embedding_dim), dtype=np.float32)
    #     buffer_idx = 0
    #     total_count = 0
    #
    #     with open(train_file, 'wb') as f_train:
    #         for batch in dataset_train.iter(batch_size=batch_size):
    #             print(f'Receiving train batch (total: {total_count:,}/{MAX_TRAIN_SAMPLES:,})')
    #             batch_embeddings = np.array(batch['emb'], dtype=np.float32)
    #             batch_len = len(batch_embeddings)
    #
    #             for i in range(batch_len):
    #                 if total_count >= MAX_TRAIN_SAMPLES:
    #                     break
    #
    #                 write_buffer[buffer_idx] = batch_embeddings[i]
    #                 buffer_idx += 1
    #                 total_count += 1
    #
    #                 # Write chunk to file when buffer is full
    #                 if buffer_idx >= write_chunk_size:
    #                     write_buffer.tofile(f_train)
    #                     print(f"Processed and wrote {total_count:,} train samples...")
    #                     buffer_idx = 0
    #
    #             # Break outer loop if we've reached max
    #             if total_count >= MAX_TRAIN_SAMPLES:
    #                 break
    #
    #         # Write remaining samples in buffer
    #         if buffer_idx > 0:
    #             write_buffer[:buffer_idx].tofile(f_train)
    #
    #     train_count = total_count
    #     print(f"Train complete: {train_count:,} samples")
    #
    #     # Load test embeddings from separate "queries" subset (non-streaming since it's small)
    #     print(f"\nLoading test queries from 'queries' subset (max {MAX_TEST_SAMPLES})...")
    #     dataset_test = load_dataset("Cohere/msmarco-v2.1-embed-english-v3", "queries", split="test")
    #
    #     # Get embeddings directly (non-streaming)
    #     test_embeddings = np.array(dataset_test['emb'], dtype=np.float32)
    #     test_idx = min(MAX_TEST_SAMPLES, len(test_embeddings))
    #
    #     print(f"Downloaded {len(test_embeddings)} queries, using first {test_idx}")
    #
    #     # Write test file
    #     test_embeddings[:test_idx].tofile(test_file)
    #     print(f"Test complete: {test_idx} samples")
    #
    # else:
    #     print(f"Using old behavior: taking test from train samples (max train: {MAX_TRAIN_SAMPLES:,})")
    #     dataset = load_dataset("Cohere/msmarco-v2.1-embed-english-v3", "passages", split="train", streaming=True)
    #
    #     # Preallocate write buffer
    #     write_buffer = np.empty((write_chunk_size, embedding_dim), dtype=np.float32)
    #     buffer_idx = 0
    #     total_count = 0
    #     test_size = MAX_TEST_SAMPLES
    #
    #     # Preallocate test buffer
    #     test_buffer = np.empty((test_size, embedding_dim), dtype=np.float32)
    #     test_idx = 0
    #
    #     with open(train_file, 'wb') as f_train:
    #         # Use iter with batch_size for faster streaming
    #         for batch in dataset.iter(batch_size=batch_size):
    #             print(f'Receiving batch (total: {total_count:,}/{MAX_TRAIN_SAMPLES:,})')
    #             batch_embeddings = np.array(batch['emb'], dtype=np.float32)
    #             batch_len = len(batch_embeddings)
    #
    #             for i in range(batch_len):
    #                 if total_count >= MAX_TRAIN_SAMPLES:
    #                     break
    #
    #                 write_buffer[buffer_idx] = batch_embeddings[i]
    #                 buffer_idx += 1
    #                 total_count += 1
    #
    #                 # Write chunk to file when buffer is full
    #                 if buffer_idx >= write_chunk_size:
    #                     write_buffer.tofile(f_train)
    #                     print(f"Processed and wrote {total_count:,} samples...")
    #                     buffer_idx = 0
    #
    #             # Break outer loop if we've reached max
    #             if total_count >= MAX_TRAIN_SAMPLES:
    #                 break
    #
    #         # Handle remaining samples
    #         remaining = buffer_idx
    #
    #         if remaining > test_size:
    #             # Write the part before the last test_size samples
    #             write_buffer[:remaining - test_size].tofile(f_train)
    #             # Copy last test_size to test buffer
    #             test_buffer[:test_size] = write_buffer[remaining - test_size:remaining]
    #             test_idx = test_size
    #         else:
    #             # All remaining samples go to test
    #             test_buffer[:remaining] = write_buffer[:remaining]
    #             test_idx = remaining
    #
    #     # Write test file
    #     test_buffer[:test_idx].tofile(test_file)
    #
    #     train_count = total_count - test_idx
    #     print(f"Done! Wrote {train_count:,} train samples and {test_idx} test samples")
    #
    # print(f'cohere: train={train_count:,}, test={test_idx}, dim={embedding_dim}')
    # print("Done")
