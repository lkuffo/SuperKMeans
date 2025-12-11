import numpy as np
import os
import h5py

np.random.seed(42)

RAW_DATA = '../../PDX/benchmarks/datasets/downloaded/'


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
    # train = train.astype(np.float32)
    # train.tofile("./data/data_glove200.bin")
    # test = test.astype(np.float32)
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

    # train, test = read_hdf5_data("contriever-768")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_contriever.bin")
    # test.tofile("./data/data_contriever_test.bin")
    # print('contriever', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("simplewiki-openai-3072-normalized")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("./data/data_wiki.bin")
    # test.tofile("./data/data_wiki_test.bin")
    # print('wiki', len(train), len(test), len(train[0]))

    # Download and process Cohere Wikipedia embeddings from HuggingFace
    from datasets import load_dataset

    print("Downloading Cohere Wikipedia embeddings from HuggingFace...")
    dataset = load_dataset("Cohere/wikipedia-22-12-en-embeddings", split="train", streaming=True)

    # Process and write embeddings in chunks directly to file
    print("Extracting and writing embeddings in batches...")

    batch_size = 10000  # Fetch 10k embeddings per batch from HuggingFace
    write_chunk_size = 100000  # Write to disk every 100k embeddings
    embedding_dim = 768

    train_file = "./data/data_cohere.bin"
    test_file = "./data/data_cohere_test.bin"

    # Preallocate write buffer
    write_buffer = np.empty((write_chunk_size, embedding_dim), dtype=np.float32)
    buffer_idx = 0
    total_count = 0
    test_size = 1000

    # Preallocate test buffer
    test_buffer = np.empty((test_size, embedding_dim), dtype=np.float32)
    test_idx = 0

    with open(train_file, 'wb') as f_train:
        # Use iter with batch_size for faster streaming
        for batch in dataset.iter(batch_size=batch_size):
            print('Receiving batch')
            batch_embeddings = np.array(batch['emb'], dtype=np.float32)
            batch_len = len(batch_embeddings)

            for i in range(batch_len):
                write_buffer[buffer_idx] = batch_embeddings[i]
                buffer_idx += 1
                total_count += 1

                # Write chunk to file when buffer is full
                if buffer_idx >= write_chunk_size:
                    write_buffer.tofile(f_train)
                    print(f"Processed and wrote {total_count} samples...")
                    buffer_idx = 0

        # Handle remaining samples
        remaining = buffer_idx

        if remaining > test_size:
            # Write the part before the last test_size samples
            write_buffer[:remaining - test_size].tofile(f_train)
            # Copy last test_size to test buffer
            test_buffer[:test_size] = write_buffer[remaining - test_size:remaining]
            test_idx = test_size
        else:
            # All remaining samples go to test
            test_buffer[:remaining] = write_buffer[:remaining]
            test_idx = remaining

    # Write test file
    test_buffer[:test_idx].tofile(test_file)

    train_count = total_count - test_idx
    print(f"Done! Wrote {train_count} train samples and {test_idx} test samples")
    print(f'cohere: train={train_count}, test={test_idx}, dim={embedding_dim}')

    print("Done")
