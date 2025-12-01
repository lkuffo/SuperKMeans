import sklearn.datasets
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
    # data.tofile("data_random.bin")

    # num_dimensions = 1024
    # num_vectors = 720896
    # rng = np.random.default_rng()
    # training_sample_idxs = rng.choice(len(train), size=num_vectors, replace=False)
    # training_sample_idxs.sort()
    # data = train[training_sample_idxs].astype(np.float32)
    # data.tofile("data_mxbai.bin")

    # train, test = read_hdf5_data("glove-100-angular")
    # train = train.astype(np.float32)
    # train.tofile("data_glove100.bin")
    # test = test.astype(np.float32)
    # test.tofile("data_glove100_test.bin")
    # print('glove100', len(train), len(test), len(train[0]))
    #
    # train, test = read_hdf5_data("glove-50-angular")
    # train = train.astype(np.float32)
    # train.tofile("data_glove50.bin")
    # test = test.astype(np.float32)
    # test.tofile("data_glove50_test.bin")
    # print('glove50', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("agnews-mxbai-1024-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # # train.tofile("data_mxbai.bin")
    # test.tofile("data_mxbai_test.bin")
    # print('mxbai', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("agnews-mxbai-1024-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # # train.tofile("data_mxbai.bin")
    # test.tofile("data_mxbai_test.bin")
    # print('mxbai', len(train), len(test), len(train[0]))
    #
    # train, test = read_hdf5_data("openai-1536-angular")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # # train.tofile("data_openai.bin")
    # test.tofile("data_openai_test.bin")
    # print('openai', len(train), len(test), len(train[0]))
    #
    # train, test = read_hdf5_data("instructorxl-arxiv-768")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # # train.tofile("data_arxiv.bin")
    # test.tofile("data_arxiv_test.bin")
    # print('arxiv', len(train), len(test), len(train[0]))
    #
    # train, test = read_hdf5_data("sift-128-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # # train.tofile("data_sift.bin")
    # test.tofile("data_sift_test.bin")
    # print('sift', len(train), len(test), len(train[0]))
    #
    # train, test = read_hdf5_data("fashion-mnist-784-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # # train.tofile("data_fmnist.bin")
    # test.tofile("data_fmnist_test.bin")
    # print('fmnist', len(train), len(test), len(train[0]))

    # train, test = read_hdf5_data("gist-960-euclidean")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("data_gist.bin")
    # test.tofile("data_gist_test.bin")
    # print('gist', len(train), len(test), len(train[0]))
    #
    # train, test = read_hdf5_data("contriever-768")
    # train = train.astype(np.float32)
    # test = test.astype(np.float32)
    # train.tofile("data_contriever.bin")
    # test.tofile("data_contriever_test.bin")
    # print('contriever', len(train), len(test), len(train[0]))

    print("Done")
