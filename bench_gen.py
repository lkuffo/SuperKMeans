import sklearn.datasets
import numpy as np
import os
import h5py
np.random.seed(42)

RAW_DATA = '../PDX/benchmarks/datasets/downloaded/'

def read_hdf5_data(dataset):
    hdf5_file_name = os.path.join(RAW_DATA, dataset + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32), np.array(hdf5_file["test"], dtype=np.float32)


if __name__ == "__main__":
    # num_dimensions = 1024
    # num_vectors = 262144
    # num_centroids = 4096
    #
    # print(f'Bench gen: \n- D={num_dimensions}\n- num_centroids={num_centroids}\n- dataset=RANDOM')
    # data, _ = sklearn.datasets.make_blobs(n_samples=num_vectors, n_features=num_dimensions, centers=num_centroids, random_state=1)
    # data = data.astype(np.float32)
    #
    # data.tofile("data_random.bin")

    num_dimensions = 1024

    num_vectors = 262144
    # num_vectors = 720896

    train, test = read_hdf5_data("agnews-mxbai-1024-euclidean")
    rng = np.random.default_rng()
    training_sample_idxs = rng.choice(len(train), size=num_vectors, replace=False)
    training_sample_idxs.sort()
    data = train[training_sample_idxs].astype(np.float32)
    data.tofile("data_mxbai.bin")
