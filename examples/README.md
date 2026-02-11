# Super K-Means examples

These examples are a demonstration of how to use our clustering library.

## C++: Simple example

**File:** [`simple_clustering.cpp`](./simple_clustering.cpp)   
**Compile:** `make simple_clustering.out`   
**Run (assuming `pwd` is `./examples`):** `./simple_clustering.out <n> <d> <k>`   
**Parameters**:   
- [Optional] `n`: Number of vectors to cluster 
- [Optional] `d`: Dimensionality of vectors
- [Optional] `k`: Number of clusters to create

## C++: Hierarchical Clustering example

Much faster while preserving quality. Recommended when n > 100K.

**File:** [`hierarchical_clustering.cpp`](./simple_clustering.cpp)   
**Compile:** `make hierarchical_clustering.out`   
**Run (assuming `pwd` is `./examples`):** `./hierarchical_clustering.out <n> <d> <k>`   
**Parameters**:   
- [Optional] `n`: Number of vectors to cluster 
- [Optional] `d`: Dimensionality of vectors
- [Optional] `k`: Number of clusters to create

## Python: Simple example

**File:** [`simple_clustering.py`](./simple_clustering.py)    
**Needs:** `pip install scikit-learn numpy`   
**Run (assuming `pwd` is `./examples`):** `python ./simple_clustering.py <n> <d> <k>`   
**Parameters**:   
- [Optional] `n`: Number of vectors to cluster 
- [Optional] `d`: Dimensionality of vectors
- [Optional] `k`: Number of clusters to create


## Python: Reading .hdf5 example

**File:** [`hdf5_clustering.py`](./hdf5_clustering.py)    
**Needs:** `pip install h5py numpy`   
**Run (assuming `pwd` is `./examples`):** `python ./hdf5_clustering.py <data_path> [<k>]`   
**Parameters**:   
- `data_path`: Path to your `.hdf5` file. We assume the `.hdf5` file has a `train` dataset with the vector embeddings.   
- [Optional] `k`: Number of clusters to create. Default: $4 * \sqrt(n)$.   


