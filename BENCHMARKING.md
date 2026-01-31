# Benchmarking

We provide a set of benchmarks to reproduce every result of our VLDB'26 publication. To compile them you need to add the following CMake variable:

```bash
cmake . -DSKMEANS_COMPILE_BENCHMARKS=ON
make
```

In the future, we plan to add a proper benchmarking framework for development.

## Prerequisites (CPU)

### OpenMP and a BLAS implementation 
Check [INSTALL.md](./INSTALL.md#installing-openmp)

> [!IMPORTANT]
> A proper BLAS implementation is **EXTREMELY** important for performance. The pre-installed BLAS in your Linux distribution and OpenBLAS installed via `apt` are **SLOW**.

### FAISS 
Our CMake will install FAISS for you. However, you need to set the proper optimization flag. For example, with a machine that supports AVX512 you shall do:
```bash
cmake .  -DFAISS_OPT_LEVEL="avx512" -DSKMEANS_COMPILE_BENCHMARKS=ON
make 
```

Some common flags:
- `avx512_spr`: Intel Granite Rapids, Intel Emerald Rapids, Intel Sapphire Rapids.
- `avx512`: Intel Ice Lake, Intel Sky Lake, AMD Zen 5 (Turin), AMD Zen 4 (Genoa).
- `avx2`: Intel Haswell, AMD Zen 3 (Milan), AMD Zen 2 (Rome).
- `sve`: AWS Graviton 4, AWS Graviton 3.
- `generic`:  AWS Graviton 2, Apple M1-M5.

You can find more information about FAISS installation [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

### Scikit-Learn
You should create a Python `venv` and install `scikit-learn`:
```bash
python3 -m venv ./venv
source ./venv/bin/activate
pip install scikit-learn
```

## Datasets Preparation

We have made our datasets publicly downloadable [here](https://drive.google.com/drive/folders/1f76UCrU52N2wToGMFg9ir1MY8ZocrN34?usp=sharing).

Once you have downloaded a dataset, you have to prepare it for the benchmarks: 
```bash
cd benchmarks
uv run --script setup_data.py --dir <dir> --dataset <dataset_id>
```

- `dir`: The directory in which you have your raw `.hdf5` file.
- `dataset`: The identifier of the dataset you want to use for benchmarking. You can check the identifiers in the collapsable list below:

<details>

<summary><b>List of Datasets</b></summary>

| Identifier   | Embeddings    | Model        | # Vectors  | Dim. | Size (GB) ↑ |
|--------------|----------------|--------------|------------|-------|--------------|
| `cohere`     | Text           | EmbedV3-EN   | 10,000,000 | 1024  | 40.96        |
| `arxiv`      | Text           | InstructorXL | 2,253,000  | 768   | 6.92         |
| `openai`     | Text           | OpenAI       | 999,000    | 1536  | 6.14         |
| `wiki`       | Text           | OpenAI       | 260,372    | 3072  | 3.20         |
| `mxbai`      | Text           | MXBAI        | 769,382    | 1024  | 3.15         |
| `contriever` | Text           | Contriever   | 999,000    | 768   | 3.07         |
| `clip`       | Image          | CLIP         | 1,281,167  | 512   | 2.62         |
| `yahoo`      | Text           | MiniLM       | 677,305    | 384   | 1.04         |
| `glove`      | Word           | GloVe        | 1,183,514  | 200   | 0.95         |
| `yandex`     | Text-to-Image  | SE-ResNeXt   | 1,000,000  | 200   | 0.80         |

</details>

### Installing `uv`
We use `uv` to simplify the installing of dependencies. You can install it by doing:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Compiling and Running Benchmarks

Compile the benchmarks:
```bash
cmake . -DSKMEANS_COMPILE_BENCHMARKS=ON -DFAISS_OPT_LEVEL="<opt_level>"
make 
cd benchmarks
```

We have a handful of `.sh` scripts under `./benchmarks` to reproduce the experiments of our publication. Below you will find each one of them and how to run it. 

### Benchmarks List

#### End-to-End
Runs clustering with Super K-Means, FAISS K-Means and Scikit-Learn K-Means with the number of clusters fixed to $k= 4*\sqrt{N}$, and the number of iterations fixed to 25. For instance, for 1M vectors, $k=4000$. 

**Output**: CSV file in `./benchmarks/results/default/end_to_end.csv` reporting construction time in milliseconds of each algorithm, and the `recall` that the centroids would yield when used for vector search tasks.

**Command (assuming `pwd` is `./benchmarks`)**:
```sh
./end_to_end.sh -b .. -p <python_path> <dataset1_id> <dataset2_id> <datasetn_id>
```

- `python_path`: Path to your Python binaries
- `dataset_id`: Identifier of the dataset you want to benchmark. You can run the benchmarks for many datasets at a time by separating the dataset IDs by a space.


#### Varying $k$
Runs clustering with Super K-Means, FAISS K-Means and Scikit-Learn K-Means using different number of clusters per dataset (100, 1K, 10K, and 100K), and the number of iterations fixed to 25.

**Output**: CSV file in `./benchmarks/results/default/varying_k.csv` reporting construction time in milliseconds of each algorithm.

**Command (assuming `pwd` is `./benchmarks`)**:
```sh
./varying_k.sh -b .. -p <python_path> <dataset_id>
```

#### Early Termination
Runs clustering with Super K-Means, FAISS K-Means and Scikit-Learn K-Means with the number of clusters fixed to $k= 4*\sqrt{N}$. In FAISS, we stop at 10 iterations. In Scikit-Learn, we stop at 300 iterations but activating the default `tolerance` parameter for early stopping. Finally, for Super K-Means, we use **Early Termination by Recall**, trying different tolerancy levels.

**Output**: CSV file in `./benchmarks/results/default/early_termination.csv` reporting construction time in milliseconds of each algorithm and the `recall` that the centroids would yield when used for vector search tasks.

**Command (assuming `pwd` is `./benchmarks`)**:
```sh
./early_termination.sh -b .. -p <python_path> <dataset_id>
```

#### Sampling
Runs clustering with Super K-Means using different percentages of sampling (from 1% to 100%), with the number of clusters fixed to $k= 4*\sqrt{N}$, and the number of iterations fixed to 25. 

**Output**: CSV file in `./benchmarks/results/default/sampling.csv` reporting construction time in milliseconds of each algorithm and the `recall` that the centroids would yield when used for vector search tasks.

**Command (assuming `pwd` is `./benchmarks`)**:
```sh
./sampling.sh -b .. <dataset_id>
```


#### Quality per Iteration
Runs clustering with Super K-Means using different iterations (from 1 to 10), with the number of clusters fixed to $k= 4*\sqrt{N}$.

**Output**: CSV file in `./benchmarks/results/default/pareto.csv` reporting construction time in milliseconds of each algorithm and the `recall` that the centroids would yield when used for vector search tasks.

**Command (assuming `pwd` is `./benchmarks`)**:
```sh
./pareto.sh -b .. <dataset_id>
```

#### Profiling

**Output**: Console logs with the runtime and profiling. Non-persistent. 

**Command (assuming `pwd` is `./benchmarks`)**

```sh
make ad_hoc_superkmeans.cpp
./ad_hoc_superkmeans <dataset_id>
```
