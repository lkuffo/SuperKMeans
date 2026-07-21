# CLAUDE.md

Working notes for the SuperKMeans repo.

## What it is

Fast k-means for high-dim vector embeddings, on `float32` **or quantized** vectors (`sq8`,
`lvq4`, `rabitq`) — you pass `float32`, it quantizes internally. Faster than FAISS /
scikit-learn at equal quality (~1M×1536 in seconds). Header-only **C++17** + **Python
bindings**; CPUs (ARM + x86) and GPUs. Core: `SuperKMeans::Train()` in
[`include/superkmeans/superkmeans.h`](include/superkmeans/superkmeans.h).
Paper: https://arxiv.org/pdf/2603.20009.

## Core idea: progressive pruning + PDX

Prunes centroids during the **assignment** step by interleaving a GEMM on the front `d'` dims
with **progressive**-pruning kernels every 64 dims in the trailing ones. Unlike Elkan's (a 1/0
decision), pruning is progressive — it keeps trying to prune while walking the trailing dims.
Vectors use a **hybrid PDX layout** (block-column-major, split every 64 dims) to make this
efficient. Papers: **ADSampling** https://dl.acm.org/doi/pdf/10.1145/3589282 (almost-lossless,
< 0.005 recall loss); **PDX** https://dl.acm.org/doi/pdf/10.1145/3725333.

## Where things live

| Path | What |
| --- | --- |
| `include/superkmeans/superkmeans.h` | Core — `Train()`, assign family, pruning loop |
| `include/superkmeans/hierarchical_superkmeans.h` | Hierarchical variant (use for n > 100K) |
| `include/superkmeans/common.h` | Shared `constexpr` constants + macros/pragmas |
| `include/superkmeans/profiler.h` | `SKM_PROFILE_SCOPE` timing |
| `include/superkmeans/distance_computers/` | **All** SIMD distance kernels |
| `include/superkmeans/quantizers/` | `f32`/`sq8`/`lvq4`/`rabitq` + `quantizer.h`, `sq_common.h` |
| `include/superkmeans/pdx/` | PDX layout + `utils.h` |
| `benchmarks/` | Benchmarks; base example `ad_hoc_superkmeans.cpp` |
| `examples/` | `simple_clustering.{cpp,py}`, `hdf5_clustering.py` |
| `python/` | Bindings (`bindings/bindings.cpp`, `superkmeans/__init__.py`); `python/README.md` |
| `tests/` | GoogleTest C++; `python/tests/` for bindings |

Docs: `README.md`, `INSTALL.md`, `BENCHMARKING.md`, `CONTRIBUTING.md`, `python/README.md`.

## Verification gate (definition of done)

**A change isn't done until all pass. Run in the FOREGROUND — never background these.**

1. **Format** — `./scripts/format.sh`, then `./scripts/format_check.sh` clean.
2. **Build** — `cmake . -DSKMEANS_COMPILE_TESTS=ON && make -j$(nproc) tests`, no errors.
3. **C++ tests** — `ctest --output-on-failure` all pass (a few parametrized cases skip by design).
4. **Lint** — `./scripts/tidy_check.sh`: no `.clang-tidy` warnings from `include/superkmeans/`.
5. **Python** — `venv/bin/pip install .` (builds the bindings), then `venv/bin/pytest python/tests/`.
6. **CLAUDE.md accurate** — if the change invalidated anything here (paths, commands, contracts,
   thresholds, gotchas, SIMD org, style), update this file in the same change.

New feature ⇒ ship a unit test with it (C++ in `tests/`, Python in `python/tests/` if exposed).

## Build & run (beyond the gate)

Header-only; consumers link the `superkmeans` target.
```bash
cmake . && make examples          # examples on by default; ./examples/simple_clustering.out <n> <d> <k>
cmake . -DSKMEANS_COMPILE_BENCHMARKS=ON -DFAISS_OPT_LEVEL="avx512" && make benchmarks
./benchmarks/ad_hoc_superkmeans.out <dataset_id>   # base example + profiling logs
```
Knobs: `-DSKMEANS_MARCH` (default `native`), `-DBLAS_LIBRARIES` (a good BLAS is critical —
distro/apt OpenBLAS is slow, build from source). See INSTALL.md.

## Code style

- **Naming**: `PascalCase` functions/classes/structs; `snake_case` variables/members;
  `UPPER_SNAKE_CASE` constants; namespaces lowercase.
- Follow `.clang-format` / `.clang-tidy`.
- **Memory**: RAII, no raw `new`/`delete`. Buffers that don't need zero-init → `new T[]` in a
  `unique_ptr`, **not** `std::vector`/`resize()`.
- Constants/magic numbers → `include/superkmeans/common.h` as `constexpr`.
- Keep comments simple. TODOs: `TODO(@<github_user>, <priority>): <summary>`.
- **Reuse before writing**: check `common.h`, `quantizers/sq_common.h`, `pdx/utils.h` first.

## Performance

Performance-critical — weigh every copy/allocation.
- **SIMD is centralized.** Distance kernels exist for **NEON / AVX2 / AVX512 / scalar**,
  dispatched at compile time. **All** SIMD lives in `distance_computers/` — don't scatter it
  elsewhere; keep all backends in sync when changing a kernel.
- **`SKM_VECTORIZE_LOOP`** (`common.h`) forces loop autovectorization (esp. FP reductions).
  Other macros there: `SKM_RESTRICT`, `SKM_ALWAYS_INLINE`, `SKM_NO_INLINE`,
  `SKM_LIKELY`/`SKM_UNLIKELY`, `SKM_PREFETCH`.
- **Profiling**: `SKM_PROFILE_SCOPE("name")` (`profiler.h`); `ad_hoc_superkmeans.out` prints logs.

## Recipes

- **Add/change a distance kernel** — edit **all** backends in `distance_computers/`
  (`neon_computers.h`, `avx2_computers.h`, `avx512_computers.h`, `scalar_computers.h`) in
  lockstep. On ARM the x86 ones aren't compiled or linted, so eyeball them / rely on x86 CI.
  Cover it in `tests/test_distance_computers.cpp`.
- **Add a quantizer** — implement the `IQuantizer<q>` pure-virtual interface
  (`quantizers/quantizer.h`: `Fit`/`Encode`/`Decode`/`ComputeNorms`/`FindNearestNeighbor`/…), add
  a value to `QuantizerType` (`common.h`), register it in `CreateQuantizer()` (`superkmeans.h`),
  expose it in the bindings (`bindings.cpp` enum + `__init__.py` `_QUANTIZERS`/`_QUANTIZER_MAP`),
  and add tests (see Testing).
- **Add a benchmark** — register the target with `skmeans_add_benchmark(<name>.out <source>)` in
  `benchmarks/CMakeLists.txt` (`skmeans_add_faiss_benchmark` for FAISS-linked ones).

## Testing

- C++ in `tests/` (GoogleTest), Python in `python/tests/`. Compile with
  `-DSKMEANS_COMPILE_TESTS=ON`, run via `ctest`.
- Quantized tests are **parametrized over quantizers** with `TEST_P` + `INSTANTIATE_TEST_SUITE_P`:
  shared u8 integration/pruning tests live **once** in `tests/test_quantized.cpp` (auto-covers
  sq8/lvq4/rabitq via `GetParam()`); quantizer-*specific* unit tests go in
  `test_quantized_{sq8,lvq4,rabitq}.cpp`. Hierarchical quantized tests are parametrized the same
  way in `test_hierarchical_superkmeans.cpp`.
- **Unless the user says otherwise, a new test must cover all quantizers *and* hierarchical with
  all quantizers** — add it as a parametrized `TEST_P` (flat) plus its hierarchical counterpart,
  not a single-quantizer case.
- Recall-checking tests need `#undef HAS_FFTW` at the top (see Gotchas).

## Gotchas

### Recall ground truth & `#undef HAS_FFTW`

SuperKMeans builds **IVF indexes for search**, so **`recall`** is the metric we defend (over
WCSS). Tests check `RECALL_GROUND_TRUTH` (`tests/recall_utils.h`) per pipeline (`f32`/`sq8`/
`lvq4`/`rabitq`, flat + hierarchical) via `EXPECT_NEAR(recall, expected, RECALL_TOL)`.
- **Generated by** `generate_recall_ground_truth.cpp` (`.out`): trains on the fixed
  `tests/test_data.bin` with the fixed config in `recall_utils.h`, prints values to paste back.
  `generate_wcss_ground_truth.cpp` is the WCSS twin.
- **`#undef HAS_FFTW`**: the ground truth used the **non-FFTW rotation path**. With FFTW
  installed CMake defines `HAS_FFTW` → FFT-based rotation → slightly different rotation → recall
  drifts past `RECALL_TOL`. So **every ground-truth-checking file starts with `#undef HAS_FFTW`**
  (`test_quantized*.cpp`, `test_hierarchical_superkmeans.cpp`, `test_wcss.cpp`, both generators)
  — keep that line.
- **Regenerating**: almost never — it can mask real regressions. **Only with explicit user
  confirmation**, when the ground truth is *meant* to change; run the generator, paste its block
  into `recall_utils.h`, note why.

### Training / assign paths (rotation, assign family, stale caches)

**Trained state is rotated.** `Train()` rotates all data up front (`SampleAndRotateVectors` →
ADSampling `pruner->Rotate`, seeded at construction), then `Fit()`/`Encode()` run on it — so the
quantizer params and trained state (`quantized_data`, `quantized_centroids`,
`horizontal_centroids`) all live in the rotated domain. Rotation is required by **(a)** RaBitQ,
and **(b) ADSampling pruning for every quantizer** (pruning bounds derive from the random
rotation → valid only on rotated data). sq8/lvq4 may skip rotation **only on a non-pruning
path**. Encoding in the wrong domain, or pruning on un-rotated data ⇒ **silently bad params /
invalid bounds** (no crash, just recall loss).

**Assign family (3 methods):**
- **`Assign`** — exact f32 brute force. Standalone, no trained state. Ground-truth reference.
- **`QuantizedAssign`** — standalone quantized analog: fits a *fresh* quantizer on the input, no
  `Train()` needed, never touches trained state, rotates **only for RaBitQ** (safe: non-pruning
  GEMM path, so sq8/lvq4 re-fit on un-rotated input). For arbitrary/new vectors.
- **`AssignTrainingPoints`** — reuses trained state; **`vectors` must equal the `Train()` data**
  (`n_vectors == n_train`). Three paths: (1) **pruning reuse** if `sampling_fraction==1.0 &&
  SupportsPruning && !use_blas_only && d ≥ 128 (DIMENSION_THRESHOLD_FOR_PRUNING) && n_clusters >
  256 (N_CLUSTERS_THRESHOLD_FOR_PRUNING) && iters > 1`; (2) **GEMM-only reuse** if `sampling==1.0`
  but the gate fails; (3) **fallback to `QuantizedAssign`** if `sampling < 1.0`. It and
  `QuantizedAssign` need **not** agree per-point (different domains for sq8/lvq4) — validate
  **recall**, not per-point agreement.

**Stale caches (classic footgun).** The quantized path caches centroid norms, data norms, and
**partial-norms** for pruning (keyed on `partial_d`). Reading a cache an earlier step didn't
refresh for the current `partial_d`/size ⇒ stale read: wrong distances, or **SIGSEGV** (cache
sized for a different `partial_d`). Defense = **self-heal**: FindNearestNeighbor rebuilds when
inconsistent (`if (cached_partial_d_ != partial_d || cache.size() != n_x) recompute(...)`, in
sq8/lvq4/rabitq). Keep that guard; use `InvalidateCaches()` / `Ensure…()` at entry points.

**Hierarchical + `iters_refinement == 0`.** Phases: mesoclustering → fineclustering →
refinement; **`iters_refinement` defaults to 0**. `partial_d` is shrunk to the refinement value
(~`vertical_d/3`) **unconditionally before** the loop, but `CacheDataPartialNorms` for it is
**inside** the loop — so at `iters_refinement == 0` the loop never runs and the trained state has
a small `partial_d` with a partial-norms cache keyed to the fineclustering `partial_d`. A later
`AssignTrainingPoints` (pruning reuse) reads that mismatch — the SIGSEGV above, survived only by
the self-heal. Preserve it if you touch these caches.

## Python bindings

One `SuperKMeans` class hides flat/hierarchical and f32/quantized selection. Full API:
`python/README.md`; source: `python/bindings/bindings.cpp` + `python/superkmeans/__init__.py`.
```python
kmeans = SuperKMeans(n_clusters=k, dimensionality=d, quantizer="rabitq")  # or f32/sq8/lvq4
centroids   = kmeans.train(data)               # float32 centroids (k×d)
assignments = kmeans.assign(data, centroids)   # exact; quantized_assign(...) needs train() first
```
