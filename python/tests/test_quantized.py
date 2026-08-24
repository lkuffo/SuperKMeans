from pathlib import Path

import numpy as np
import pytest

from superkmeans import SuperKMeans


QUANTIZERS = ["sq8", "lvq4", "rabitq"]

_TEST_DATA = Path(__file__).resolve().parents[2] / "tests" / "test_data.bin"
_TEST_DATA_D = 1024


def load_test_data(n, d):
    """First n rows and first d dims of the fixed test_data.bin."""
    full = np.fromfile(_TEST_DATA, dtype=np.float32, count=n * _TEST_DATA_D)
    return np.ascontiguousarray(full.reshape(n, _TEST_DATA_D)[:, :d])


class TestQuantizedSuperKMeans:
    """Test suite for the quantized clustering path (sq8 / lvq4 / rabitq)."""

    def test_invalid_quantizer(self):
        with pytest.raises(ValueError, match="quantizer"):
            SuperKMeans(n_clusters=10, dimensionality=128, quantizer="bogus")

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_train_shape(self, quantizer):
        d, k = 128, 300
        data = load_test_data(5000, d)

        km = SuperKMeans(
            n_clusters=k, dimensionality=d, quantizer=quantizer,
            iters=10, sampling_fraction=1.0,
        )
        centroids = km.train(data)

        assert centroids.shape == (k, d)
        assert centroids.dtype == np.float32
        assert km.is_trained_
        assert km.quantizer_ == quantizer

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_assign_family_valid(self, quantizer):
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)

        km = SuperKMeans(
            n_clusters=k, dimensionality=d, quantizer=quantizer,
            iters=10, sampling_fraction=1.0,
        )
        centroids = km.train(data)

        for labels in (
            km.assign(data, centroids),
            km.quantized_assign(data, centroids),
            km.assign_training_points(data, centroids),
        ):
            assert labels.shape == (n,)
            assert labels.dtype == np.uint32
            assert np.all(labels < k)

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_quantized_close_to_exact(self, quantizer):
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)

        km = SuperKMeans(
            n_clusters=k, dimensionality=d, quantizer=quantizer,
            iters=10, sampling_fraction=1.0,
        )
        centroids = km.train(data)

        exact = km.assign(data, centroids)
        quantized = km.quantized_assign(data, centroids)
        reuse = km.assign_training_points(data, centroids)

        quant_vs_exact = np.mean(quantized == exact)
        reuse_vs_quant = np.mean(reuse == quantized)
        assert quant_vs_exact > 0.5, f"{quantizer}: quantized vs exact {quant_vs_exact:.3f}"
        assert reuse_vs_quant > 0.5, f"{quantizer}: reuse vs quantized {reuse_vs_quant:.3f}"

    def test_f32_quantized_assign(self):
        """quantizer='f32' is a full quantizer: quantized_assign matches exact assign."""
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)

        km = SuperKMeans(
            n_clusters=k, dimensionality=d, quantizer="f32",
            iters=10, sampling_fraction=1.0,
        )
        centroids = km.train(data)

        exact = km.assign(data, centroids)
        quantized = km.quantized_assign(data, centroids)
        assert np.mean(quantized == exact) >= 0.99

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_hierarchical_quantized(self, quantizer):
        """Exercise the QuantizedHierarchicalSuperKMeans binding end-to-end."""
        n, d, k = 8000, 128, 64
        data = load_test_data(n, d)

        km = SuperKMeans(
            n_clusters=k, dimensionality=d, quantizer=quantizer,
            hierarchical=True,
            iters_mesoclustering=3, iters_fineclustering=3, iters_refinement=1,
        )
        centroids = km.train(data)

        assert centroids.shape == (k, d)
        assert km.is_trained_
        assert km.hierarchical_ is True

        labels = km.assign(data, centroids)
        q_labels = km.quantized_assign(data, centroids)
        reuse_labels = km.assign_training_points(data, centroids)
        for lab in (labels, q_labels, reuse_labels):
            assert lab.shape == (n,)
            assert lab.dtype == np.uint32
            assert np.all(lab < k)
        assert np.mean(q_labels == labels) > 0.5

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_quantized_assign_before_train(self, quantizer):
        """quantized_assign is standalone (fits a fresh quantizer on the input), so it
        works before train() and closely matches the post-train result."""
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)

        trained = SuperKMeans(
            n_clusters=k, dimensionality=d, quantizer=quantizer,
            iters=10, sampling_fraction=1.0,
        )
        centroids = trained.train(data)
        after = trained.quantized_assign(data, centroids)

        fresh = SuperKMeans(n_clusters=k, dimensionality=d, quantizer=quantizer)
        assert not fresh.is_trained_
        before = fresh.quantized_assign(data, centroids)  # no train() needed

        assert before.shape == (n,)
        assert before.dtype == np.uint32
        assert np.all(before < k)
        assert np.mean(before == after) >= 0.99

    def test_assign_training_points_requires_train(self):
        """assign_training_points reuses trained state, so it must raise before train()."""
        n, d, k = 1000, 128, 50
        data = load_test_data(n, d)
        centroids = np.ascontiguousarray(data[:k], dtype=np.float32)
        km = SuperKMeans(n_clusters=k, dimensionality=d, quantizer="sq8")
        with pytest.raises(RuntimeError):
            km.assign_training_points(data, centroids)

    @pytest.mark.parametrize("quantizer", QUANTIZERS)
    def test_repr_shows_quantizer(self, quantizer):
        km = SuperKMeans(n_clusters=10, dimensionality=128, quantizer=quantizer)
        assert f"quantizer={quantizer!r}" in repr(km)

    @staticmethod
    def _assert_same_clustering(data, run):
        """train() and train(overwrite_input=True) express the same clustering in different
        domains, so compare the clustering itself: the assignments, plus the centroid norms
        (invariant under an orthonormal rotation)."""
        buf, centroids = run(data.copy(), False)
        buf_in_place, centroids_in_place = run(data.copy(), True)

        assert centroids_in_place.shape == centroids.shape
        np.testing.assert_allclose(
            np.sort(np.linalg.norm(centroids_in_place, axis=1)),
            np.sort(np.linalg.norm(centroids, axis=1)),
            rtol=1e-3,
        )
        engine = SuperKMeans(n_clusters=centroids.shape[0], dimensionality=centroids.shape[1])
        agreement = np.mean(
            engine.assign(buf_in_place, centroids_in_place) == engine.assign(buf, centroids)
        )
        assert agreement > 0.95, f"assignment agreement {agreement:.4f}"

    @pytest.mark.parametrize("quantizer", QUANTIZERS + ["f32"])
    def test_overwrite_input_same_clustering_as_train(self, quantizer):
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)

        def run(buf, overwrite_input):
            km = SuperKMeans(
                n_clusters=k, dimensionality=d, quantizer=quantizer,
                iters=10, sampling_fraction=1.0,
            )
            return buf, km.train(buf, overwrite_input=overwrite_input)

        self._assert_same_clustering(data, run)

    @pytest.mark.parametrize("quantizer", QUANTIZERS + ["f32"])
    def test_overwrite_input_hierarchical_same_clustering_as_train(self, quantizer):
        n, d, k = 8000, 128, 64
        data = load_test_data(n, d)

        def run(buf, overwrite_input):
            km = SuperKMeans(
                n_clusters=k, dimensionality=d, quantizer=quantizer, hierarchical=True,
                iters_mesoclustering=3, iters_fineclustering=3, iters_refinement=1,
            )
            return buf, km.train(buf, overwrite_input=overwrite_input)

        self._assert_same_clustering(data, run)

    def test_overwrite_input_rotates_the_caller_buffer(self):
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)
        overwritten = data.copy()

        km = SuperKMeans(
            n_clusters=k, dimensionality=d, iters=5, sampling_fraction=1.0,
        )
        km.train(overwritten, overwrite_input=True)

        assert not np.allclose(overwritten, data)
        np.testing.assert_allclose(
            np.linalg.norm(overwritten, axis=1), np.linalg.norm(data, axis=1), rtol=1e-4
        )

    def test_overwrite_input_leaves_data_and_centroids_in_one_domain(self):
        """unrotate_centroids is forced off, so the rotated buffer and the returned centroids
        are directly comparable and assign() is valid without any extra step."""
        n, d, k = 5000, 128, 300
        data = load_test_data(n, d)
        rotated = data.copy()

        km = SuperKMeans(
            n_clusters=k, dimensionality=d, iters=5, sampling_fraction=1.0,
        )
        centroids = km.train(rotated, overwrite_input=True)

        exact = km.assign(rotated, centroids)
        reuse = km.assign_training_points(rotated, centroids)
        assert np.mean(reuse == exact) > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
