"""MinHash correctness checks using tiny movie sets and known similarities."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from minhash import compute_signatures


class ComputeSignaturesTests(unittest.TestCase):
    def test_matches_simple_reference_and_keeps_input_unchanged(self) -> None:
        matrix = csr_matrix([
            [0, 0, 0, 0, 0],  # Leading, middle, and trailing empty users.
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=bool)
        original = matrix.copy()
        seed = 42
        length = 20
        expected = np.full((6, length), 5, dtype=np.uint8)
        reference_rng = np.random.default_rng(seed)
        for column in range(length):
            ranks = reference_rng.permutation(5)
            for user, movies in enumerate([[], [1, 3], [], list(range(5)), [2], []]):
                if movies:
                    expected[user, column] = min(int(ranks[movie]) for movie in movies)

        actual = compute_signatures(matrix, length, np.random.default_rng(seed))

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(matrix.indptr, original.indptr)
        np.testing.assert_array_equal(matrix.indices, original.indices)
        np.testing.assert_array_equal(matrix.data, original.data)

    def test_seed_reproducibility_and_signature_prefix(self) -> None:
        matrix = csr_matrix([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
        first = compute_signatures(matrix, 30, np.random.default_rng(7))
        repeated = compute_signatures(matrix, 30, np.random.default_rng(7))
        shorter = compute_signatures(matrix, 10, np.random.default_rng(7))
        different = compute_signatures(matrix, 30, np.random.default_rng(8))
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(first[:, :10], shorter)
        self.assertFalse(np.array_equal(first, different))

    def test_collision_rates_reflect_jaccard_similarity(self) -> None:
        matrix = csr_matrix([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],  # Intersection 2, union 4: Jaccard = 0.5.
            [1, 1, 1, 0, 0, 0],  # Identical to the first user.
            [0, 0, 0, 1, 1, 1],  # Disjoint from the first user.
        ], dtype=bool)
        signatures = compute_signatures(matrix, 4000, np.random.default_rng(123))
        self.assertAlmostEqual(float(np.mean(signatures[0] == signatures[1])), 0.5, delta=0.04)
        np.testing.assert_array_equal(signatures[0], signatures[2])
        self.assertFalse(np.any(signatures[0] == signatures[3]))

    def test_empty_inputs_keep_shape_and_sentinel(self) -> None:
        for shape in [(0, 0), (0, 5), (3, 0), (3, 5)]:
            with self.subTest(shape=shape):
                signatures = compute_signatures(csr_matrix(shape), 8, np.random.default_rng(1))
                self.assertEqual(signatures.shape, (shape[0], 8))
                self.assertTrue(np.all(signatures == shape[1]))

    def test_explicit_zeros_are_ignored_without_mutating_input(self) -> None:
        matrix = csr_matrix(([0, 5], [0, 1], [0, 1, 2]), shape=(2, 3))
        signatures = compute_signatures(matrix, 10, np.random.default_rng(1))
        self.assertTrue(np.all(signatures[0] == 3))
        self.assertTrue(np.all(signatures[1] < 3))
        self.assertEqual(matrix.nnz, 2)
        np.testing.assert_array_equal(matrix.data, [0, 5])

    def test_dtype_can_hold_empty_user_sentinel(self) -> None:
        for movies, expected_dtype in [(255, np.uint8), (256, np.uint16), (65536, np.uint32)]:
            with self.subTest(movies=movies):
                matrix = csr_matrix(([True], [movies - 1], [0, 0, 1]), shape=(2, movies))
                signatures = compute_signatures(matrix, 2, np.random.default_rng(1))
                self.assertEqual(signatures.dtype, np.dtype(expected_dtype))
                self.assertTrue(np.all(signatures[0] == movies))
                self.assertTrue(np.all(signatures[1] < movies))

    def test_rejects_invalid_arguments(self) -> None:
        matrix = csr_matrix([[True]])
        for length in [0, -1, 2.5, True]:
            with self.subTest(length=length), self.assertRaises(ValueError):
                compute_signatures(matrix, length, np.random.default_rng(1))
        with self.assertRaises(TypeError):
            compute_signatures(np.array([[True]]), 10, np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
