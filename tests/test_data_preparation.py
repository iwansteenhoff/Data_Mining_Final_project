"""Small correctness checks; run with python -m unittest discover -s tests."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from data_preparation import load_user_movies


class LoadUserMoviesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input_path = Path(self.directory.name) / "ratings.npy"

    def test_unsorted_records_ignore_ratings_and_preserve_id_mapping(self) -> None:
        records = np.array([[3, 4, 0], [1, 3, 5], [3, 1, 2], [1, 1, 1]])
        np.save(self.input_path, records)

        matrix = load_user_movies(self.input_path)

        self.assertIsInstance(matrix, csr_matrix)
        self.assertEqual(matrix.dtype, np.dtype(bool))
        self.assertTrue(matrix.has_canonical_format)
        np.testing.assert_array_equal(
            matrix.toarray(),
            [[True, False, True, False],
             [False, False, False, False],
             [True, False, False, True]],
        )
        np.testing.assert_array_equal(np.load(self.input_path), records)

    def test_duplicate_records_do_not_overflow_or_create_stored_zeros(self) -> None:
        np.save(self.input_path, np.tile([1, 2, 5], (256, 1)))

        matrix = load_user_movies(self.input_path)

        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(matrix.nnz, 1)
        self.assertTrue(matrix[0, 1])
        self.assertTrue(matrix.data.all())

    def test_empty_records(self) -> None:
        np.save(self.input_path, np.empty((0, 3), dtype=np.int64))
        matrix = load_user_movies(self.input_path)
        self.assertEqual(matrix.shape, (0, 0))
        self.assertEqual(matrix.nnz, 0)

    def test_rejects_invalid_arrays(self) -> None:
        invalid_arrays = [
            np.array([1, 2, 3]),
            np.ones((2, 2), dtype=int),
            np.array([[1.0, 2.0, 3.0]]),
            np.array([[0, 1, 5]]),
            np.array([[1, -1, 5]]),
            np.array([[0, 1, 5]], dtype=np.uint64),
        ]
        for records in invalid_arrays:
            with self.subTest(records=records):
                np.save(self.input_path, records)
                with self.assertRaises(ValueError):
                    load_user_movies(self.input_path)

    def test_unsigned_ids(self) -> None:
        np.save(self.input_path, np.array([[1, 2, 5]], dtype=np.uint64))
        matrix = load_user_movies(self.input_path)
        self.assertEqual(matrix.shape, (1, 2))
        self.assertTrue(matrix[0, 1])

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_user_movies(self.input_path)


if __name__ == "__main__":
    unittest.main()
