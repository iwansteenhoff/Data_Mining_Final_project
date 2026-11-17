from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def load_user_movies(input_path: Path) -> csr_matrix:
    """Return a binary CSR matrix with users as rows and movies as columns.

    Input must be an integer .npy array with three columns: user ID, movie ID,
    and rating. IDs must be positive; ratings are ignored, including zero ratings.
    Row i represents user i + 1 and column j represents movie j + 1. Missing IDs
    leave empty rows or columns; an empty input returns a (0, 0) matrix.

    The input is memory-mapped read-only. Validation uses bounded chunks, and
    index copies use 32-bit integers when possible. Boolean sparse values keep
    duplicate records as presence without integer overflow. The returned CSR
    matrix has sorted, unique movie indices per user and no stored zeros.

    Raises ValueError for an invalid array shape, dtype, or nonpositive IDs.
    File errors from NumPy are propagated to the caller.
    """
    records = np.load(input_path, mmap_mode="r", allow_pickle=False)

    number_of_records = records.shape[0]

    number_of_users = 0
    number_of_movies = 0
    chunk_size = 1_000_000
    for start in range(0, number_of_records, chunk_size):
        ids = records[start : start + chunk_size, :2]
        largest_ids = ids.max(axis=0)
        number_of_users = max(number_of_users, int(largest_ids[0]))
        number_of_movies = max(number_of_movies, int(largest_ids[1]))

    largest_index = max(number_of_records, number_of_users, number_of_movies)
    index_dtype = np.int32 if largest_index <= np.iinfo(np.int32).max else np.int64

    # Own these copies so converting to zero-based indices never changes the file.
    user_indices = records[:, 0].astype(index_dtype, copy=True)
    movie_indices = records[:, 1].astype(index_dtype, copy=True)
    user_indices -= 1
    movie_indices -= 1

    # COO accepts unsorted records; conversion to CSR merges duplicates and sorts.
    user_movies = coo_matrix(
        (np.ones(number_of_records, dtype=np.bool_), (user_indices, movie_indices)),
        shape=(number_of_users, number_of_movies),
        dtype=np.bool_,
    ).tocsr()
    user_movies.data.fill(True)
    return user_movies
