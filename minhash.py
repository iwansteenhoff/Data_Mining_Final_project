import numpy as np
from scipy.sparse import csr_matrix


def compute_signatures(
    user_movies: csr_matrix,
    signature_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return signatures with shape (number_of_users, signature_length).

    Each permutation assigns every movie a unique rank from 0 to n_movies - 1.
    A user's entry is the smallest rank among their rated movies. Following the
    PDF's tips, this uses actual permutations and vectorized sparse-row minima;
    the only Python loop is over signature columns (typically 80–150).

    All randomness comes from rng. Empty users receive the sentinel n_movies,
    which is larger than any valid rank. The unsigned result dtype is the
    smallest that holds this sentinel (uint16 for the assignment dataset).

    Input must be a CSR matrix; nonzero values indicate rated movies. Explicit
    zeros are removed from a private copy if necessary. The input is unchanged.
    No dense user-by-movie array is created. Runtime is O(h * (nnz + n_movies
    + n_users)), with one temporary rank per stored movie entry.
    """

    # Data preparation produces only True entries; handle stored zeros in other
    # inputs without modifying the caller's sparse matrix.
    if not user_movies.data.all():
        user_movies = user_movies.copy()
        user_movies.eliminate_zeros()

    number_of_users, number_of_movies = user_movies.shape
    rank_dtype = np.min_scalar_type(number_of_movies)
    signatures = np.full(
        (number_of_users, signature_length), number_of_movies, dtype=rank_dtype
    )
    if user_movies.nnz == 0:
        return signatures

    # CSR indptr marks each user's slice in indices. Excluding empty rows avoids
    # invalid reduceat offsets and leaves their signatures at the sentinel.
    nonempty_users = np.flatnonzero(np.diff(user_movies.indptr))
    row_starts = user_movies.indptr[nonempty_users]

    for column in range(signature_length):
        # ranks[movie_index] is that movie's random rank. A permutation used as
        # a rank lookup is itself uniform, so no inverse permutation is needed.
        ranks = rng.permutation(number_of_movies).astype(rank_dtype, copy=False)
        signatures[nonempty_users, column] = np.minimum.reduceat(
            ranks[user_movies.indices], row_starts
        )

    return signatures
