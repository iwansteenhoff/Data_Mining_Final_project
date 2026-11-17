from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


SIMILARITY_THRESHOLD = 0.5


def jaccard_similarity(movie_ids_a: np.ndarray, movie_ids_b: np.ndarray) -> float:
    """Compute exact Jaccard similarity for two sorted, unique movie-index arrays.

    TODO: count the intersection and divide by len(a) + len(b) - intersection.
    Choose and document a convention for two empty sets if supporting them.
    The assignment's users all have at least 300 rated movies.
    """
    raise NotImplementedError("Part 4: implement exact Jaccard similarity.")


def verify_and_write_pairs(
    user_movies: csr_matrix,
    candidate_pairs: Iterable[tuple[int, int]],
    output_path: Path,
) -> int:
    """Write unique pairs with similarity strictly > 0.5; return their count.

    TODO:
    1. Initialize the output file for this run, including when no pair qualifies.
    2. Normalize each pair to u1 < u2, exclude self-pairs, and deduplicate.
    3. Convert user IDs to row indices by subtracting 1.
    4. Optionally reject pairs whose set-size ratio cannot exceed 0.5.
    5. Extract movie indices from CSR indices/indptr and compute exact Jaccard.
    6. Append qualifying pairs as 'u1,u2\\n', without scores or a header.
    7. Preserve progress by closing the output after each appended pair, as
       suggested by the PDF, and track memory used by candidate deduplication.

    Similarity equal to 0.5 must be rejected. Never accept a pair based only on
    its approximate MinHash similarity.
    """
    raise NotImplementedError("Part 4: implement candidate verification and output.")
