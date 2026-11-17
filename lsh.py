"""Part 3: use LSH banding to generate promising user pairs.

For similarity s, the idealized probability of becoming a candidate is
1 - (1 - s**rows_per_band)**bands. Use this when choosing parameters, then
measure actual runtime and results across several seeds.
"""

from collections.abc import Iterator

import numpy as np


def generate_candidate_pairs(
    signatures: np.ndarray,
    bands: int,
    rows_per_band: int,
) -> Iterator[tuple[int, int]]:
    """Yield ordered candidate pairs using 1-based user IDs.

    TODO:
    1. Require positive b and r with b * r <= signature length.
    2. Use r signature entries per band as a bucket key.
    3. Group users with identical keys within each band.
    4. Yield pairs from buckets containing at least two users.
    5. Decide how to postpone or skip very large buckets and document the policy.

    Ignore unused columns when b * r is smaller than the signature length.
    Yield pairs lazily to avoid storing every candidate at once. Repeated pairs
    across bands are allowed here; part 4 must remove them. For matrix rows i < j,
    yield (i + 1, j + 1).
    """
    raise NotImplementedError("Part 3: implement banding and candidate generation.")
