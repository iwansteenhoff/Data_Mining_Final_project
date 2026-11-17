import argparse
from pathlib import Path


def run_experiments(input_path: Path, seeds: list[int]) -> None:
    """Compare configurations and record measurements for each seed.

    TODO:
    1. Choose a modest grid of h, b, and r, with b * r <= h.
    2. Run the complete pipeline for each configuration and seed.
    3. Record stage timings, total runtime, peak memory, and unique pair counts.
    4. Use separate output files for each configuration/seed combination.
    5. Compare repeatability and pair overlap across seeds.
    6. Save a results table for the report, including hardware details.

    A tiny synthetic dataset can be checked against all-pairs exact similarity
    to validate correctness. Do not use all-pairs comparison on the full data.
    Do not claim full-dataset recall unless an exact reference is available.
    """
    raise NotImplementedError("Part 5: implement experiments and measurement.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MinHash/LSH parameters across seeds.")
    parser.add_argument("--input", type=Path, default=Path("user_movie_rating.npy"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    run_experiments(args.input, args.seeds)


if __name__ == "__main__":
    main()
