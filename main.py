import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from data_preparation import load_user_movies
from lsh import generate_candidate_pairs
from minhash import compute_signatures
from postprocessing import verify_and_write_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find Netflix user pairs with Jaccard > 0.5.")
    parser.add_argument("--seed", type=int, required=True, help="Nonnegative random seed.")
    parser.add_argument("--input", type=Path, default=Path("user_movie_rating.npy"))
    parser.add_argument("--output", type=Path, default=Path("result.txt"))
    parser.add_argument("--signature-length", type=int, default=120)
    parser.add_argument("--bands", type=int, default=24)
    parser.add_argument("--rows-per-band", type=int, default=5)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if min(args.signature_length, args.bands, args.rows_per_band) <= 0:
        parser.error("signature length, bands, and rows per band must be positive")
    if args.bands * args.rows_per_band > args.signature_length:
        parser.error("bands * rows-per-band must not exceed signature-length")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    return args


def main() -> None:
    args = parse_args()
    start = perf_counter()
    rng = np.random.default_rng(args.seed)
    user_movies = load_user_movies(args.input)
    signatures = compute_signatures(user_movies, args.signature_length, rng)
    candidates = generate_candidate_pairs(signatures, args.bands, args.rows_per_band)
    pair_count = verify_and_write_pairs(user_movies, candidates, args.output)
    print(f"Found {pair_count} pairs in {perf_counter() - start:.1f} seconds.")


if __name__ == "__main__":
    main()
