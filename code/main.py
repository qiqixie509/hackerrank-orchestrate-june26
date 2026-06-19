import argparse

from config import settings
from loaders import load_claims
from pipeline import process_claim
from writer import write_output_csv

# Opus 4.8 pricing per token (USD).
PRICE_INPUT = 5e-6
PRICE_OUTPUT = 25e-6
PRICE_CACHE_WRITE = 6.25e-6  # ~1.25x input
PRICE_CACHE_READ = 0.5e-6    # ~0.1x input


def estimate_cost(totals: dict) -> float:
    return (
        totals["input"] * PRICE_INPUT
        + totals["output"] * PRICE_OUTPUT
        + totals["cache_write"] * PRICE_CACHE_WRITE
        + totals["cache_read"] * PRICE_CACHE_READ
    )


def run(limit=None, input_csv=None, output_csv=None) -> None:
    input_csv = input_csv or settings.claims_csv
    output_csv = output_csv or settings.output_csv

    claims = load_claims(input_csv)
    if limit is not None:
        claims = claims[:limit]

    rows = []
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    for i, claim in enumerate(claims, 1):
        row, usage = process_claim(claim)
        rows.append(row)
        for key in totals:
            totals[key] += usage.get(key, 0)
        print(
            f"[{i}/{len(claims)}] {claim.user_id} {claim.claim_object}"
            f" -> {row['claim_status']}"
        )

    write_output_csv(rows, output_csv)

    print(f"\nWrote {len(rows)} rows to {output_csv}")
    print(f"Token usage: {totals}")
    print(f"Approx cost: ${estimate_cost(totals):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run damage-claim verification over a claims CSV."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.limit,
        help="Process only the first N rows (cost control).",
    )
    parser.add_argument(
        "--input", default=None, help="Input claims CSV (default: claims.csv)."
    )
    parser.add_argument(
        "--output", default=None, help="Output CSV (default: output.csv)."
    )
    args = parser.parse_args()
    run(limit=args.limit, input_csv=args.input, output_csv=args.output)


if __name__ == "__main__":
    main()
