#!/usr/bin/env python3
"""
CLI Script to generate synthetic US residential real estate data.
Usage:
    python generate_data.py [--samples 20000] [--output data/properties.csv]
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_FILE, DATA_DIR
from src.data_generator import generate_real_estate_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic real estate dataset.")
    parser.add_argument("--samples", type=int, default=20000, help="Number of records to generate (default: 20000)")
    parser.add_argument("--output", type=str, default=str(DATA_FILE), help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print(f"Generating {args.samples:,} synthetic property records...")
    df = generate_real_estate_data(n_samples=args.samples, random_seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset successfully saved to: {output_path}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Target 'price' range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f} (Median: ${df['price'].median():,.0f})")
    print(f"Cities included: {', '.join(sorted(df['city'].unique()))}")


if __name__ == "__main__":
    main()

