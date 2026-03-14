#!/usr/bin/env python3
"""
Extract a single Pfam family alignment from the bulk Pfam-A.full.gz file.

Usage:
    python extract_pfam_family.py --input Pfam-A.full.gz --family PF00076 --output data/PF00076.stockholm
    python extract_pfam_family.py --input Pfam-A.full.gz --family PF00076  # prints to stdout
    python extract_pfam_family.py --input Pfam-A.full.gz --list            # list all families + seq counts
"""

import argparse
import gzip
import sys


def extract_family(input_path, family_id, output_path=None):
    """Extract a single family from concatenated Stockholm file (plain or gzipped)."""
    family_id = family_id.upper()
    opener = gzip.open if input_path.endswith(".gz") else open

    capturing = False
    lines = []

    with opener(input_path, "rt") as f:
        for line in f:
            if line.startswith("#=GF AC"):
                acc = line.split()[2].strip().split(".")[0].upper()
                if acc == family_id:
                    capturing = True

            if capturing:
                lines.append(line)
                if line.strip() == "//":
                    break

    if not lines:
        print(f"Error: Family {family_id} not found in {input_path}", file=sys.stderr)
        sys.exit(1)

    text = "".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
        n_seqs = sum(1 for l in lines if l.strip() and not l.startswith("#") and l.strip() != "//")
        print(f"Extracted {family_id}: {n_seqs} sequences -> {output_path}")
    else:
        sys.stdout.write(text)


def list_families(input_path):
    """List all families and their sequence counts."""
    opener = gzip.open if input_path.endswith(".gz") else open

    current_acc = None
    n_seqs = 0

    print(f"{'Family':<12} {'Sequences':>10}")
    print("-" * 24)

    with opener(input_path, "rt") as f:
        for line in f:
            if line.startswith("#=GF AC"):
                current_acc = line.split()[2].strip().split(".")[0]
                n_seqs = 0
            elif line.strip() == "//":
                if current_acc:
                    print(f"{current_acc:<12} {n_seqs:>10}")
                current_acc = None
            elif current_acc and line.strip() and not line.startswith("#"):
                n_seqs += 1


def main():
    from config_utils import get_pfam_bulk_file, get_family_msa_path

    parser = argparse.ArgumentParser(description="Extract Pfam family from bulk alignment file")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to Pfam-A.full.gz (default: from config.json)")
    parser.add_argument("--family", type=str, help="Pfam accession (e.g. PF00076)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output Stockholm file path (default: from config.json)")
    parser.add_argument("--list", action="store_true", help="List all families and sequence counts")
    args = parser.parse_args()

    if args.input is None:
        args.input = get_pfam_bulk_file()

    if args.list:
        list_families(args.input)
    elif args.family:
        if args.output is None:
            args.output = get_family_msa_path(args.family)
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        extract_family(args.input, args.family, args.output)
    else:
        parser.error("Provide --family or --list")


if __name__ == "__main__":
    main()
