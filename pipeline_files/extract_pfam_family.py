#!/usr/bin/env python3
"""
Extract a single Pfam family alignment from the bulk Pfam-A.full file.

First run: decompress .gz and build an index (family -> byte offset).
Subsequent runs: instant extraction via seek().

Usage:
    python extract_pfam_family.py --family PF00076              # extract one family (auto paths from config)
    python extract_pfam_family.py --family PF00076 PF00321      # extract multiple families
    python extract_pfam_family.py --build-index                 # rebuild index only
    python extract_pfam_family.py --input Pfam-A.full.gz --family PF00076 --output my.stockholm
"""

import argparse
import gzip
import json
import os
import sys


def ensure_decompressed(gz_path):
    """Decompress .gz to plain text if not already done. Returns path to plain file."""
    plain_path = gz_path.rstrip(".gz")
    if os.path.exists(plain_path):
        return plain_path
    if not os.path.exists(gz_path):
        print(f"Error: {gz_path} not found", file=sys.stderr)
        sys.exit(1)
    print(f"Decompressing {gz_path} -> {plain_path} (one-time, may take a few minutes)...")
    with gzip.open(gz_path, "rb") as f_in, open(plain_path, "wb") as f_out:
        while True:
            chunk = f_in.read(64 * 1024 * 1024)  # 64MB chunks
            if not chunk:
                break
            f_out.write(chunk)
    print(f"Done. Decompressed size: {os.path.getsize(plain_path) / 1e9:.1f} GB")
    return plain_path


def build_index(plain_path):
    """Build an index mapping family accession -> byte offset. Saves as .index.json."""
    index_path = plain_path + ".index.json"
    print(f"Building index for {plain_path}...")
    index = {}
    with open(plain_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            line_str = line.decode("utf-8", errors="replace")
            if line_str.startswith("# STOCKHOLM"):
                entry_start = pos
            elif line_str.startswith("#=GF AC"):
                acc = line_str.split()[2].strip().split(".")[0].upper()
                index[acc] = entry_start
    with open(index_path, "w") as f:
        json.dump(index, f)
    print(f"Indexed {len(index)} families -> {index_path}")
    return index


def load_index(plain_path):
    """Load index, building it if needed."""
    index_path = plain_path + ".index.json"
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    return build_index(plain_path)


def extract_family_fast(plain_path, family_id, index, output_path=None):
    """Extract a family using byte offset from index."""
    family_id = family_id.upper()
    if family_id not in index:
        print(f"Error: {family_id} not found in index ({len(index)} families available)", file=sys.stderr)
        sys.exit(1)

    offset = index[family_id]
    lines = []
    with open(plain_path, "r") as f:
        f.seek(offset)
        for line in f:
            lines.append(line)
            if line.strip() == "//":
                break

    text = "".join(lines)
    n_seqs = sum(1 for l in lines if l.strip() and not l.startswith("#") and l.strip() != "//")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)
        print(f"Extracted {family_id}: {n_seqs} sequences -> {output_path}")
    else:
        sys.stdout.write(text)

    return n_seqs


def main():
    from config_utils import get_pfam_bulk_file, get_family_msa_path

    parser = argparse.ArgumentParser(description="Extract Pfam family from bulk alignment file")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to Pfam-A.full.gz or Pfam-A.full (default: from config.json)")
    parser.add_argument("--family", type=str, nargs="+",
                        help="Pfam accession(s) (e.g. PF00076 PF00321)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (only for single family; default: from config.json)")
    parser.add_argument("--build-index", action="store_true",
                        help="Decompress and build index only (no extraction)")
    args = parser.parse_args()

    # Resolve input path
    bulk_path = args.input or get_pfam_bulk_file()

    # Ensure decompressed
    plain_path = ensure_decompressed(bulk_path)

    # Load or build index
    index = load_index(plain_path)

    if args.build_index:
        print(f"Index ready: {len(index)} families")
        return

    if not args.family:
        parser.error("Provide --family or --build-index")

    # Extract each family
    for fam in args.family:
        out = args.output if (args.output and len(args.family) == 1) else get_family_msa_path(fam)
        extract_family_fast(plain_path, fam, index, out)


if __name__ == "__main__":
    main()
