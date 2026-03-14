#!/usr/bin/env python3
"""
Extract Pfam family alignments from the bulk Pfam-A.full.gz file.

Works directly on the .gz file — no decompression needed.
First run builds an index (~5-10 min). Subsequent extractions are instant.

Usage:
    python extract_pfam_family.py --family PF00076              # extract one family
    python extract_pfam_family.py --family PF00076 PF00321      # extract multiple
    python extract_pfam_family.py --build-index                 # build index only
    python extract_pfam_family.py --input /path/to/Pfam-A.full.gz --family PF00076
"""

import argparse
import json
import os
import sys

INDEX_FILENAME = "pfam_index.json"


def _get_index_path(gz_path):
    return os.path.join(os.path.dirname(gz_path), INDEX_FILENAME)


def build_index(gz_path):
    """
    Build a byte-offset index for the .gz file.
    Uses indexed_gzip for random access to gzipped files.
    Falls back to sequential gzip scan if indexed_gzip is not installed.
    """
    index_path = _get_index_path(gz_path)

    try:
        import indexed_gzip as igzip
        print(f"Building index for {gz_path} (using indexed_gzip)...")
        index = {}
        with igzip.IndexedGzipFile(gz_path) as f:
            entry_start = 0
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

            # Save the indexed_gzip internal index for fast future opens
            igzip_index_path = gz_path + ".gzidx"
            f.export_index(igzip_index_path)
            print(f"  Saved gzip index: {igzip_index_path}")

    except ImportError:
        import gzip
        print(f"Building index for {gz_path} (sequential scan, install indexed_gzip for faster access)...")
        print(f"  Note: pip install indexed_gzip")
        index = {}
        # For plain gzip we can't seek, so we store line numbers instead
        # and will need to do sequential scan for extraction
        entry_start_line = 0
        line_num = 0
        with gzip.open(gz_path, "rt") as f:
            for line in f:
                if line.startswith("# STOCKHOLM"):
                    entry_start_line = line_num
                elif line.startswith("#=GF AC"):
                    acc = line.split()[2].strip().split(".")[0].upper()
                    index[acc] = entry_start_line
                line_num += 1
                if line_num % 5000000 == 0:
                    print(f"    Scanned {line_num // 1000000}M lines, found {len(index)} families...")

    with open(index_path, "w") as f:
        json.dump(index, f)
    print(f"Indexed {len(index)} families -> {index_path}")
    return index


def load_index(gz_path):
    """Load existing index or build one."""
    index_path = _get_index_path(gz_path)
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)
    return build_index(gz_path)


def extract_family(gz_path, family_id, index, output_path=None):
    """Extract a single family from the gz file using the index."""
    family_id = family_id.upper()
    if family_id not in index:
        print(f"Error: {family_id} not found in index ({len(index)} families available)", file=sys.stderr)
        sys.exit(1)

    offset = index[family_id]

    try:
        import indexed_gzip as igzip
        # Fast path: seek directly in compressed file
        igzip_index_path = gz_path + ".gzidx"
        lines = []
        with igzip.IndexedGzipFile(gz_path) as f:
            if os.path.exists(igzip_index_path):
                f.import_index(igzip_index_path)
            f.seek(offset)
            while True:
                line = f.readline().decode("utf-8", errors="replace")
                if not line:
                    break
                lines.append(line)
                if line.strip() == "//":
                    break

    except ImportError:
        import gzip
        # Slow path: scan to line number
        lines = []
        capturing = False
        line_num = 0
        with gzip.open(gz_path, "rt") as f:
            for line in f:
                if line_num == offset:
                    capturing = True
                if capturing:
                    lines.append(line)
                    if line.strip() == "//":
                        break
                line_num += 1

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
                        help="Path to Pfam-A.full.gz (default: from config.json)")
    parser.add_argument("--family", type=str, nargs="+",
                        help="Pfam accession(s) (e.g. PF00076 PF00321)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (single family only; default: from config)")
    parser.add_argument("--build-index", action="store_true",
                        help="Build index only (no extraction)")
    args = parser.parse_args()

    bulk_path = args.input or get_pfam_bulk_file()

    if not os.path.exists(bulk_path):
        # Check for decompressed version too
        plain = bulk_path.rstrip(".gz")
        if not os.path.exists(plain):
            print(f"Error: {bulk_path} not found", file=sys.stderr)
            sys.exit(1)

    index = load_index(bulk_path)

    if args.build_index:
        print(f"Index ready: {len(index)} families")
        return

    if not args.family:
        parser.error("Provide --family or --build-index")

    for fam in args.family:
        out = args.output if (args.output and len(args.family) == 1) else get_family_msa_path(fam)
        extract_family(bulk_path, fam, index, out)


if __name__ == "__main__":
    main()
