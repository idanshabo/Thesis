#!/usr/bin/env python3
"""
Disk cleanup utility for cluster storage.

Scans a directory tree and reports:
  1. Largest files
  2. Duplicate files (by hash)
  3. Compressible files (large text/csv/tsv/log files not yet gzipped)
  4. Common junk (*.pyc, __pycache__, .ipynb_checkpoints, core dumps, etc.)

Modes:
  --scan    (default) Report only, don't change anything
  --clean   Actually delete duplicates, compress files, remove junk

Usage:
    python cleanup_disk.py --path /sci/labs/orzuk/orzuk --scan
    python cleanup_disk.py --path /sci/labs/orzuk/orzuk --clean --min-size 100  # 100MB minimum
    python cleanup_disk.py --path /sci/labs/orzuk/orzuk --scan --top 50
"""

import argparse
import hashlib
import gzip
import os
import shutil
import sys
from collections import defaultdict


# Extensions that compress well
COMPRESSIBLE_EXT = {".txt", ".csv", ".tsv", ".log", ".fasta", ".fa", ".faa",
                    ".stockholm", ".sto", ".stk", ".sam", ".vcf", ".bed",
                    ".gff", ".gtf", ".pdb", ".xml", ".json", ".aln", ".out",
                    ".alignment", ".nwk", ".tree", ".nex", ".nexus"}

# Junk patterns to remove
JUNK_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache"}
JUNK_EXT = {".pyc", ".pyo"}
JUNK_PREFIXES = ("core.",)  # core dumps


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


def file_hash(path, chunk_size=8192):
    """Fast hash: read first and last 64KB + file size for speed."""
    size = os.path.getsize(path)
    h = hashlib.md5()
    h.update(str(size).encode())
    try:
        with open(path, "rb") as f:
            h.update(f.read(65536))
            if size > 131072:
                f.seek(-65536, 2)
                h.update(f.read(65536))
    except (OSError, IOError):
        return None
    return h.hexdigest()


def full_hash(path):
    """Full MD5 for confirming duplicates."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1048576), b""):
                h.update(chunk)
    except (OSError, IOError):
        return None
    return h.hexdigest()


def scan_directory(root, min_size_mb=10, top_n=30, verbose=True):
    """Scan directory and return cleanup report."""
    all_files = []
    junk_files = []
    junk_dirs_found = []
    compressible = []
    hash_groups = defaultdict(list)

    min_size = min_size_mb * 1024 * 1024

    if verbose:
        print(f"Scanning {root} (min file size: {min_size_mb} MB)...")

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Detect junk directories
        for d in list(dirnames):
            if d in JUNK_DIRS:
                full = os.path.join(dirpath, d)
                try:
                    dir_size = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fns in os.walk(full)
                        for f in fns
                    )
                except OSError:
                    dir_size = 0
                junk_dirs_found.append((full, dir_size))

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)

            # Skip symlinks
            if os.path.islink(fpath):
                continue

            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue

            # Junk files
            ext = os.path.splitext(fname)[1].lower()
            if ext in JUNK_EXT or any(fname.startswith(p) for p in JUNK_PREFIXES):
                junk_files.append((fpath, size))
                continue

            if size < min_size:
                continue

            all_files.append((fpath, size))

            # Check if compressible
            if ext in COMPRESSIBLE_EXT and not fpath.endswith(".gz"):
                compressible.append((fpath, size))

            # Hash for duplicate detection
            h = file_hash(fpath)
            if h:
                hash_groups[h].append((fpath, size))

    # Find confirmed duplicates (full hash check)
    duplicates = []
    for quick_hash, group in hash_groups.items():
        if len(group) < 2:
            continue
        # Confirm with full hash
        full_groups = defaultdict(list)
        for path, size in group:
            fh = full_hash(path)
            if fh:
                full_groups[fh].append((path, size))
        for fh, confirmed in full_groups.items():
            if len(confirmed) >= 2:
                duplicates.append(confirmed)

    # Sort results
    all_files.sort(key=lambda x: -x[1])
    compressible.sort(key=lambda x: -x[1])
    duplicates.sort(key=lambda x: -x[0][1])

    return {
        "top_files": all_files[:top_n],
        "duplicates": duplicates,
        "compressible": compressible,
        "junk_files": junk_files,
        "junk_dirs": junk_dirs_found,
    }


def print_report(report):
    """Print a human-readable cleanup report."""
    print(f"\n{'=' * 70}")
    print(f"  DISK CLEANUP REPORT")
    print(f"{'=' * 70}")

    # Top largest files
    print(f"\n--- LARGEST FILES ---")
    total_top = 0
    for path, size in report["top_files"]:
        print(f"  {human_size(size):>10}  {path}")
        total_top += size
    print(f"  {'─' * 60}")
    print(f"  {human_size(total_top):>10}  TOTAL (top {len(report['top_files'])} files)")

    # Duplicates
    dup_savings = 0
    if report["duplicates"]:
        print(f"\n--- DUPLICATE FILES ({len(report['duplicates'])} groups) ---")
        for group in report["duplicates"][:20]:
            size = group[0][1]
            dup_savings += size * (len(group) - 1)
            print(f"\n  {human_size(size)} x {len(group)} copies:")
            for path, _ in group:
                print(f"    {path}")
        if len(report["duplicates"]) > 20:
            print(f"\n  ... and {len(report['duplicates']) - 20} more groups")
        print(f"\n  Potential savings from duplicates: {human_size(dup_savings)}")
    else:
        print(f"\n--- NO DUPLICATES FOUND ---")

    # Compressible
    comp_savings = 0
    if report["compressible"]:
        print(f"\n--- COMPRESSIBLE FILES ({len(report['compressible'])}) ---")
        for path, size in report["compressible"][:20]:
            est_compressed = size * 0.15  # text files typically compress to ~15%
            comp_savings += size - est_compressed
            print(f"  {human_size(size):>10}  {path}")
        if len(report["compressible"]) > 20:
            print(f"  ... and {len(report['compressible']) - 20} more files")
        print(f"\n  Estimated savings from compression: {human_size(comp_savings)} (assuming ~85% ratio)")
    else:
        print(f"\n--- NO COMPRESSIBLE FILES FOUND ---")

    # Junk
    junk_total = sum(s for _, s in report["junk_files"]) + sum(s for _, s in report["junk_dirs"])
    if junk_total > 0:
        print(f"\n--- JUNK ({len(report['junk_files'])} files, {len(report['junk_dirs'])} dirs) ---")
        print(f"  Total junk: {human_size(junk_total)}")
        for path, size in report["junk_dirs"][:10]:
            print(f"  {human_size(size):>10}  {path}/")

    # Summary
    total_savings = dup_savings + comp_savings + junk_total
    print(f"\n{'=' * 70}")
    print(f"  ESTIMATED TOTAL SAVINGS: {human_size(total_savings)}")
    print(f"    Duplicates:    {human_size(dup_savings)}")
    print(f"    Compression:   {human_size(comp_savings)}")
    print(f"    Junk:          {human_size(junk_total)}")
    print(f"{'=' * 70}")


def do_clean(report, dry_run=False):
    """Execute cleanup actions. Returns bytes freed."""
    freed = 0
    action = "Would remove" if dry_run else "Removing"

    # 1. Remove junk dirs
    for path, size in report["junk_dirs"]:
        print(f"  {action} junk dir: {path} ({human_size(size)})")
        if not dry_run:
            shutil.rmtree(path, ignore_errors=True)
            freed += size

    # 2. Remove junk files
    for path, size in report["junk_files"]:
        if not dry_run:
            try:
                os.remove(path)
                freed += size
            except OSError:
                pass

    # 3. Remove duplicate copies (keep the first/oldest)
    for group in report["duplicates"]:
        # Keep the first one, remove the rest
        keep = group[0]
        for path, size in group[1:]:
            print(f"  {action} duplicate: {path} ({human_size(size)})")
            print(f"    Keeping: {keep[0]}")
            if not dry_run:
                try:
                    os.remove(path)
                    freed += size
                except OSError:
                    pass

    # 4. Compress large text files
    for path, size in report["compressible"]:
        gz_path = path + ".gz"
        if os.path.exists(gz_path):
            continue
        print(f"  {'Would compress' if dry_run else 'Compressing'}: {path} ({human_size(size)})")
        if not dry_run:
            try:
                with open(path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                new_size = os.path.getsize(gz_path)
                os.remove(path)
                freed += size - new_size
                print(f"    {human_size(size)} -> {human_size(new_size)}")
            except (OSError, IOError) as e:
                print(f"    Error: {e}")
                if os.path.exists(gz_path):
                    os.remove(gz_path)

    print(f"\n  Total freed: {human_size(freed)}")
    return freed


def main():
    parser = argparse.ArgumentParser(description="Disk cleanup utility")
    parser.add_argument("--path", required=True, help="Directory to scan")
    parser.add_argument("--scan", action="store_true", default=True,
                        help="Scan and report only (default)")
    parser.add_argument("--clean", action="store_true",
                        help="Actually perform cleanup")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what --clean would do without doing it")
    parser.add_argument("--min-size", type=int, default=10,
                        help="Minimum file size in MB to consider (default: 10)")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of largest files to show (default: 30)")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = scan_directory(args.path, min_size_mb=args.min_size, top_n=args.top)
    print_report(report)

    if args.clean or args.dry_run:
        if args.clean and not args.dry_run:
            print(f"\n⚠  About to clean {args.path}")
            resp = input("  Type 'yes' to confirm: ")
            if resp.strip().lower() != "yes":
                print("  Aborted.")
                return
        do_clean(report, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
