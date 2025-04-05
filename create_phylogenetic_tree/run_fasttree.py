def run_fasttree(fasta_file, output_path):
    """Run FastTree on protein alignment with output to chosen location"""
    cmd = [
        'FastTree',
        '-lg',                  # LG model
        '-gamma',              # Gamma rate categories
        '-out', output_path,   # Output to Drive
        fasta_file
    ]

    try:
        result = subprocess.run(cmd,
                              check=True,
                              capture_output=True,
                              text=True)
        print("FastTree analysis completed successfully")
        print(f"Tree saved to: {output_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)
        print("STDOUT:", e.stdout)
        raise
