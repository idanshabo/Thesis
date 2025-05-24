import os
import subprocess


def run_fasttree(fasta_file_path, output_path=None):
    """Run FastTree on protein alignment with output to chosen location"""
    if not output_path:
        base_path = os.path.splitext(fasta_file_path)[0].replace('.fasta', '')
        output_path = base_path + '.tree'
    cmd = [
        'FastTree',
        '-lg',                  # LG model
        '-gamma',              # Gamma rate categories
        '-out', output_path,   # Output to Drive
        fasta_file_path
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
