from create_esm_embeddings import create_esm_embeddings_from_fasta


def run_pipeline(MSA_file_path, output_path=None):
    if not output_path:
        base_path = os.path.splitext(MSA_file_path)[0].replace('.fasta', '')
        output_path = base_path + '/embeddings_output'
    create_esm_embeddings_from_fasta(MSA_file_path, output_path)
    return save_dir
