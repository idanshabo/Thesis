from create_esm_embeddings import create_esm_embeddings_from_fasta


def run_pipeline(MSA_file_path, model=ESMC.from_pretrained("esmc_300m"), seve_dir=None):
    create_esm_embeddings_from_fasta(MSA_file_path, save_dir)
    return save_dir
