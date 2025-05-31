from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree
from create_esm_embeddings import create_esm_embeddings_from_fasta


def run_pipeline(MSA_file_path, model=ESMC.from_pretrained("esmc_300m"), seve_dir=None):
    create_esm_embeddings_from_fasta(MSA_file_path, save_dir)
    return save_dir
