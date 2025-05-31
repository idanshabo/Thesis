from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree

    fasta_file = '/content/drive/MyDrive/protein_data/PF03618.fasta'
    save_dir = "embeddings_output"  
    model = ESMC.from_pretrained("esmc_300m")


def run_pipeline(MSA_file_path, model=ESMC.from_pretrained("esmc_300m"), seve_dir=None):
    create_esm_embeddings_from_fasta(MSA_file_path, save_dir)
    return save_dir
