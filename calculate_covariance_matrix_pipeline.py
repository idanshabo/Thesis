from read_stockholm_file_and_print_content import read_stockholm_file_and_print_content
from convert_stockholm_to_fasta import convert_stockholm_to_fasta
from create_phylogenetic_tree.run_fasttree import run_fasttree
from phylogenetic_tree_to_covariance_matrix import tree_to_covariance_matrix

def run_pipeline(MSA_file_path, print_file_content=False):
    if print_file_content:
        read_stockholm_file_and_print_content(MSA_file_path)
    fasta_file_path = convert_stockholm_to_fasta(MSA_file_path)
    phylogenetic_tree_path = run_fasttree(fasta_file_path)
    cov_mat = tree_to_covariance_matrix(phylogenetic_tree_path)
    return cov_mat
