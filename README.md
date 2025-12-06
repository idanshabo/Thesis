# 🧬 Protein Family Splitting via Matrix Normal Models & ESM Embeddings
Author: Idan Shabo

This repository contains a bioinformatics and machine learning pipeline designed to analyze protein families. The core objective is to identify functional sub-families (splits) within a Multiple Sequence Alignment (MSA) by distinguishing between evolutionary relatedness and functional distinctness.
To achieve this, the pipeline combines Phylogenetics (to model evolutionary history), Protein Language Models (ESM for feature extraction), and Matrix Normal Distributions (for statistical modeling of non-i.i.d biological data).

🚀 1. Quick Start: Running the Pipeline
The pipeline is designed to be modular. Below is an example of how to run the full workflow programmatically, specifically optimized for Google Colab or a Linux environment.
Prerequisites
Python 3.x
FastTree: Must be installed and in your system PATH (apt-get install fasttree).
GPU: Recommended for ESM embedding generation.
Execution Script
Python
import sys
from google.colab import drive

# 1. Mount Data Source
drive.mount('/content/drive')

# 2. Setup Environment
# Clone the repository
!git clone https://github.com/idanshabo/Thesis.git

# Add repo to python path
sys.path.append('/content/Thesis')

# Install Python dependencies
!pip install -r /content/Thesis/requirements.txt

# Install FastTree (Required for step 1 of the pipeline)
!apt-get install -y fasttree

# 3. Import and Run
from pipeline_files.msa_to_split_full_pipeline import run_pipeline

# Define your input MSA
msa_file_path = '/content/drive/MyDrive/Thesis/protein_data/PF00900.alignment.full'

# Execute
results = run_pipeline(
    msa_file_path,
    number_of_nodes_to_evaluate=15,  # Max number of clusters (splits) to check
    pca_min_components=100,          # Dimensionality of the embeddings to retain
    standardize=True                 # Normalize data before modeling
)

print("Best Split Configuration:", results)


⚙️ 2. Pipeline Architecture & Logic
The function run_pipeline orchestrates a complex workflow. Unlike standard clustering, this pipeline explicitly accounts for the fact that protein sequences are not independent; they share an evolutionary history.
Here is the step-by-step logic:
Step 1: Phylogenetic Tree Construction
Input: Raw MSA file.
Action: The pipeline first calls FastTree (using the LG evolutionary model) to construct a Maximum Likelihood phylogenetic tree.
Purpose: The tree is not just for validation; it is a mathematical prerequisite for the statistical model. It defines the relationships between sequences.
Step 2: Phylogenetic Covariance Estimation
Logic: Standard machine learning assumes data points are independent (i.i.d). In biology, sequences are correlated.
Action: The pipeline converts the phylogenetic tree distances into a Covariance Matrix (often denoted as $U$ or $\Omega_{rows}$).
Result: A matrix representing how correlated every sequence is to every other sequence based purely on evolution.
Step 3: ESM Embedding & Dimensionality Reduction
Action: The MSA sequences are fed into the ESM (Evolutionary Scale Modeling) protein language model (Meta).
Output: High-dimensional numerical vectors (embeddings) representing the structural and functional properties of the proteins.
PCA: To make the statistical modeling computationally feasible, Principal Component Analysis (PCA) is applied (controlled by pca_min_components), reducing the data to the most informative features.
Step 4: Matrix Normal Modeling (The Core)
The Model: The pipeline models the data using a Matrix Normal Distribution ($MN$).
$$X \sim MN_{n \times p}(M, U, V)$$
$X$: The protein data.
$U$: The Phylogenetic Covariance Matrix (calculated in Step 2). This tells the model to "expect" similarity between related sequences.
$V$: The feature covariance (how protein features relate to each other).
Fitting: The pipeline estimates the parameters that best fit the observed embeddings, conditioned on the evolutionary tree.
Step 5: Splitting & Model Selection (BIC)
Search: The pipeline iteratively tests different numbers of sub-families ("nodes"), from 1 up to number_of_nodes_to_evaluate.
BIC: For each split configuration, it calculates the Bayesian Information Criterion (BIC).
The BIC penalizes complexity. It helps answer: "Does splitting this family into 3 groups explain the data better than 2 groups, even after accounting for the added complexity?"
Output: The configuration with the lowest BIC score is selected as the optimal biological split.

3. Outputs & Logs
Console Output
During execution, the pipeline prints:
"Running FastTree...": Logs from the external FastTree binary.
"Generating Embeddings...": Progress bars for the ESM model inference.
"Fitting Model k=...": Status of the Matrix Normal fitting for different cluster counts.
"BIC Score...": The calculated score for each tested split.
Return Object (results)
The function returns a dictionary/object containing:
best_split: The optimal number of clusters determined by BIC.
labels: The assigned cluster ID for each protein sequence.
bic_history: A record of scores for all evaluated $k$ values (useful for plotting the "elbow" curve).
covariance_matrix: The computed phylogenetic covariance used in the model.

4. Repository Structure
pipeline_files/: Contains the core logic scripts (msa_to_split_full_pipeline.py).
convert_stockholm_to_fasta.py: Helper script for format conversion.
requirements.txt: Python dependencies.

