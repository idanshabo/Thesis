# 🧬 Protein Family Splitting via Matrix Normal Models & ESM Embeddings  
**Author:** Idan Shabo  

This repository contains a bioinformatics and machine learning pipeline designed to analyze protein families.  
The goal is to identify functional sub-families (splits) within a Multiple Sequence Alignment (MSA) by distinguishing evolutionary relatedness from functional distinctness.

To achieve this, the pipeline integrates:

- **Phylogenetics** — evolutionary tree construction  
- **Protein Language Models (ESM)** — feature extraction  
- **Matrix Normal Models** — handling non-i.i.d biological data  

---

# 🚀 1. Quick Start: Running the Pipeline

The pipeline is modular and optimized for **Google Colab** or **Linux**.

### Prerequisites
- Python 3.x  
- FastTree (must be in system PATH): `apt-get install fasttree`  
- GPU (recommended for ESM embeddings)

---

## Execution Script (Python — Google Colab example)

```python
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
run_pipeline(msa_file_path,
             print_file_content=False,       # Change to True if you want to print the raw MSA content
             output_path=None,               # Change to your desired output path, default is MSA location
             number_of_nodes_to_evaluate=15, # Number of splits to check
             pca_min_variance=0.99,          # Min variance for PCA to retain
             pca_min_components=100,         # Dimensionality of the embeddings to retain
             standardize=True                # Normalize data before modeling
)
```

---

## Script Outputs

In the MSA file directory, the pipeline generates two folders:

1. **Calculation folder**  
   Contains all intermediate calculations  
   (e.g., `PF00900_calculations/`)

2. **Outputs folder**  
   Includes plots for each significant split:  
   - MSA plot  
   - Protein covariance plot  
   - TM-score plot (predicted structures)  
   - TM-score plot (experimental structures, when available)  
   - PCA variance distribution plot  

---

# ⚙️ 2. Pipeline Architecture & Logic

The function **`run_pipeline`** orchestrates a multi-stage workflow.  
Below is the detailed logic.

---

### 1. Phylogenetic Tree Construction
- **Input:** Raw MSA  
- **Action:** Calls FastTree (LG model) to build a maximum-likelihood tree  
- **Purpose:** Defines evolutionary relationships required by later steps  

### 2. Phylogenetic Covariance Estimation
- Sequences are **not i.i.d**  
- Convert tree distances into a covariance matrix **U**  
- Represents pure evolutionary correlation between sequences  

### 3. ESM Embedding & Dimensionality Reduction
- MSA sequences → **ESM embeddings**  
- PCA reduces dimensionality (controlled by `pca_min_components`)  

### 4. Matrix Normal Modeling (Core Method)

The model:

\[
X \sim MN_{n \times p}(M, U, V)
\]

- **X:** protein embeddings  
- **U:** phylogenetic covariance  
- **V:** feature covariance  

The pipeline fits model parameters conditioned on the evolutionary tree.

### 5. Splitting & Model Selection (BIC)

- Tests a range of possible sub-family counts  
- Computes **Bayesian Information Criterion (BIC)**  
- Selects the optimal split with the lowest BIC  

---

# 3. Outputs & Logs

The pipeline prints real-time logs such as:  
- “Running FastTree…”  
- “Generating Embeddings…”  
- “Fitting Model k=…”  
- “BIC Score…”  

### Return Object
The function returns a dictionary containing:

- **best_split** — optimal number of clusters  
- **labels** — cluster IDs per sequence  
- **bic_history** — all BIC scores  
- **covariance_matrix** — computed phylogenetic covariance  

---

# 4. Repository Structure

- [pipeline_files/](pipeline_files/): Contains the core logic scripts (`msa_to_split_full_pipeline.py`).  
- [convert_stockholm_to_fasta.py](convert_stockholm_to_fasta.py): Helper script for format conversion.  
- [requirements.txt](requirements.txt): Python dependencies.  
