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

## Execution Script (Python — Linux / Local Server)
To run this pipeline on a standard Linux environment (e.g., Ubuntu/Debian), use the following steps to set up the environment and execute the script.

1. Terminal Setup
Run these commands in your terminal to install system requirements and set up the Python environment.

```Bash
# 1. Install FastTree (System dependency)
sudo apt-get update
sudo apt-get install -y fasttree

# 2. Clone the repository
git clone https://github.com/idanshabo/Thesis.git
cd Thesis

# 3. Create and activate a virtual environment (Recommended)
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt
```

2. Run the Pipeline
Create a python script (e.g., run_analysis.py) in the root of the repository:

```Python
import os
import sys

# Add current directory to path to ensure local imports work
sys.path.append(os.getcwd())

from pipeline_files.msa_to_split_full_pipeline import run_pipeline

# --- Configuration ---
# Path to your input MSA file (Relative or Absolute path)
msa_file_path = 'protein_files/PF00900.alignment.full'

# Optional: Define a specific output directory
output_dir = './results/PF00900_Analysis'

# --- Execute ---
if __name__ == "__main__":
    print(f"Starting pipeline processing for: {msa_file_path}")
    
    run_pipeline(
        msa_file_path,
        output_path=output_dir,          # Results will be saved here
        print_file_content=False,        # Set to True to debug input reading
        number_of_nodes_to_evaluate=15,  # Number of splits to check
        pca_min_variance=0.99,           # Min variance for PCA to retain
        pca_min_components=100,          # Dimensionality of the embeddings to retain
        standardize=True                 # Normalize data before modeling
    )
    print("Pipeline finished successfully.")
```

Then, execute it from the terminal:

```Bash
python run_analysis.py
```
Note on GPU: This pipeline utilizes ESM models which are computationally intensive.

With GPU: Ensure you have the correct PyTorch version with CUDA support installed for fast processing.

CPU Only: The pipeline will run, but embedding generation will be significantly slower.

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

### 1. MSA file format conversion stockholm to fasta
- Converts raw MSA stockholm format to fasta format
 
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
- PCA reduces dimensionality (controlled both by `pca_min_components` and by `pca_min_variance`)  

### 4. Matrix Normal Modeling (Core Method)

The model:

\[
X \sim MN_{n \times p}(M, U, V)
\]

- **X:** protein embeddings  
- **U:** phylogenetic covariance  
- **V:** feature covariance (after dimentionality reduction using PCA)

The pipeline fits model parameters conditioned on the evolutionary tree.

### 5. Splitting & Model Selection (BIC)

- Tests a range of possible sub-family counts  
- Computes **Bayesian Information Criterion (BIC)** based on the matrix normal distribution 
- Selects significant splits that are also distinct from one another

---

# 3. Outputs & Logs

The pipeline prints real-time logs such as:  
- “Running FastTree…”  
- “Generating Embeddings…”  
- “Fitting Model k=…”  
- “BIC Score…”  

### Return Object
The pipeline saves many outputs to the chosen directiry. 
to the calculations directory: 
- **MSA file fasta format**
- **phylogenetic tree**
- **Covariance matrix** - based on the phylogenetic tree
- **ESM embeddings** - both raw, centered and normalized
- **ESM structre predictions**
- **PDB structures from experiments** - that are relevant for proteins in the specific PFAM

to the outputs directory:
- **results file** — containing BIC significance scores
- per every significant split:
  - **ordered MSA file divided into the 2 groups**
  - **covariance matrix visualization** — divided into the 2 groups
  - **side by side plot of covariance and tm scores** — for a sample of 50 proteins from every group
  - **structure visualization of a representative protein from each group** — both sive by side and aligned
  - **raw file containing the split information**

---

# 4. Repository Structure

- [pipeline_files/](pipeline_files/): Contains the full pipeline runners (`msa_to_split_full_pipeline.py`).  
- [requirements.txt](requirements.txt): Python dependencies.
- [protein_files](protein_files/) contain example for input and ourputs of the pipeline.
- other folders contain different parts of the logic.
