# 🧬 Protein Family Splitting via Matrix Normal Models & ESM Embeddings  
**Author:** Idan Shabo  

This repository contains a bioinformatics and machine learning pipeline designed to analyze protein families.  
The goal is to identify functional sub-families (splits) within a Multiple Sequence Alignment (MSA) by distinguishing evolutionary relatedness from functional distinctness.

To achieve this, the pipeline integrates:
- **Phylogenetics** — evolutionary tree construction  
- **Protein Language Models (ESM)** — structural or sequence feature extraction  
- **Matrix Normal Models & LRT** — handling non-i.i.d biological data using Likelihood Ratio Tests and Parametric Bootstrapping

---

# 🚀 1. Quick Start: Running the Pipeline

The pipeline is now **highly modular** and driven by a Command-Line Interface (CLI). You can run the entire pipeline at once, or run specific operations (preprocessing, splitting, visualizing) independently.

### Prerequisites
- Python 3.x  
- FastTree (must be in system PATH): `apt-get install fasttree`  
- Internet connection (for fetching InterPro API metadata)
- GPU (recommended for ESM embeddings)

---

## Execution: Google Colab

**Cell 1: Setup Environment (Run Once per session)**
```python
from google.colab import drive
drive.mount('/content/drive')

import sys
!git clone [https://github.com/idanshabo/Thesis.git](https://github.com/idanshabo/Thesis.git)
!pip install -r /content/Thesis/requirements.txt
!apt-get install -y fasttree
```

**Cell 2: Run the Pipeline**
Use the `%cd` command to ensure Python finds the utility modules, then execute the CLI.

```bash
%cd /content/Thesis

!PYTHONPATH=/content/Thesis python pipeline_files/full_pipeline.py \
  --input "/content/drive/MyDrive/Thesis/protein_data/PF00818.alignment.full" \
  --family "PF00818" \
  --operation "full" \
  --embedding "structure" \
  --nodes 3 \
  --pca_var 0.99 \
  --standardize "TRUE" \
  --generate_plots "TRUE"
```

---

## Execution: Linux / Local Server

**1. Terminal Setup**
```bash
# Install FastTree (System dependency)
sudo apt-get update
sudo apt-get install -y fasttree

# Clone the repository
git clone [https://github.com/idanshabo/Thesis.git](https://github.com/idanshabo/Thesis.git)
cd Thesis

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Run via CLI**
You can run the pipeline directly from your terminal. Make sure to set the `PYTHONPATH` so the local imports resolve correctly.

```bash
PYTHONPATH=. python pipeline_files/full_pipeline.py \
  --input "protein_files/PF00818.alignment.full" \
  --family "PF00818" \
  --operation "full" \
  --embedding "sequence"
```

---

## 🎛️ CLI Arguments

| Argument | Description | Options/Default |
| :--- | :--- | :--- |
| `--input` | **(Required)** Path to the input MSA file. | *String* |
| `--family` | **(Required)** Family identifier (e.g., PF00818). | *String* |
| `--operation` | **(Required)** Which part of the pipeline to execute. | `preprocess`, `find_best_split`, `visualize`, `full` |
| `--embedding` | Type of ESM embedding to generate/use. | `sequence`, `structure` (Default: `sequence`) |
| `--nodes` | Number of top splits to evaluate. | *Integer* |
| `--pca_var` | Minimum variance for phylogenetic PCA. | *Float* (Skip pPCA if not provided) |
| `--standardize`| Apply standardization to data. | `TRUE`, `FALSE` (Default: `TRUE`) |
| `--generate_plots`| Generate visual outputs during the visualize step. | `TRUE`, `FALSE` (Default: `TRUE`) |

*Pro-Tip: Use `--operation preprocess` to quickly fetch InterPro metadata and MSA stats without running heavy computations.*

---

# ⚙️ 2. Pipeline Architecture & Logic

The `full_pipeline.py` script orchestrates a multi-stage workflow:

1. **Preprocessing & Metadata:** Parses the MSA, calculates basic statistics, and fetches functional metadata (e.g., membrane, kinase, viral flags) from the InterPro API.
2. **Phylogenetic Tree Construction:** Calls FastTree (LG model) to build a maximum-likelihood tree from the sequence alignment.
3. **Phylogenetic Covariance Estimation:** Converts tree distances into a phylogenetic covariance matrix (`U`), representing pure evolutionary correlation between sequences.
4. **ESM Embedding & pPCA:** Extracts ESM embeddings. Applies **Phylogenetic PCA (pPCA)** to reduce dimensionality while respecting the evolutionary covariance, isolating true functional variance.
5. **Matrix Normal Modeling & LRT (Core Method):** Models the protein embeddings as a Matrix Normal distribution conditioned on the evolutionary tree. Uses a **Likelihood Ratio Test (LRT)** and Westfall-Young Parametric Bootstrapping to evaluate the significance of potential sub-family splits.

---

# 📂 3. Outputs & Logs

The pipeline utilizes smart caching, meaning interrupted steps will not recompute heavy tasks (like FastTree or Covariance chunks) if they already exist in the calculations folder.

### Generated Directories
For a given family (e.g., `PF00818`), the pipeline generates:

1. **`PF00818_calculations/`** Contains all intermediate calculations:
   - Converted FASTA file
   - FastTree output (`.tree`)
   - Ordered Covariance Matrices
   - Raw and Aligned ESM embeddings

2. **`PF00818_outputs/`** Contains the final analysis:
   - **`pipeline_metadata.json`**: A comprehensive tracker containing MSA stats, API descriptions, runtime errors, and final split significance.
   - **`[embedding_mode]_embeddings/significant_splits/`**: A folder for each significant split (e.g., `rank1/`, `rank2/`) containing:
     - `split_rankX.json` (Taxa division and p-values)
     - Ordered MSA visualizations
     - Split covariance matrix heatmaps
     - Comparative sequence logos
     - TM-score distributions and Representative Structural Alignments (predicted and/or experimental).
