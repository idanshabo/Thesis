# 🧬 Thesis Pipeline – Idan Shabo

Welcome to the GitHub repository for my thesis work, focused on analyzing protein families using phylogenetics, protein language models, and statistical modeling. The project explores whether certain proteins can fold into two different conformations, combining bioinformatics and machine learning approaches.

---

## 📌 Objective

To identify and analyze proteins that may adopt multiple conformations by:
- Extracting and processing MSA (Multiple Sequence Alignment) data.
- Constructing and visualizing phylogenetic trees.
- Generating protein embeddings using ESM models (of META).
- Estimating statistical distributions over the embeddings.

---

## 🧪 Pipeline Overview

### 1. Protein Selection
- **Goal**: Find proteins that fold in two distinct ways.
- **Status**: Not yet started.

---

### 2. Retrieve MSA from InterPro (Pfam)
- MSA files are **manually downloaded** from InterPro due to API issues.
- Current protein family: **`PF03618`** *(others to be added later)*.

---

### 3. Convert MSA Format
- Input: `Stockholm (.sto)`  
- Output: `FASTA (.fasta)`
- Script: [`convert_stockholm_to_fasta.py`](#) *(update with actual link)*

---

### 4. Construct Phylogenetic Tree
- Tool: **FastTree** with **LG model**.
- Output: Newick-formatted `.tree` file.
- Interpretation: Branch length = substitutions per site.
- Script: [Tree calculation code](#)

---

### 5. Visualize Phylogenetic Tree
- Tool: [iTOL Web Viewer](https://itol.embl.de/)
- Usage: Upload your `.tree` file and explore interactively.

---

### 6. Estimate Covariance Matrix
- Based on the generated phylogenetic tree.
- Validated via:
  - Tree distance vs. covariance correlation.
  - Intra- vs inter-phylum distance comparisons.
- Code: [Covariance matrix estimation](#)

---

### 7. Generate Protein Embeddings
- Model: **ESM-3 / ESMC (2024)**  
- Token-level embeddings: `960`-dim vectors per amino acid.  
- Final output: Mean embedding per protein.
- Scripts:
  - [Generate per-token embeddings](#)
  - [Aggregate mean embeddings](#)

---

### 8. Fit Matrix Normal Distribution
- Estimation via **MLE** based on literature:
  - [MLE Paper 1](#)
  - [MLE Paper 2](#)
- Implemented cases:
  - Both `U`, `V` unknown
  - `U` known, `V` unknown
- Sampling using `scipy.stats.matrix_normal` and `wishart`.
- Code: [Fit function + tests](#)

---

## ✅ Current Status
- Most components are implemented and tested.
- Work on supporting additional protein families is ongoing.
- All steps are modular for easy updates.

---

## 📂 Repository Structure (Example)

