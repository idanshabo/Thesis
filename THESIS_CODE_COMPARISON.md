# Thesis vs Code Comparison and Change Log

**Generated:** 2026-03-14
**Thesis draft:** idan_thesis.pdf (compiled 2026-03-03 from chapters in /chapters/)
**Code repo:** /mnt/c/Code/Github/idan_thesis (last commit: ab597f0, 2026-03-14)
**G Drive docs:** /mnt/g/My Drive/Students/IdanShabo/ (reviewed files dated 2026-03-01 to 2026-03-10)

---

## 1. Executive Summary

The **code has been extensively rewritten** since the thesis was drafted (Feb 26, 2026).
The thesis still describes the **old BIC-based pipeline**, while the code now implements
a fundamentally different **two-stage LRT + ANOVA architecture**. Additionally, the G
Drive documents describe several tools (OU comparison, interpretability analysis,
simulation validation, cross-PLM comparison, SLURM launcher) that **do not exist in the
current GitHub repo** -- they were written in advisory sessions but never pushed.

### Key Mismatches at a Glance

| Aspect | Thesis Says | Code Does | G Drive Says |
|--------|-------------|-----------|--------------|
| Split test | BIC (delta-BIC < 0) | LRT + Westfall-Young bootstrap | LRT correct |
| PLM model | ESM-3 | ESM-C 300M (+ ProstT5) | ESM-C 300M |
| Mean test | Not described | Phylogenetic ANOVA (RRPP) | ANOVA with caveats |
| PCA | Standard PCA | PhylogeneticPCA (R = X^T C^{-1} X / (n-1)) | pPCA correct |
| Candidate filtering | min_support=0.8, min_prop=0.1 | tree_alpha dual rule (no support) | tree_alpha correct |
| V interpretation | Not present | V_A, V_B saved as CSV only | Full tool implemented (NOT in repo) |
| OU model | Not present | Not in repo | Standalone tool (NOT in repo) |
| Simulation validation | Not present | Not in repo | Full framework (NOT in repo) |
| Cross-PLM comparison | Not present | Not in repo | CLI tool (NOT in repo) |
| ECCB paper | Not present | N/A | paper01.tex in G Drive |

---

## 2. Detailed Thesis-Code Mismatches

### 2.1 CRITICAL: Split Testing Method

**Thesis (Ch 3-4):** Uses BIC to compare M0 (single regime) vs M1 (split regime).
Significant if delta-BIC < 0. No p-value. Tests a mixture of mean shift + covariance
shift + tree structure effects simultaneously.

**Code:** Two-stage procedure:
- Stage 1: Recursive Phylogenetic ANOVA (RRPP) for mean shifts -> sub-families
- Stage 2: LRT for V_A = V_B with Westfall-Young bootstrap -> covariance splits
- Proper GLS mean estimation, bootstrap p-values, multiple testing correction

**Impact:** The entire Methods chapter (Ch 3, sections on split testing) and Results
chapter (Ch 4, all significance claims) need rewriting. The old BIC results and the
new LRT results are fundamentally different tests.

### 2.2 CRITICAL: PLM Model Identity

**Thesis:** Says "ESM-3" throughout (the multimodal model from Hayes et al. 2025 Science).

**Code:** Uses `ESMC.from_pretrained("esmc_300m")` which is **ESM-C 300M**, a completely
different model. ESM-C is a single-sequence model with 960-dim embeddings. ESM-3 is
multimodal (seq+struct+func) with 1536-dim embeddings.

**Impact:** Every mention of "ESM-3" in the thesis must be replaced with "ESM-C 300M".
Citations need updating. The implications for the circularity discussion change
(ESM-C is less circular than ESM-3).

### 2.3 HIGH: PCA Method

**Thesis (Ch 3):** Describes standard PCA, acknowledges IID assumption violation,
argues it's "just for dimensionality reduction."

**Code:** Implements `PhylogeneticPCA` class that computes the evolutionary VCV matrix
R = X^T C^{-1} X / (n-1) and eigendecomposes it. This properly accounts for
phylogenetic non-independence.

**Impact:** Thesis should describe pPCA instead of standard PCA. Cite Clavel et al. 2019.

### 2.4 HIGH: Candidate Split Selection

**Thesis:** Describes selecting top splits by BIC ranking.

**Code:** Uses tree_alpha dual rule: each side of a candidate split must have >=
tree_alpha fraction of BOTH species count AND induced branch length.
Bootstrap support filtering has been completely removed.

**Impact:** Methods chapter needs updating.

### 2.5 MEDIUM: Thesis says "alignment-free"

**Thesis abstract:** Claims "alignment-free method."

**Reality:** The pipeline takes an MSA as input and builds the tree from it.
The embedding step is per-sequence (alignment-free), but the overall pipeline
is alignment-dependent. Should say "leverages alignment-free protein representations."

### 2.6 MEDIUM: Results still reference BIC

**Thesis (Ch 4):** All results use BIC language ("delta-BIC < 0", "BIC parameter
penalty"). Figure captions reference BIC. The 13-family summary table is based on
BIC split counts.

**Code outputs (0_pipeline_outputs/):** Now use LRT Lambda statistics and p_adj values.
The 4 families in the repo show results from the new pipeline.

### 2.7 MINOR: Nucleotide vs amino acid

**Thesis p.3, p.8:** References "A, C, G, T" (nucleotides) in context of protein
phylogenetics. Should use amino acid references since the pipeline works on proteins.

### 2.8 MINOR: Title grammar

"Proteins Embeddings" should be "Protein Embeddings."

### 2.9 MINOR: Pybus et al. 2012 citation

Thesis cites Pybus et al. 2012 as proposing the MND framework for phylogenetics.
This paper is about spatial epidemiology of HIV. Better references: Revell & Collar
2009, Felsenstein 1985, Harmon 2019 book.

### 2.10 MINOR: Placeholder text

Thesis p.6: "[INSERT EXACT PARAMETER SCALE, e.g., 1.4B, 7B, or 98B]" not filled in.

---

## 3. Files Referenced in G Drive Documents But MISSING from Code Repo

These files were written during advisory sessions (by Claude) but never pushed to GitHub:

| File | Purpose | Status |
|------|---------|--------|
| `evaluate_split_options/ou_model.py` | BM vs OU comparison (4 functions) | NOT IN REPO |
| `evaluate_split_options/interpretability_analysis.py` | Mean shift + V decomposition (6 functions) | NOT IN REPO |
| `pipeline_files/compare_bm_ou.py` | Standalone BM vs OU CLI | NOT IN REPO |
| `pipeline_files/interpretability_analysis_cli.py` | Standalone interpretability CLI | NOT IN REPO |
| `pipeline_files/compare_plm_embeddings.py` | Cross-PLM comparison CLI | NOT IN REPO |
| `pipeline_files/simulate_test_validation.py` | Generic simulation framework | NOT IN REPO |
| `cluster/run_simulations.sh` | SLURM launcher for simulations | NOT IN REPO |
| Various ESM-3/CLSS/ProTrek support in `create_esm_embeddings.py` | Multi-PLM modes | PARTIALLY IN REPO (ProstT5 yes, others unclear) |
| `new_thesis_sections.tex` | New thesis sections for simulation/circularity/cross-PLM | IN G DRIVE ONLY |

**Action needed:** These files need to be pushed to GitHub. The project_summary_for_future_chats.txt
explicitly lists this as Action Item #2.

---

## 4. Changes Since the G Drive Documents Were Written (2026-03-10 -> 2026-03-14)

### What changed in code (GitHub commits after 2026-03-10):

Recent commits:
- `ab597f0` Add 0_pipeline_outputs (sample outputs for 4 families)
- `a33ed70` Update evaluate_split_options.py
- `856a8bd` Update full_pipeline_mean_shift.py
- `2e7bed3` Update full_pipeline_mean_shift.py
- `f088fce` Update visualisations.py

Key observations:
1. **Pipeline outputs were added** -- 4 families (PF00321, PF01340, PF14549, PF16168)
   with both sequence and structure embedding results. These are NEW since the G Drive
   documents were written.
2. The pipeline has been **run successfully** on these families with the new architecture.
3. Results show the two-stage pipeline works: ANOVA detects 5-9 sub-families per family,
   and the LRT finds 1-5 significant covariance splits per family.
4. The `evaluate_split_options.py` and `full_pipeline_mean_shift.py` were updated
   (likely bug fixes or parameter tuning).

### What the G Drive documents expected but hasn't happened:

1. **Simulation runs on cluster:** The simulate_test_validation.py was supposed to be
   run on HUJI cluster. Status: unknown (file not in repo, no results visible).
2. **ECCB paper completion:** Deadline was 2026-03-16 (2 days from now). The paper
   (paper01.tex) has TODO placeholders for simulation numbers and application results.
3. **Missing files not pushed:** ou_model.py, interpretability tools, etc.
4. **Thesis rewrite:** The thesis chapters still describe the old BIC pipeline.

---

## 5. Known Bugs Still Present in Code

### HIGH Priority

1. **Sub-matrix min-shift in ANOVA** (`recursive_tree_traversal.py:~89`):
   `C_local = torch.clamp(C_local - torch.min(C_local), min=0.0)`
   Should subtract clade-root distance, not global minimum. Affects ONLY the ANOVA
   (Phase 2); pPCA and LRT now use unshifted U_local.

2. **ANOVA multiple testing:** K candidate splits tested at each recursive level
   with no correction for multiple testing. Raw minimum p-value compared to
   anova_alpha=0.05. Inflates Type I error for sub-family detection.
   Location: `recursive_tree_traversal.py:~168-192`

### MEDIUM Priority

3. **Double normalization:** `create_normalized_mean_embeddings_matrix.py` applies
   global scalar normalization before the pipeline's per-dimension standardization.

4. **n-1 vs n inconsistency:** pPCA uses R = X^T C^{-1} X / (n-1), while LRT uses
   V_hat = S/n. Both valid but inconsistent (cosmetic, eigenvectors unaffected).

### LOW Priority

5. **O(n^3) covariance computation** in phylogenetic_tree_to_covariance_matrix.py
6. **Sequential bootstrap loop** (not parallelized)
7. **Dead import** in create_esm_embeddings_pipeline.py (`create_esm_embeddings_from_fasta`)
8. **ESMFold silently skips** sequences > 400 residues

---

## 6. Thesis Sections That Need Updating

### Must Rewrite:
- **Ch 3 Methods:** Replace BIC split test description with two-stage ANOVA + LRT
- **Ch 3 Methods:** Replace "ESM-3" with "ESM-C 300M" throughout
- **Ch 3 Methods:** Replace standard PCA description with PhylogeneticPCA
- **Ch 4 Results:** All significance claims need updating (BIC -> LRT p-values)
- **Ch 4 Results:** 13-family summary table needs redo with new pipeline results
- **Abstract:** Fix "alignment-free" claim

### Must Add:
- Circularity discussion (in Ch 5 Discussion)
- Simulation validation results (Ch 4 or appendix)
- ProstT5 as second PLM
- Candidate split selection description (tree_alpha dual rule)
- Proper citations: Clavel et al. 2019, Revell & Collar 2009, Cao et al. 2025,
  Collyer et al. 2015, Hie et al. 2022

### Already Written (in G Drive, needs merging):
- new_thesis_sections.tex: simulation protocol, circularity discussion, cross-PLM
- hypothesis_testing_LRT_subsection.tex: LRT derivation

---

## 7. What the Pipeline Output Results Show

The 0_pipeline_outputs/ directory contains actual results from 4 Pfam families
run through the current (post-rewrite) pipeline:

| Family | Seqs | Subfamilies | Sig Splits (seq) | Sig Splits (struct) |
|--------|------|-------------|-------------------|---------------------|
| PF00321 | 335 | 9 | 3 | 1 |
| PF01340 | ? | ? | 3 | 5 |
| PF14549 | ? | ? | 2 | 4 |
| PF16168 | ? | ? | 1 | 5 |

Key observations:
- Structure embeddings (ProstT5) often find MORE significant covariance splits
  than sequence embeddings (ESM-C 300M). This is interesting and potentially
  publishable.
- All significant splits have p_adj = 0.001 (1/1001), meaning the observed
  LRT statistic exceeded all 1000 bootstrap replicates.
- pPCA dimension varies dramatically across sub-families (from 1 to 24 components),
  reflecting different amounts of phylogenetic variance.

---

## 8. Immediate TODO List (Priority Order)

### For ECCB Paper (deadline: 2026-03-16):
1. [ ] Fill in simulation TODO placeholders in paper01.tex (need cluster results)
2. [ ] Fill in application TODO placeholders with results from 0_pipeline_outputs/
3. [ ] Compile and proofread paper01.tex

### For Code Repo:
4. [ ] Push missing files to GitHub (ou_model.py, interpretability tools,
       simulate_test_validation.py, cluster scripts, compare_plm_embeddings.py)
5. [ ] Run simulations on cluster (Type I error, power, robustness)
6. [ ] Fix ANOVA min-shift bug (use clade-root distance instead of global min)
7. [ ] Add multiple-testing correction to ANOVA recursive step

### For Thesis:
8. [ ] Replace "ESM-3" with "ESM-C 300M" throughout
9. [ ] Rewrite Methods chapter: BIC -> two-stage ANOVA + LRT
10. [ ] Rewrite Results chapter with new pipeline outputs
11. [ ] Add circularity discussion (from new_thesis_sections.tex)
12. [ ] Add simulation validation section (from new_thesis_sections.tex)
13. [ ] Update PCA -> pPCA description
14. [ ] Fix minor issues (nucleotides, title grammar, Pybus citation, placeholder)
15. [ ] Add missing references (Clavel 2019, Revell 2009, Cao 2025, Collyer 2015)

### Nice-to-Have:
16. [ ] Run interpretability analysis on significant splits
17. [ ] Run BM vs OU comparison
18. [ ] Run cross-PLM comparison (ESM-C vs ProstT5 vs CLSS)
19. [ ] Remove dead code (deprecated pipelines, unused BIC functions)
20. [ ] Parallelize bootstrap loop
