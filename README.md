# Pathogenic Variations Illuminate Functional Constraints in Intrinsically Disordered Proteins

This repository contains the source code and analysis pipeline for the study investigating the distribution, structural context, and functional impact of pathogenic missense variants within Intrinsically Disordered Regions (IDRs) compared to ordered protein regions.

The project integrates multi-omics data from **ClinVar**, **MobiDB**, **AlphaMissense** predictions, and various functional annotations including **ELM motifs**, **PTMs**, and **UniProt Regions of Interest (ROI)**.

## 📊 Project Overview

The repository is organized into two main components:

### 1. Analysis Pipeline (`paper_replication`)
This component reproduces the statistical analyses and figures presented in the manuscript:
* **Structural Analysis:** Distribution of variants across the proteome and structural composition (Figure 1).
* **Disease Ontology:** Analysis of genetic complexity (Monogenic vs. Polygenic) and IDR-specific mutations (Figure 2).
* **Functional Landscape:** Investigation of PDB coverage, linear motifs, and post-translational modifications (Figure 3).

### 2. Prediction Model (`prediction_model`)
This component implements machine learning experiments, specifically a **Decision Tree** model, to predict pathogenicity based on sequence features and linear motif context.

## 🛠 Modules

The core analysis logic is encapsulated in the `src` package:

* **`src.clinvar.stats`**: Handles structural statistics, proteome coverage, and gene-level overlaps.
* **`src.clinvar.ontology`**: Categorization of disease types and genetic complexity.
* **`src.clinvar.functional`**: Performs enrichment analyses for PTMs, ELMs, and other functional annotations.

## 🚀 Usage

### 1. Install Dependencies
Ensure you have Python 3.10.12 installed.

```bash
pip install requirements.txt
```
### 2. Get Data
The pre-processed datasets required for the analysis are available in the repository Releases.
#### 1. Download the `data.tar.bz2` file from the latest Release.
#### 2. Extract the archive into the repository root directory:

```bash
# Extract the data (Linux/Mac)
tar -xjf data.tar.bz2
```

### 3. Run Analysis

Navigate to the `notebooks/` directory to explore the [analysis_pipeline](notebooks/analysis_pipeline) or run the [prediction_model](notebooks/prediction_model).

#### *  [Analysis Pipeline](notebooks/analysis_pipeline) : Run these notebooks to reproduce the figures (Fig 1-3) from the article.
#### *  [Prediction Model](notebooks/prediction_modele) : Run these notebooks to train and evaluate the decision tree model. And predict your own motifs

## 📧 Contact
**Zsuzsanna Dosztányi**
*Dlab / Department of Biochemistry / Eötvös Loránd University*
[zsuzsanna.dosztanyi@ttk.elte.hu](mailto:zsuzsanna.dosztanyi@ttk.elte.hu)

**Norbert Deutsch**
*Dlab / Department of Biochemistry / Eötvös Loránd University*
[norbert.deutsch@ttk.elte.hu](mailto:norbert.deutsch@ttk.elte.hu)