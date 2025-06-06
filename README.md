# scGPT Model Analysis

This repository contains code for analyzing gene expression data and embeddings using a scGPT model. The analysis includes generating embeddings for gene expression data across different timepoints and visualizing the relationships between genes.

## Features

- Generation of gene embeddings from expression data
- Analysis of gene similarities across different timepoints
- Visualization tools including:
  - Expression heatmaps
  - PCA visualization
  - t-SNE clustering
  - UMAP visualization
  - Gene trajectory analysis
  - Expression vs. embedding correlation plots

## File Structure

- `find_similar_genes.py`: Script for analyzing gene similarities based on embeddings
- `plot_embeddings.py`: Visualization tools for gene embeddings and expression data

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- umap-learn

## Usage

1. Prepare your gene expression data in CSV format
2. Run the embedding generation script:
```bash
python find_similar_genes.py
```

## Output

The scripts generate several visualization files:
- `gene_similarities_t{timepoint}.png`: Heatmaps showing similarity patterns at each timepoint
- `most_similar_gene_pairs.csv`: CSV file containing similarity data between gene pairs

## Author

Zarifeh Heidari Rarani
Postdoctoral Associate
Department of Immunology, Center for Systems Immunology
University of Pittsburgh 