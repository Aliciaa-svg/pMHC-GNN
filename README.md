# pMHC-GNN

## 0. Installation
```
torch
torch-geometric
torch-cluster
torch-scatter
torch-sparse
transformers
pandas
scikit-learn
numpy
```

## 1. Encode raw peptide and MHC sequences
Run each code block in `encode.ipynb` sequentially.

The raw peptide and MHC sequences are encoded using a pretrained protein language model. 

The resulting initial embeddings will be saved to `data/` directory.

## 2. Graph construction
Run each code block in `relation.ipynb` sequentially.

The structures for the peptide-peptide, peptide-MHC, and MHC-MHC networks will be constructed and saved to `/data` directory.

## 3. Train a prediction model
