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

The structures for the peptide-peptide, peptide-MHC, and MHC-MHC networks will be constructed and saved to `data/` directory.

## 3. Train a prediction model
```
python -u train_nodesplit.py \
    --seed 16 \
    --hidden_dim 128 \
    --seq_model 'cnn' \
    --disjoint_rate 0.4 \
    --val_rate 0.1 \
    --max_epoch 300 \
    --early_stop 20 \
    --lr 0.001
```
The `seq_model` argument specifies the sequence encoder to be used. The `disjoint_rate` argument defines the ratio of edges for supervised learning.
The `val_rate` argument specifies the ratio of edges for validation, and if validation performance does not improve within `early_stop` epochs, the training process will be halted.

After the training process completes, the prediction model will be saved to `ckpt/` directory
