# pMHC-GNN
pMHC-GNN is a heterogeneous Graph Neural Networks-based framework that integrates both sequence and structural information for peptide-MHC binding prediction.

## File description
* `encode.ipynb` encode raw peptide and MHC sequences using a pretrained protein language model.
* `relation.ipynb` construct peptide-peptide, peptide-MHC, and MHC-MHC networks based on the provided dataset and the initial embeddings of peptides and MHCs.
* `link_split.py` code for spliting the graph into training/validation/testing subgraphs.
* `model.py` code for model structure.
* `train_nodesplit.py` main code for training the model.

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

The raw peptide and MHC sequences are encoded using a pretrained protein language model [unikei/bert-base-proteins](https://huggingface.co/unikei/bert-base-proteins). 

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

**Checkpoint**
[checkpoint](https://drive.google.com/file/d/1qxTOZQomoqfzk3TVqH1rkz06u3_66Npy/view?usp=sharing)
