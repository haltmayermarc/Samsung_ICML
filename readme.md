## Usage Guide

### Step1 - Create Training Set.
Using the 'assemble_LOD.py' code, you can genrate the training samples. For the coefficient type "--type" you can choose between 'quantile','checkerboard', 'horizontal' and 'vertical'.
```
python3 assemble_LOD.py --type quantile
```

### Step2 - Train the GNN
A miimalal command to train the GNN for the quantile-based coefficient is:
```
python3 train_GNN.py --type quantile --basis_order 1
```
If you want to train using the weak form loss:
```
python3 train_GNN.py --type quantile --basis_order 1 --loss weak_form
```
