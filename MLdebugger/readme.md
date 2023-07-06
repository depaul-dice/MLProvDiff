# 1. Setup
## Installment of torch_geometric
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-geometric
```

Replace "TORCH" and "CUDA" with your own version of torch and cuda. We use torch 2.0.1 and cuda 11.7, so we use the following command:

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1+cu117.html
```

