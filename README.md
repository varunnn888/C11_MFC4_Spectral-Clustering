# Density-Adaptive Graph Laplacian Construction for Robust Spectral Clustering

Overview
-
This repository implements a density-adaptive (self-tuning) affinity construction and spectral clustering pipeline applied to the EuroSAT dataset. The included script extracts ResNet50 embeddings, builds two affinity graphs (global-sigma and self-tuning local-scale), runs spectral clustering, and saves labeled montages and cluster purity metrics.

## Team Members

| Name | Roll Number |
|------|-------------|
| Jovika N B| CB.SC.U4AIE24223 |
| P Poojitha| CB.SC.U4AIE24237 |
| Potnuru Varun | CB.SC.U4AIE24244 |
| Ramu Jahna Bindu | CB.SC.U4AIE24249 |

---
## Reference Papers

1.Zelnik-Manor, L., & Perona, P. (2004).
  Self-Tuning Spectral Clustering.
  Advances in Neural Information Processing Systems (NeurIPS).
  https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf
  
2.von Luxburg, U. (2007).
  A Tutorial on Spectral Clustering.
  Statistics and Computing.
  https://link.springer.com/article/10.1007/s11222-007-9033-z

3.Ng, A. Y., Jordan, M. I., & Weiss, Y. (2001).
  On Spectral Clustering: Analysis and an Algorithm.
  Advances in Neural Information Processing Systems.
  https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf
Key script
-
- [eurosat_full_density_adaptive_spectral.py](eurosat_full_density_adaptive_spectral.py) — main runnable script that performs dataset loading, embedding extraction, graph construction (global and self-tuning), spectral clustering, evaluation, and visualization.

Repository structure (important files / folders)
-
- [eurosat_full_density_adaptive_spectral.py](eurosat_full_density_adaptive_spectral.py)
- [outputs_eurosat_labeled](outputs_eurosat_labeled) — generated outputs (embeddings, affinity matrices, predictions, images, CSVs).

Typical outputs produced
-
- `embeddings.npy` — ResNet50 features (cached to avoid repeated extraction)
- `y_true.npy` — saved ground-truth labels for the subset
- `y_pred_kmeans.npy`, `y_pred_global_sigma.npy`, `y_pred_self_tuning.npy` — clustering results
- `affinity_global_sigma.npy`, `affinity_self_tuning.npy` — affinity matrices
- `cluster_purity_table.csv` — per-cluster purity table for each method
- Labeled montage PNGs like `self_tuning_cluster_0_LABELED.png` for visual inspection
- Eigenvalue plots and cluster size plots

Requirements
-
- Python 3.8+
- Libraries (install via pip):
  - numpy
  - scipy
  - scikit-learn
  - pandas
  - matplotlib
  - pillow
  - tqdm
  - torch
  - torchvision

Install example
-
Create a virtual environment and install required packages:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install numpy scipy scikit-learn pandas matplotlib pillow tqdm torch torchvision
```

Dataset (EuroSAT) and expected layout
-
This script expects a local EuroSAT dataset folder organized like:

EuroSAT/
  └─ 2750/
     └─ Forest/
     └─ River/
     └─ SeaLake/
     └─ Residential/
     └─ Highway/

Set the dataset root in the script by editing the `DATASET_ROOT` constant at the top of [eurosat_full_density_adaptive_spectral.py](eurosat_full_density_adaptive_spectral.py). The script will try `os.path.join(DATASET_ROOT, "2750")` first and fall back to `DATASET_ROOT` if the subfolder isn't present.

Quick usage
-
1. Edit `DATASET_ROOT` in [eurosat_full_density_adaptive_spectral.py](eurosat_full_density_adaptive_spectral.py) to point to your EuroSAT dataset folder.
2. Optionally change `SELECTED_CLASSES`, `IMAGES_PER_CLASS`, `NUM_CLUSTERS`, and graph parameters (`KNN_GRAPH_K`, `LOCAL_SCALE_K`, `GLOBAL_SIGMA_MODE`) at the top of the script.
3. Run the script:

```bash
python eurosat_full_density_adaptive_spectral.py
```

Notes
-
- The script caches `embeddings.npy` inside `outputs_eurosat_labeled` to speed repeated runs. Delete it to re-extract features.
- If you have a CUDA-capable GPU and the correct PyTorch installation, the script will automatically use `cuda`.
- If a class folder is missing the script raises a helpful FileNotFoundError; verify `DATASET_ROOT` and dataset layout.

Configuration flags (top of script)
-
- `DATASET_ROOT` — path to EuroSAT dataset root
- `SELECTED_CLASSES` — list of classes to include (default five classes)
- `IMAGES_PER_CLASS` — how many images to sample per class
- `NUM_CLUSTERS` — number of clusters to compute
- `KNN_GRAPH_K` — k for k-NN graph edges
- `LOCAL_SCALE_K` — local scale neighbor index for self-tuning sigma
- `GLOBAL_SIGMA_MODE` — "median" or "mean" used for the global sigma baseline

Evaluations produced
-
- The script prints NMI, ARI, and best-mapped ACC for each method (KMeans baseline, spectral with global sigma, and spectral with self-tuning affinity).
- A `cluster_purity_table.csv` is saved summarizing per-cluster majority class and purity percentage.

Troubleshooting
-
- If you see an error about missing class folder, verify the `DATASET_ROOT` setting and that the expected class subfolders (e.g., `Forest`, `River`, `SeaLake`, `Residential`, `Highway`) exist. The script prints the dataset path it's using to help debugging.
- If embeddings extraction is slow, ensure `BATCH_SIZE` is reasonable and consider extracting once on GPU and reusing the saved `embeddings.npy`.



