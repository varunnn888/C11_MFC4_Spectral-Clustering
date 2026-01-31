# Density-Adaptive Graph Laplacian Construction for Robust Spectral Clustering

Overview
-
This repository implements a density-adaptive (self-tuning) affinity construction and spectral clustering pipeline applied to the EuroSAT dataset. The included script extracts ResNet50 embeddings, builds two affinity graphs (global-sigma and self-tuning local-scale), runs spectral clustering, and saves labeled montages and cluster purity metrics.

## Course
**22MAT230 – Mathematics for Computing IV**  
Amrita Vishwa Vidyapeetham, Coimbatore

---

## Team Details

**Team 11 – Batch C**

| Name | Roll Number |
|-----|------------|
| Jovika N B | CB.SC.U4AIE24223 |
| P Poojitha | CB.SC.U4AIE24237 |
| Potnuru Varun| CB.SC.U4AIE24244 |
| Ramu Jahna Bindu| CB.SC.U4AIE24249 |

---
## Objective

- The objective of this project is to design and implement a density-adaptive spectral clustering framework for unsupervised clustering of satellite images. The goal is to improve clustering performance in datasets with non-uniform density and complex visual structures, such as remote sensing imagery.

## Motivation

- Traditional clustering algorithms such as K-Means assume uniform data distribution and spherical clusters, which often fail for real-world datasets like satellite images. The EuroSAT dataset contains land-cover classes with varying densities and overlapping visual patterns (e.g., rivers and highways).
- This project is interesting because it applies graph theory and spectral analysis to adapt similarity computation locally, leading to more robust and meaningful clustering results.
## Spectral Clustering – Key References

1. **Ng, Jordan & Weiss (2001)**  
   *On Spectral Clustering: Analysis and an Algorithm*
   - Introduced the widely used spectral clustering algorithm.  
   - Method: similarity graph → Laplacian → eigenvectors → k-means.  
   - Established spectral clustering as a robust alternative to traditional clustering.  
   [Read here](https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf)

2. **Zelnik-Manor & Perona (2004)**  
   *Self-Tuning Spectral Clustering* 
   - Tackled sensitivity to scale parameter (σ).  
   - Proposed **local scaling** for robustness across varying data densities.  
   - Made spectral clustering more practical for real-world datasets.  
   [Read here](https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf)

3.  **von Luxburg (2007)**  
   *A Tutorial on Spectral Clustering* 
   - Comprehensive overview of theory and practice.  
   - Explained graph Laplacians, normalized cuts, and algorithmic variants.  
   - Provided intuition and guidance for applications.  
   [Read here](https://link.springer.com/article/10.1007/s11222-007-9033-z)

---

Key script
-
- [eurosat_full_density_adaptive_spectral.py](eurosat_full_density_adaptive_spectral.py) — main runnable script that performs dataset loading, embedding extraction, graph construction (global and self-tuning), spectral clustering, evaluation, and visualization.

Repository structure (important files / folders)
-
- [eurosat_full_density_adaptive_spectral.py](eurosat_full_density_adaptive_spectral.py)
- [outputs_eurosat_labeled](outputs_eurosat_labeled) — generated outputs (embeddings, affinity matrices, predictions, images, CSVs).

## Methodology

### 1. Dataset Preparation
- Satellite image data is preprocessed to ensure **uniform resolution** and **consistency** across samples.

### 2. Feature Extraction
- Texture features are extracted using **Histogram of Oriented Gradients (HOG)**.  
- Images are transformed into **high-dimensional feature vectors**.

### 3. Distance Computation
- Pairwise distances between feature vectors are computed.  
- Captures **similarity relationships** in feature space.

### 4. Fixed-Scale Graph Construction (Baseline)
- A similarity graph is built using a **global Gaussian kernel** with a fixed scale parameter.  
- Represents the **standard spectral clustering approach**.

### 5. Density-Adaptive Graph Construction (Proposed Method)
- Incorporates **local density information**.  
- Each data point is assigned a **local scale** derived from its *k-nearest neighbors*.  
- Produces a **density-adaptive similarity graph**.

### 6. Graph Laplacian Formation
- **Normalized graph Laplacians** are constructed for both fixed-scale and density-adaptive graphs.

### 7. Spectral Analysis
- **Eigenvalues and eigenvectors** of the Laplacians are analyzed.  
- Studies **spectral stability** and **robustness** under varying data densities.

### 8. Spectral Embedding and Clustering
- Low-dimensional **spectral embeddings** are obtained from Laplacian eigenvectors.  
- Used for **unsupervised clustering** of satellite images.

### 9. Visualization and Evaluation
- Visualizations include:  
  - Spectral embeddings  
  - Eigenvalue spectra  
  - Clustered image outputs  
- Comparison between **fixed-scale** and **density-adaptive** approaches.

## Mathematical Techniques Used:

- Deep feature extraction using pretrained CNN (ResNet50)

- k-Nearest Neighbor (k-NN) graph construction

- Density-adaptive (self-tuning) Gaussian affinity matrix

- Graph Laplacian normalization

- Spectral embedding

- K-Means clustering 
---



Typical outputs produced
-
- `embeddings.npy` — ResNet50 features (cached to avoid repeated extraction)
- `y_true.npy` — saved ground-truth labels for the subset
- `y_pred_kmeans.npy`, `y_pred_global_sigma.npy`, `y_pred_self_tuning.npy` — clustering results
- `affinity_global_sigma.npy`, `affinity_self_tuning.npy` — affinity matrices
- `cluster_purity_table.csv` — per-cluster purity table for each method
- Labeled montage PNGs like `self_tuning_cluster_0_LABELED.png` for visual inspection
- Eigenvalue plots and cluster size plots

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


Notes
-
- The script caches `embeddings.npy` inside `outputs_eurosat_labeled` to speed repeated runs. Delete it to re-extract features.
- If you have a CUDA-capable GPU and the correct PyTorch installation, the script will automatically use `cuda`.
- If a class folder is missing the script raises a helpful FileNotFoundError; verify `DATASET_ROOT` and dataset layout.
  
## Project Updates
- Successfully implemented **feature extraction** for the complete satellite image dataset.  
- Constructed both **fixed-scale** and **density-adaptive similarity graphs**.  
- Implemented **normalized graph Laplacian computation**.  
- Performed **eigenvalue analysis** and **spectral embedding**.  
- Observed **improved spectral stability** with the density-adaptive Laplacian.  
- Generated **visual evidence** through spectral plots and clustered image grids.
  
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

## Challenges / Issues Faced :

- **Parameter Sensitivity**  
  Standard spectral clustering is highly sensitive to the choice of the global scale parameter (σ), making results unstable.

- **Heterogeneous Data Densities**  
  Satellite imagery often contains regions with varying densities, complicating similarity graph construction and clustering accuracy.

- **Spectral Interpretation**  
  Correctly interpreting spectral embeddings and eigenvalue behavior requires careful analysis to avoid misrepresentation of cluster structures.

- **Theory vs. Implementation Consistency**  
  Ensuring mathematical consistency between theoretical formulations and practical implementation posed significant challenges.

- **Graph Construction Debugging**  
  Debugging similarity graph construction and aligning the implementation with reference papers demanded meticulous validation and troubleshooting.

  ## Future Implementation

- **Scalability**  
  Extend the method to **larger and more diverse datasets** to validate robustness across different domains.

- **Adaptive Neighborhood Selection**  
  Develop techniques to **automatically select neighborhood size** for density estimation, reducing manual parameter tuning.

- **Computational Efficiency**  
  Optimize algorithms to improve **efficiency for large-scale graphs**, enabling faster spectral analysis.

- **Cross-Domain Testing**  
  Apply the approach to **other application domains** (e.g., medical imaging, social networks, text data) to assess generalizability.

- **Integration with Deep Learning**  
  Combine **density-adaptive Laplacians** with **deep learning features** for hybrid models that leverage both spectral methods and representation learning.
