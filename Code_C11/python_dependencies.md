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

Quick usage
-
1. Edit `DATASET_ROOT` in [eurosat_full_density_adaptive_spectral.py](eurosat_full_density_adaptive_spectral.py) to point to your EuroSAT dataset folder.
2. Optionally change `SELECTED_CLASSES`, `IMAGES_PER_CLASS`, `NUM_CLUSTERS`, and graph parameters (`KNN_GRAPH_K`, `LOCAL_SCALE_K`, `GLOBAL_SIGMA_MODE`) at the top of the script.
3. Run the script:

```bash
python eurosat_full_density_adaptive_spectral.py
```
