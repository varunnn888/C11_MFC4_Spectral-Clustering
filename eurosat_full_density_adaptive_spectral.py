import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize

from scipy.optimize import linear_sum_assignment


# ==========================================================
# 1) CONFIGURATION
# ==========================================================
DATASET_ROOT = r"C:\Users\VARUN\OneDrive\Desktop\MFC4\MFC4\EuroSAT_Dataset" #DATASET ROOT PATH

# EuroSAT typically has: EuroSAT/2750/<class folders...>
DATASET_PATH = os.path.join(DATASET_ROOT, "2750")
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = DATASET_ROOT

SELECTED_CLASSES = ["Forest", "River", "SeaLake", "Residential", "Highway"]
IMAGES_PER_CLASS = 400  # 5 x 400 = 2000 images

RANDOM_SEED = 42
NUM_CLUSTERS = 5

# Graph params
KNN_GRAPH_K = 25
LOCAL_SCALE_K = 10

# Global sigma baseline
GLOBAL_SIGMA_MODE = "median"  # "median" or "mean"

# Performance
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output folder
OUT_DIR = "outputs_eurosat_labeled"
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================================
# 2) REPRODUCIBILITY
# ==========================================================
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ==========================================================
# 3) LOAD DATASET SUBSET
# ==========================================================
def load_eurosat_subset(dataset_path, classes, images_per_class):
    X_paths = []
    y_labels = []

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    for cls in classes:
        folder = os.path.join(dataset_path, cls)

        if not os.path.exists(folder):
            raise FileNotFoundError(
                f"\n Class folder not found: {folder}\n"
                f" Your DATASET_PATH is: {dataset_path}\n"
                f" It must contain folders like: Forest, River, SeaLake...\n"
                f" Fix path or check dataset extraction.\n"
            )

        all_imgs = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        np.random.shuffle(all_imgs)
        chosen = all_imgs[:images_per_class]

        X_paths.extend(chosen)
        y_labels.extend([class_to_idx[cls]] * len(chosen))

    return X_paths, np.array(y_labels), class_to_idx, idx_to_class


# ==========================================================
# 4) RESNET50 EMBEDDINGS EXTRACTOR
# ==========================================================
class ResNet50Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # remove FC

    def forward(self, x):
        with torch.no_grad():
            z = self.backbone(x)  # (B, 2048, 1, 1)
            z = z.squeeze(-1).squeeze(-1)  # (B, 2048)
        return z


def extract_embeddings(image_paths, batch_size=32):
    weights = models.ResNet50_Weights.DEFAULT

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    ])

    model = ResNet50Embedder().to(DEVICE)
    model.eval()

    embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting ResNet50 embeddings"):
        batch_paths = image_paths[i:i + batch_size]

        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(preprocess(img))

        batch_tensor = torch.stack(batch_imgs).to(DEVICE)
        batch_emb = model(batch_tensor).cpu().numpy()
        embeddings.append(batch_emb)

    return np.vstack(embeddings)


# ==========================================================
# 5) METRICS 
# ==========================================================
def clustering_accuracy(y_true, y_pred):
    """
    Best mapping using Hungarian algorithm
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    D = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype=np.int64)

    for i in range(len(y_true)):
        cost[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    correct = cost[row_ind, col_ind].sum()
    return correct / len(y_true)


def evaluate_clustering(y_true, y_pred, method_name="method"):
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = clustering_accuracy(y_true, y_pred)

    print(f"\n================ {method_name.upper()} RESULTS ================")
    print(f"NMI  = {nmi:.4f}")
    print(f"ARI  = {ari:.4f}")
    print(f"ACC* = {acc:.4f} (after best mapping)")
    print("====================================================\n")

    return nmi, ari, acc


def plot_cluster_sizes(y_pred, save_path, title):
    plt.figure()
    unique, counts = np.unique(y_pred, return_counts=True)
    plt.bar(unique, counts)
    plt.title(title)
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.savefig(save_path, dpi=200)
    plt.show()


# ==========================================================
# 5.1) CLUSTER PURITY TABLE 
# ==========================================================
def generate_cluster_purity_table(y_true, y_pred, idx_to_class, method_name):
    """
    Purity(cluster) = majority_true_class_count / cluster_size
    """
    rows = []
    clusters = np.unique(y_pred)

    for cluster_id in clusters:
        cluster_indices = np.where(y_pred == cluster_id)[0]
        true_labels_cluster = y_true[cluster_indices]

        counts = np.bincount(true_labels_cluster, minlength=len(idx_to_class))

        majority_label = int(np.argmax(counts))
        majority_class = idx_to_class[majority_label]

        cluster_size = len(cluster_indices)
        majority_count = int(counts[majority_label])
        purity = majority_count / cluster_size if cluster_size > 0 else 0.0

        rows.append({
            "Method": method_name,
            "Cluster_ID": int(cluster_id),
            "Cluster_Size": int(cluster_size),
            "Majority_Class": majority_class,
            "Majority_Count": majority_count,
            "Purity_%": round(purity * 100, 2)
        })

    df = pd.DataFrame(rows).sort_values(by=["Cluster_ID"]).reset_index(drop=True)
    return df


# ==========================================================
# 6) LABELED MONTAGE
# ==========================================================
def get_font(size=14):
    """
    Safe font loader for Windows/Linux.
    """
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


def save_cluster_montage_labeled(image_paths, y_true, y_pred, cluster_id, idx_to_class,
                                 save_name, images_per_row=5, max_images=25, tile_size=160):
    chosen_indices = [i for i in range(len(image_paths)) if y_pred[i] == cluster_id]
    chosen_indices = chosen_indices[:max_images]

    if len(chosen_indices) == 0:
        print(f"⚠️ No images found for cluster {cluster_id}")
        return

    font = get_font(14)

    imgs = []
    for i in chosen_indices:
        img = Image.open(image_paths[i]).convert("RGB").resize((tile_size, tile_size))
        draw = ImageDraw.Draw(img)

        true_name = idx_to_class[int(y_true[i])]
        text1 = f"Pred: {cluster_id}"
        text2 = f"True: {true_name}"

        draw.rectangle([(0, 0), (tile_size, 35)], fill=(0, 0, 0))
        draw.text((5, 2), text1, fill=(255, 255, 255), font=font)
        draw.text((5, 18), text2, fill=(255, 255, 255), font=font)

        imgs.append(img)

    rows = int(np.ceil(len(imgs) / images_per_row))
    montage = Image.new("RGB", (tile_size * images_per_row, tile_size * rows), (255, 255, 255))

    for idx, img in enumerate(imgs):
        r = idx // images_per_row
        c = idx % images_per_row
        montage.paste(img, (c * tile_size, r * tile_size))

    montage.save(save_name)
    print(f" Saved LABELED montage: {save_name}")


# ==========================================================
# 7) GRAPH CONSTRUCTION
# ==========================================================
def build_self_tuning_affinity(X, knn_k=15, local_scale_k=7):
    n = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=max(knn_k, local_scale_k) + 1, metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    sigma = distances[:, local_scale_k]
    sigma[sigma < 1e-12] = 1e-12

    W = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for jpos in range(1, knn_k + 1):
            j = indices[i, jpos]
            dij = distances[i, jpos]
            W[i, j] = np.exp(-(dij ** 2) / (sigma[i] * sigma[j]))

    W = 0.5 * (W + W.T)
    return W


def build_global_sigma_affinity(X, knn_k=15, mode="median"):
    n = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=knn_k + 1, metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    all_knn_distances = distances[:, 1:].reshape(-1)
    sigma = np.median(all_knn_distances) if mode == "median" else np.mean(all_knn_distances)
    sigma = max(sigma, 1e-12)

    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for jpos in range(1, knn_k + 1):
            j = indices[i, jpos]
            dij = distances[i, jpos]
            W[i, j] = np.exp(-(dij ** 2) / (2 * (sigma ** 2)))

    W = 0.5 * (W + W.T)
    return W, sigma


# ==========================================================
# 8) SPECTRAL CLUSTERING
# ==========================================================
def spectral_clustering_from_affinity(W, n_clusters, method_tag="method"):
    D = np.sum(W, axis=1)
    D[D < 1e-12] = 1e-12
    D_inv_sqrt = 1.0 / np.sqrt(D)

    L = (W * D_inv_sqrt).T * D_inv_sqrt
    eigenvals, eigenvecs = np.linalg.eigh(L)

    plt.figure()
    plt.plot(eigenvals[::-1], marker="o")
    plt.title(f"Eigenvalues (Descending) - {method_tag}")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    save_eig = os.path.join(OUT_DIR, f"eigenvalues_{method_tag}.png")
    plt.savefig(save_eig, dpi=200)
    plt.show()

    U = eigenvecs[:, -n_clusters:]
    Y = normalize(U, norm="l2")

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
    y_pred = kmeans.fit_predict(Y)

    return y_pred, eigenvals


# ==========================================================
# 9) MAIN
# ==========================================================
def main():
    print(" Device:", DEVICE)
    print(" Dataset root:", DATASET_ROOT)
    print(" Dataset path used:", DATASET_PATH)
    print(" Selected classes:", SELECTED_CLASSES)

    # Load subset
    image_paths, y_true, class_to_idx, idx_to_class = load_eurosat_subset(
        DATASET_PATH, SELECTED_CLASSES, IMAGES_PER_CLASS
    )
    print(f" Total images loaded = {len(image_paths)}")

    # Extract/load embeddings
    emb_path = os.path.join(OUT_DIR, "embeddings.npy")
    if os.path.exists(emb_path):
        print(" Found embeddings.npy → Loading...")
        X = np.load(emb_path)
    else:
        print(" Extracting ResNet50 embeddings...")
        X = extract_embeddings(image_paths, batch_size=BATCH_SIZE)
        np.save(emb_path, X)

    print(" Embeddings shape:", X.shape)

    # Save true labels
    np.save(os.path.join(OUT_DIR, "y_true.npy"), y_true)

    # ======================================================
    # BASELINE 1: KMeans on embeddings
    # ======================================================
    print(" Running Baseline 1: KMeans on embeddings...")
    kmeans_base = KMeans(n_clusters=NUM_CLUSTERS, random_state=RANDOM_SEED, n_init="auto")
    y_pred_kmeans = kmeans_base.fit_predict(X)

    evaluate_clustering(y_true, y_pred_kmeans, method_name="KMeans on embeddings")
    np.save(os.path.join(OUT_DIR, "y_pred_kmeans.npy"), y_pred_kmeans)

    plot_cluster_sizes(
        y_pred_kmeans,
        save_path=os.path.join(OUT_DIR, "cluster_sizes_kmeans.png"),
        title="Cluster Sizes - KMeans on Embeddings"
    )

    for cid in range(NUM_CLUSTERS):
        save_cluster_montage_labeled(
            image_paths, y_true, y_pred_kmeans, cid, idx_to_class,
            save_name=os.path.join(OUT_DIR, f"kmeans_cluster_{cid}_LABELED.png"),
            images_per_row=5, max_images=25
        )

    df_purity_kmeans = generate_cluster_purity_table(y_true, y_pred_kmeans, idx_to_class,
                                                     "KMeans on Embeddings")

    # ======================================================
    # BASELINE 2: Spectral Global Sigma
    # ======================================================
    print(" Running Baseline 2: Spectral (Global Sigma)...")
    W_global, sigma_used = build_global_sigma_affinity(X, knn_k=KNN_GRAPH_K, mode=GLOBAL_SIGMA_MODE)
    print(f" Global sigma used ({GLOBAL_SIGMA_MODE}) = {sigma_used:.4f}")

    y_pred_global, _ = spectral_clustering_from_affinity(W_global, NUM_CLUSTERS, method_tag="global_sigma")
    evaluate_clustering(y_true, y_pred_global, method_name="Spectral (Global Sigma)")

    np.save(os.path.join(OUT_DIR, "y_pred_global_sigma.npy"), y_pred_global)
    np.save(os.path.join(OUT_DIR, "affinity_global_sigma.npy"), W_global)

    plot_cluster_sizes(
        y_pred_global,
        save_path=os.path.join(OUT_DIR, "cluster_sizes_global_sigma.png"),
        title="Cluster Sizes - Spectral (Global Sigma)"
    )

    for cid in range(NUM_CLUSTERS):
        save_cluster_montage_labeled(
            image_paths, y_true, y_pred_global, cid, idx_to_class,
            save_name=os.path.join(OUT_DIR, f"global_sigma_cluster_{cid}_LABELED.png"),
            images_per_row=5, max_images=25
        )

    df_purity_global = generate_cluster_purity_table(y_true, y_pred_global, idx_to_class,
                                                     "Spectral (Global Sigma)")

    # ======================================================
    # YOUR METHOD: Self-Tuning Spectral
    # ======================================================
    print(" Running Your Method: Spectral (Self-Tuning Density Adaptive)...")
    W_self = build_self_tuning_affinity(X, knn_k=KNN_GRAPH_K, local_scale_k=LOCAL_SCALE_K)

    y_pred_self, _ = spectral_clustering_from_affinity(W_self, NUM_CLUSTERS, method_tag="self_tuning")
    evaluate_clustering(y_true, y_pred_self, method_name="Spectral (Self-Tuning Density Adaptive)")

    np.save(os.path.join(OUT_DIR, "y_pred_self_tuning.npy"), y_pred_self)
    np.save(os.path.join(OUT_DIR, "affinity_self_tuning.npy"), W_self)

    plot_cluster_sizes(
        y_pred_self,
        save_path=os.path.join(OUT_DIR, "cluster_sizes_self_tuning.png"),
        title="Cluster Sizes - Spectral (Self-Tuning)"
    )

    for cid in range(NUM_CLUSTERS):
        save_cluster_montage_labeled(
            image_paths, y_true, y_pred_self, cid, idx_to_class,
            save_name=os.path.join(OUT_DIR, f"self_tuning_cluster_{cid}_LABELED.png"),
            images_per_row=5, max_images=25
        )

    df_purity_self = generate_cluster_purity_table(y_true, y_pred_self, idx_to_class,
                                                   "Spectral (Self-Tuning)")

    # ======================================================
    #  FINAL: PRINT + SAVE PURITY TABLE
    # ======================================================
    df_purity_all = pd.concat([df_purity_kmeans, df_purity_global, df_purity_self], ignore_index=True)
    df_purity_all = df_purity_all.sort_values(by=["Method", "Cluster_ID"]).reset_index(drop=True)

    print("\n================ CLUSTER PURITY TABLE ================\n")
    print(df_purity_all.to_string(index=False))

    purity_csv_path = os.path.join(OUT_DIR, "cluster_purity_table.csv")
    df_purity_all.to_csv(purity_csv_path, index=False)
    print(f"\n Purity table saved as: {purity_csv_path}")

    print("\n DONE  All outputs saved inside:", os.path.abspath(OUT_DIR))
    print(" Open images like: self_tuning_cluster_0_LABELED.png to understand clusters visually.\n")


if __name__ == "__main__":
    main()
