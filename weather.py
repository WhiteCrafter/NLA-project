import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

# ============================================================
# SETTINGS
# ============================================================

N = 100           # number of rainfall maps
H, W = 25, 25     # grid size

# ============================================================
# LOAD DATASET
# ============================================================

loaded = np.load("CP_NLA/rainfall_data.npy", allow_pickle=True).item()
matrices = loaded["maps"]
timestamps = loaded["timestamps"].astype("datetime64[s]").tolist()
N, H, W = matrices.shape

# ============================================================
# VISUALIZE SAMPLE MAPS
# ============================================================

plt.figure(figsize=(12, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(matrices[i], cmap="Blues")
    plt.title(f"{timestamps[i].date()}")
    plt.colorbar()
plt.tight_layout()
plt.show()

# ============================================================
# FROBENIUS DISTANCE MATRIX
# ============================================================

def frob(A, B):
    return np.linalg.norm(A - B, ord='fro')

dist_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        dist_matrix[i, j] = frob(matrices[i], matrices[j])

similarity = -(dist_matrix ** 2)  # AP requires negative squared distance

# ============================================================
# RUN AFFINITY PROPAGATION
# ============================================================

ap = AffinityPropagation(affinity="precomputed", random_state=0)
ap.fit(similarity)

labels = ap.labels_
centers = ap.cluster_centers_indices_

print("Number of clusters found:", len(centers))
print("Cluster centers:", centers)

# ============================================================
# SHOW CLOSEST MEMBERS PER CLUSTER
# ============================================================

def closest_indices_to_center(dist_matrix, center_idx, top_k=5):
    """Return indices of the closest maps to a given center (including itself)."""
    return np.argsort(dist_matrix[center_idx])[:top_k]

top_k = 5
print("\nCluster memberships (center + closest maps):")
for cluster_id, center_idx in enumerate(centers):
    nearest = closest_indices_to_center(dist_matrix, center_idx, top_k=top_k)
    nearest_str = ", ".join(f"{i} ({timestamps[i].date()})" for i in nearest)
    print(f"Cluster {cluster_id} | center {center_idx} ({timestamps[center_idx].date()}) | closest {top_k}: {nearest_str}")

# ============================================================
# VISUALIZE CLUSTER CENTERS
# ============================================================

num_clusters = len(centers)
cols = top_k  # show center + nearest maps in each row
rows = num_clusters
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
axes = np.atleast_2d(axes)

for row, center_idx in enumerate(centers):
    nearest = closest_indices_to_center(dist_matrix, center_idx, top_k=top_k)
    for col, idx in enumerate(nearest):
        ax = axes[row, col]
        ax.imshow(matrices[idx], cmap="Blues")
        title = "Center" if idx == center_idx else f"{timestamps[idx].date()}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

plt.suptitle("Cluster centers and closest maps (rows = clusters)")
plt.tight_layout()
plt.show()
