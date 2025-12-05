import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AffinityPropagation

import mylib

# ============================================================
# SETTINGS
# ============================================================

N = 100           # number of rainfall maps
H, W = 25, 25     # grid size
ALGO = "dbscan"       # choose "ap" or "dbscan"
DBSCAN_EPS = 45
DBSCAN_MIN_SAMPLES = 4

# ============================================================
# LOAD DATASET
# ============================================================

loaded = np.load("./rainfall_data3.npy", allow_pickle=True).item()
matrices = loaded["maps"]
timestamps = loaded["timestamps"].astype("datetime64[s]").tolist()
N, H, W = matrices.shape

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
# CLUSTERING HELPERS
# ============================================================

def cluster_members_sorted(labels, cluster_id, center_idx):
    """Return all indices in the cluster, center first, then by distance to center."""
    members = np.where(labels == cluster_id)[0].tolist()
    others = [idx for idx in members if idx != center_idx]
    others_sorted = sorted(others, key=lambda idx: dist_matrix[center_idx, idx])
    return [center_idx] + others_sorted


def derive_medoids(labels):
    """For each cluster (excluding noise), pick member with min total distance as center."""
    centers = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue  # noise cluster
        members = np.where(labels == cluster_id)[0]
        sub = dist_matrix[np.ix_(members, members)]
        centers[cluster_id] = members[np.argmin(sub.sum(axis=1))]
    return centers


def run_dbscan(eps, min_samples):
    # Use 40th percentile of pairwise distances as a reasonable default if none provided.
    if eps is None:
        upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        eps = float(np.percentile(upper, 40))
    labels = mylib.dbscan(dist_matrix, eps=eps, min_samples=min_samples)
    centers = mylib.derive_medoids(dist_matrix, labels)
    return labels, centers, eps


def run_affinity():
    ap = AffinityPropagation(affinity="precomputed", random_state=0)
    ap.fit(similarity)
    labels = ap.labels_
    centers = {cluster_id: center_idx for cluster_id, center_idx in enumerate(ap.cluster_centers_indices_)}
    return labels, centers


# ============================================================
# RUN CHOSEN CLUSTERING
# ============================================================

if ALGO == "dbscan":
    labels, centers_map, eps_used = run_dbscan(DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    algo_name = f"DBSCAN (eps={eps_used:.3f}, min_samples={DBSCAN_MIN_SAMPLES})"
else:
    labels, centers_map = run_affinity()
    algo_name = "Affinity Propagation"

cluster_ids = [cid for cid in sorted(set(labels)) if cid != -1]
noise_points = int(np.sum(labels == -1))

print(f"Algorithm: {algo_name}")
print("Number of clusters found:", len(cluster_ids))
if noise_points:
    print("Noise points (label = -1):", noise_points)
print("Cluster centers:", [centers_map[cid] for cid in cluster_ids])

# ============================================================
# SHOW CLOSEST MEMBERS PER CLUSTER
# ============================================================

print("\nCluster memberships (center + all maps):")
for cluster_id in cluster_ids:
    center_idx = centers_map[cluster_id]
    ordered = cluster_members_sorted(labels, cluster_id, center_idx)
    ordered_str = ", ".join(f"{i} ({timestamps[i].date()})" for i in ordered)
    print(f"Cluster {cluster_id} | center {center_idx} ({timestamps[center_idx].date()}) | members: {ordered_str}")
if noise_points:
    noise_idxs = np.where(labels == -1)[0]
    noise_str = ", ".join(f"{i} ({timestamps[i].date()})" for i in noise_idxs)
    print(f"\nNoise (unclustered): {noise_str}")

# ============================================================
# VISUALIZE ALL MAPS (PLOTLY)
# ============================================================

max_cols = 6  # limit width; adjust as needed
cols = min(max_cols, N)
rows = int(np.ceil(N / cols))

all_titles = [f"{timestamps[i].date()}" for i in range(N)]
sample_fig = make_subplots(rows=rows, cols=cols, subplot_titles=all_titles)

for i in range(N):
    row = (i // cols) + 1
    col = (i % cols) + 1
    sample_fig.add_trace(
        go.Heatmap(
            z=matrices[i],
            colorscale="Blues",
            colorbar={"title": "Rain"},
            showscale=(i == N - 1),  # single colorbar on last plot
        ),
        row=row,
        col=col,
    )

sample_fig.update_layout(
    title=f"All rainfall maps ({algo_name})",
    height=max(400, rows * 250),
    width=cols * 250,
)
sample_fig.write_html("sample_maps.html", auto_open=True)
print("Opened sample map grid in browser (saved as sample_maps.html)")

# ============================================================
# VISUALIZE CLUSTER CENTERS (PLOTLY)
# ============================================================

num_clusters = len(cluster_ids)

if num_clusters == 0:
    print("No clusters to visualize (all points treated as noise).")
else:
    cluster_members = [
        cluster_members_sorted(labels, cluster_id, centers_map[cluster_id])
        for cluster_id in cluster_ids
    ]
    max_cluster_size = max(len(members) for members in cluster_members)

    cluster_titles = []
    for row_idx, members in enumerate(cluster_members):
        center_idx = centers_map[cluster_ids[row_idx]]
        for col in range(max_cluster_size):
            if col < len(members):
                idx = members[col]
                title = "Center" if idx == center_idx else f"{timestamps[idx].date()}"
            else:
                title = ""
            cluster_titles.append(title)

    cluster_fig = make_subplots(
        rows=num_clusters,
        cols=max_cluster_size,
        subplot_titles=cluster_titles,
    )

    for row_idx, members in enumerate(cluster_members):
        for col_idx, idx in enumerate(members):
            cluster_fig.add_trace(
                go.Heatmap(z=matrices[idx], colorscale="Blues", showscale=False),
                row=row_idx + 1,
                col=col_idx + 1,
            )

    cluster_fig.update_layout(
        title=f"Cluster centers and all cluster members ({algo_name})",
        height=max(300, num_clusters * 300),
        width=max_cluster_size * 300,
    )
    cluster_fig.write_html("cluster_centers.html", auto_open=True)
    print("Opened cluster center grid in browser (saved as cluster_centers.html)")
