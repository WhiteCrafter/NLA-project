import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AffinityPropagation
from mylib import dbscan, derive_medoids


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster rainfall maps and visualize results.")
    parser.add_argument(
        "--data",
        default="rainfall_data3.npy",
        help="Path to .npy file containing 'maps' and 'timestamps' keys.",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=6,
        help="Maximum number of columns in the overview grid.",
    )
    parser.add_argument(
        "--norm",
        default="fro",
        choices=["fro", "1", "2"],
        help="Matrix norm used for pairwise distances.",
    )
    parser.add_argument(
        "--algo",
        choices=["ap", "dbscan"],
        default="ap",
        help="Clustering algorithm to use: affinity propagation (ap) or DBSCAN (dbscan).",
    )
    parser.add_argument("--eps", type=float, default=45, help="DBSCAN: radius for neighborhood.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="DBSCAN: minimum samples to form a dense region.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Generate HTML outputs without auto-opening them in the browser.",
    )
    return parser.parse_args()


def frob(A, B, ord_value):
    return np.linalg.norm(A - B, ord=ord_value)


def cluster_members_sorted(labels, dist_matrix, cluster_id, center_idx):
    """Return all indices in the cluster, center first, then by distance to center."""
    members = np.where(labels == cluster_id)[0].tolist()
    others = [idx for idx in members if idx != center_idx]
    others_sorted = sorted(others, key=lambda idx: dist_matrix[center_idx, idx])
    return [center_idx] + others_sorted


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # ============================================================
    # LOAD DATASET
    # ============================================================
    loaded = np.load(data_path, allow_pickle=True).item()
    matrices = loaded["maps"]
    timestamps = loaded["timestamps"].astype("datetime64[s]").tolist()
    N, H, W = matrices.shape

    # ============================================================
    # FROBENIUS DISTANCE MATRIX
    # ============================================================
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = frob(matrices[i], matrices[j], args.norm)

    # ============================================================
    # RUN AFFINITY PROPAGATION
    # ============================================================
    if args.algo == "ap":
        similarity = -(dist_matrix ** 2)  # AP requires negative squared distance
        ap = AffinityPropagation(affinity="precomputed", random_state=0)
        ap.fit(similarity)
        labels = ap.labels_
        centers = ap.cluster_centers_indices_
        cluster_ids = list(range(len(centers)))
    else:
        labels = dbscan(dist_matrix, eps=45, min_samples=5)
        medoids = derive_medoids(dist_matrix, labels)
        cluster_ids = list(sorted(medoids.keys()))
        centers = np.array([medoids[cid] for cid in cluster_ids], dtype=int)

    print(f"Algorithm: {args.algo.upper()} | Number of clusters found: {len(centers)}")
    print("Cluster centers:", centers)

    # ============================================================
    # SHOW CLOSEST MEMBERS PER CLUSTER
    # ============================================================
    print("\nCluster memberships (center + all maps):")
    for cluster_id, center_idx in zip(cluster_ids, centers):
        ordered = cluster_members_sorted(labels, dist_matrix, cluster_id, center_idx)
        ordered_str = ", ".join(f"{i} ({timestamps[i].date()})" for i in ordered)
        print(
            f"Cluster {cluster_id} | center {center_idx} ({timestamps[center_idx].date()}) | members: {ordered_str}"
        )

    # ============================================================
    # VISUALIZE ALL MAPS (PLOTLY)
    # ============================================================
    max_cols = max(1, args.max_cols)
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
        title="All rainfall maps",
        height=max(400, rows * 250),
        width=cols * 250,
    )
    auto_open = not args.no_open
    sample_fig.write_html("sample_maps.html", auto_open=auto_open)
    print(
        f"{'Opened' if auto_open else 'Wrote'} sample map grid in browser (saved as sample_maps.html)"
    )

    # ============================================================
    # VISUALIZE CLUSTER CENTERS (PLOTLY)
    # ============================================================
    num_clusters = len(centers)
    cluster_members = [
        cluster_members_sorted(labels, dist_matrix, cluster_id, center_idx)
        for cluster_id, center_idx in zip(cluster_ids, centers)
    ]
    max_cluster_size = max(len(members) for members in cluster_members)

    cluster_titles = []
    for row_idx, members in enumerate(cluster_members):
        center_idx = centers[row_idx]
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
        title="Cluster centers and all cluster members (rows = clusters)",
        height=max(300, num_clusters * 300),
        width=max_cluster_size * 300,
    )
    cluster_fig.write_html("cluster_centers.html", auto_open=auto_open)
    print(
        f"{'Opened' if auto_open else 'Wrote'} cluster center grid in browser (saved as cluster_centers.html)"
    )


if __name__ == "__main__":
    main()
r_fig.write_html("cluster_centers.html", auto_open=True)
print("Opened cluster center grid in browser (saved as cluster_centers.html)")
