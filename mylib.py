import numpy as np


def _region_query(dist_matrix, idx, eps):
    """Return neighbors of idx (including idx itself) within eps."""
    return np.where(dist_matrix[idx] <= eps)[0]


def dbscan(dist_matrix, eps, min_samples):
    n = dist_matrix.shape[0]
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = _region_query(dist_matrix, i, eps)
        if neighbors.size < min_samples:
            labels[i] = -1  # noise for now; could become border later
            continue

        # Start new cluster
        labels[i] = cluster_id
        seeds = list(neighbors[neighbors != i])  # exclude self to avoid duplicate work

        # Expand cluster
        while seeds:
            point = seeds.pop()
            if not visited[point]:
                visited[point] = True
                point_neighbors = _region_query(dist_matrix, point, eps)
                if point_neighbors.size >= min_samples:
                    # core point: add its neighbors to be processed
                    for nb in point_neighbors:
                        if nb not in seeds:
                            seeds.append(nb)
            if labels[point] == -1:
                labels[point] = cluster_id  # border point
            # if already assigned to this cluster, nothing to do
            elif labels[point] != cluster_id:
                labels[point] = cluster_id

        cluster_id += 1

    return labels


def derive_medoids(dist_matrix, labels):
    """For each non-noise cluster, return medoid indices."""
    centers = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        members = np.where(labels == cluster_id)[0]
        sub = dist_matrix[np.ix_(members, members)]
        centers[cluster_id] = members[np.argmin(sub.sum(axis=1))]
    return centers
