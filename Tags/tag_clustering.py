#importing libraries
import numpy as np
import csv
import sys

# ----------------------------------------------------------
# Norms
# ----------------------------------------------------------
def norm2(a): return np.sqrt((a ** 2).sum(axis=-1))
def norm1(a): return np.abs(a).sum(axis=-1)
def norminf(a): return np.max(np.abs(a), axis=-1)

NORM_MAP = {
    "l1": norm1,
    "l2": norm2,
    "linf": norminf
}

# ----------------------------------------------------------
# DBSCAN implementation (pure NumPy)
# ----------------------------------------------------------
def dbscan(points, eps, min_samples, norm):
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    labels = np.full(n, -1)
    cluster_id = 0

    def neighbors(idx):
        d = norm(points - points[idx])
        return np.where(d <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neigh = neighbors(i)
        if len(neigh) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = list(neigh)

            while queue:
                j = queue.pop()
                if not visited[j]:
                    visited[j] = True
                    neigh_j = neighbors(j)
                    if len(neigh_j) >= min_samples:
                        queue.extend(neigh_j)
                if labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

    return labels


# ----------------------------------------------------------
# Tag vectors
# ----------------------------------------------------------
tag_vectors = {
    "math":            np.array([1,0,0,0,0,0]),
    "NLA":             np.array([0.9,0.2,0,0,0,0]),
    "calculus":        np.array([0.85,0,0,0,0,0]),
    "analysis":        np.array([0.8,0,0,0,0,0]),
    "coding":          np.array([0,1,0,0,0,0]),
    "python":          np.array([0,0.9,0,0,0,0]),
    "cpp":             np.array([0,0.8,0,0,0,0]),
    "algorithms":      np.array([0.1,0.85,0,0,0,0]),
    "project":         np.array([0,0.7,0,0,0,0.3]),
    "art":             np.array([0,0,1,0,0,0]),
    "design":          np.array([0,0,0.85,0,0,0]),
    "drawing":         np.array([0,0,0.9,0,0,0.3]),
    "fantasy":         np.array([0,0,0,1,0,0]),
    "worldbuilding":   np.array([0,0,0,0.9,0,0.3]),
    "lore":            np.array([0,0,0,0.85,0,0]),
    "dnd":             np.array([0,0,0,0.8,0.3,0.3]),
    "gaming":          np.array([0,0,0,0,1,0]),
    "game-dev":        np.array([0,0.5,0,0,0.9,0]),
    "helldivers":      np.array([0,0,0,0,0.95,0]),
    "hobby":           np.array([0,0,0,0,0,1]),
    "journal":         np.array([0,0,0,0,0,0.9]),
    "recipes":         np.array([0,0,0,0,0,0.8])
}


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def load_documents(csv_path, verbose=False):
    documents = {}
    if verbose:
        print(f"[INFO] Loading CSV: {csv_path}")

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                name = row[0]
                tags = [tag.strip() for tag in row[1:] if tag.strip()]
                documents[name] = tags
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found: {csv_path}")
        sys.exit(1)

    return documents


def document_vector(tags):
    vecs = [tag_vectors[t] for t in tags if t in tag_vectors]
    if not vecs:
        return np.zeros(6)
    return np.mean(vecs, axis=0)


# ----------------------------------------------------------
# Argument parser
# ----------------------------------------------------------
def get_args():
    import argparse
    p = argparse.ArgumentParser(description="DBSCAN / AP clustering on document tag vectors")

    p.add_argument("--csv", default="documents.csv")
    p.add_argument("--method", choices=["dbscan", "ap"], default="dbscan")
    p.add_argument("--eps", type=float, default=0.35)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--norm", choices=["l1", "l2", "linf"], default="l2")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    args = get_args()

    if args.verbose:
        print(f"[INFO] Method: {args.method}")
        print(f"[INFO] Norm: {args.norm}")
        if args.method == "dbscan":
            print(f"[INFO] DBSCAN eps={args.eps}, min_samples={args.min_samples}")

    # Load documents
    documents = load_documents(args.csv, verbose=args.verbose)
    doc_embeddings = {name: document_vector(tags) for name, tags in documents.items()}
    array = np.vstack(list(doc_embeddings.values()))
    names_list = list(doc_embeddings.keys())

    # ------------------------------------------------------
    # Execute chosen clustering method
    # ------------------------------------------------------
    if args.method == "dbscan":
        norm_fn = NORM_MAP[args.norm]
        labels = dbscan(array, args.eps, args.min_samples, norm_fn)

    elif args.method == "ap":
        from sklearn.cluster import AffinityPropagation
        ap = AffinityPropagation(damping=0.9, random_state=0)
        labels = ap.fit_predict(array)

    # ------------------------------------------------------
    # Collect clusters
    # ------------------------------------------------------
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    # Print results
    print("\n=== CLUSTERS ===")
    for label, idxs in clusters.items():
        print(f"\nCluster {label}:")
        print({names_list[i] for i in idxs})


if __name__ == "__main__":
    main()
