#  Clustering Notes & Documents Based on Tags

Using semantic tag embeddings + DBSCAN / AffinityPropagation

## Why cluster notes?

If you have a growing Obsidian or Markdown library, tags help manually, but clustering reveals hidden structure:

Discover related notes even if they don’t share identical tags

Identify topic groups in your knowledge base

Detect hybrid/crossover notes

Cleanly separate math / coding / art / fantasy / gaming topics

This document explains and demonstrates how to cluster notes using semantic embeddings of tags, using DBSCAN and AffinityPropagation.

##  Semantic Embeddings for Tags

There are two ways to encode tags numerically:

 1. Boolean tag vector (0/1)

Not good — treats unrelated tags as equally distant.

 2. Semantic tag embeddings

We use 6 conceptual axes:

[math, coding, art, fantasy, gaming, hobby]


Example:

Tag	Embedding Meaning
NLA	Mostly math, slightly coding
design	Art + visual theory
dnd	Fantasy + gaming + hobby
game-dev	Coding + gaming

A document vector is the average of all tag vectors.

##  Norms for Distance

We test three common norms:

def norm2(a):   # Euclidean norm
def norm1(a):   # Manhattan norm
def norminf(a): # Infinity norm


L2 — smooth distances

L1 — good when documents mix tags

L∞ — only cares about the dominant semantic axis

##  Density-Based Clustering (DBSCAN)

DBSCAN groups dense areas and marks unrelated documents as noise (-1).

Parameters:

eps — max neighbor distance

min_samples — how dense a region must be

Advantages:

Don’t need number of clusters

Handles weird outliers

Good for hybrid-tag documents

##  Distance Matrix + DBSCAN Implementation
def pairwise_distances(points, norm):
    n = len(points)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = norm(points[i] - points[j])
            d[i, j] = d[j, i] = dist
    return d


def dbscan(dist_matrix, eps, min_samples):
    n = dist_matrix.shape[0]
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dist_matrix[i] <= eps)[0]

        if neighbors.size < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = [p for p in neighbors if p != i]

        while seeds:
            point = seeds.pop()
            if not visited[point]:
                visited[point] = True
                point_neighbors = np.where(dist_matrix[point] <= eps)[0]
                if point_neighbors.size >= min_samples:
                    for nb in point_neighbors:
                        if nb not in seeds:
                            seeds.append(nb)

            if labels[point] == -1 or labels[point] != cluster_id:
                labels[point] = cluster_id

        cluster_id += 1

    return labels

##  Tag Embeddings
tag_vectors = {
    "math":            np.array([1.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "NLA":             np.array([0.90, 0.20, 0.00, 0.00, 0.00, 0.00]),
    "calculus":        np.array([0.85, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "analysis":        np.array([0.80, 0.00, 0.00, 0.00, 0.00, 0.00]),

    "coding":          np.array([0.00, 1.00, 0.00, 0.00, 0.00, 0.00]),
    "python":          np.array([0.00, 0.90, 0.00, 0.00, 0.00, 0.00]),
    "cpp":             np.array([0.00, 0.80, 0.00, 0.00, 0.00, 0.00]),
    "algorithms":      np.array([0.10, 0.85, 0.00, 0.00, 0.00, 0.00]),
    "project":         np.array([0.00, 0.70, 0.00, 0.00, 0.00, 0.30]),

    "art":             np.array([0.00, 0.00, 1.00, 0.00, 0.00, 0.00]),
    "design":          np.array([0.00, 0.00, 0.85, 0.00, 0.00, 0.00]),
    "drawing":         np.array([0.00, 0.00, 0.90, 0.00, 0.00, 0.30]),

    "fantasy":         np.array([0.00, 0.00, 0.00, 1.00, 0.00, 0.00]),
    "worldbuilding":   np.array([0.00, 0.00, 0.00, 0.90, 0.00, 0.30]),
    "lore":            np.array([0.00, 0.00, 0.00, 0.85, 0.00, 0.00]),
    "dnd":             np.array([0.00, 0.00, 0.00, 0.80, 0.30, 0.30]),

    "gaming":          np.array([0.00, 0.00, 0.00, 0.00, 1.00, 0.00]),
    "game-dev":        np.array([0.00, 0.50, 0.00, 0.00, 0.90, 0.00]),
    "helldivers":      np.array([0.00, 0.00, 0.00, 0.00, 0.95, 0.00]),

    "hobby":           np.array([0.00, 0.00, 0.00, 0.00, 0.00, 1.00]),
    "journal":         np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.90]),
    "recipes":         np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.80])
}

##  Sample Documents

Your full synthetic dataset (math, coding, art, fantasy, gaming, hobby, and mixes) remains unchanged.
I am not repeating it here for brevity, but it plugs in exactly as-is.

##  Document Embedding Function
def document_vector(tags):
    vecs = [tag_vectors[tag] for tag in tags if tag in tag_vectors]
    if not vecs:
        return np.zeros(6)
    return np.mean(vecs, axis=0)

##  Running DBSCAN
doc_embeddings = {name: document_vector(tags) for name, tags in documents.items()}
names = list(doc_embeddings.keys())
array = np.vstack([v for v in doc_embeddings.values()])

dist_matrix = pairwise_distances(array, norm2)

EPS = np.median(dist_matrix[np.triu_indices_from(dist_matrix, 1)])
MIN_SAMPLES = 3

labels = dbscan(dist_matrix, eps=EPS, min_samples=MIN_SAMPLES)
clusters = summarize_clusters(labels, names)

##  Example Output
DBSCAN eps=0.412, min_samples=3

Cluster 0: ['build_guide_slayer', 'game_review_hd2']
Cluster 1: ['la_notes', 'probability_notes', 'calc_homework', 'analysis_summary']
Cluster 2: ['world_map_art', 'character_concepts', 'magic_creature_design']
Cluster 3: [large coding-math-fantasy mixed group...]
Cluster -1 (Noise): [weird crossover docs]


Noise = documents too unique or cross-topic for DBSCAN.

## Affinity Propagation (AP)

Alternative clustering with no eps needed:

from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation(damping=0.9, random_state=0)
labels = ap.fit_predict(array)


AP:

Automatically decides number of clusters

No noise cluster

Picks “exemplar” documents as cluster centers

## 🖥 CLI Tool Version

Supports:

--method dbscan
--method ap
--csv FILE
--norm l1|l2|linf
--eps FLOAT
--min-samples INT
--verbose


Example:

python tag_clustering.py --method ap
python tag_clustering.py --method dbscan --eps 0.32 --norm l1

## Summary

This system provides a full workflow:

Semantic tag embeddings

Multiple distance norms

DBSCAN for density-based clusters

AffinityPropagation for exemplar-based clusters

Outlier detection

CLI-ready code

CSV-driven dataset support

You can now automatically analyze and group your entire knowledge base.