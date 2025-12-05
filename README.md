# Rainfall Clustering Explorer

## 1. Motivation and overview

This project explores whether simple, distance-based clustering can recover recurring spatial patterns in gridded rainfall fields. The workflow is intentionally lightweight: compute pairwise Frobenius distances between maps, convert them into a similarity matrix, and run Affinity Propagation (AP) to discover exemplar maps that act as cluster centers. Plotly is used to produce web-native, shareable heatmaps for qualitative inspection.

  

Goals:
- Provide a transparent baseline for rainfall regime discovery using only spatial structure.
- Make results inspectable without desktop plotting backends (HTML outputs open in the browser).
- Keep reproducibility straightforward: fixed random seed for clustering, simple `.npy` data I/O, and minimal dependencies.

  

## 2. Data description

### 2.1 Input format
- File: `rainfall_data2.npy` (update `weather.py` if your file name differs).
- Structure: a dict with keys:
  - `"maps"`: shape `(N, H, W)`, floats, rainfall intensity (arbitrary units).
  - `"timestamps"`: shape `(N,)`, datetime-like entries convertible to `datetime64[s]`.

  
## 3. Methodology

### 3.1 Preprocessing

- Data are loaded directly; no normalization is applied beyond what the generator produces.

- Shapes are inferred from the loaded array; `N, H, W` are overwritten from the file.

  

### 3.2 Distance and similarity

- Pairwise dissimilarity: Frobenius norm between grids $\(d_{ij} = || A_i - A_j || \)$

- Converted to similarity for AP: $\( s_{ij} = -d_{ij}^2 \)$ (negative squared distance).

- Complexity: $\(O(N^2 \cdot H \cdot W)\)$ for the dense distance matrix; practical for small $\(N\)$ (hundreds) but not for very large archives without approximation.

  

### 3.3 Clustering with Affinity Propagation

- AP is run with `affinity="precomputed"` on the similarity matrix.

- AP identifies exemplars (cluster centers) without pre-specifying `k`.

- Fixed `random_state=0` for deterministic results.

- Outputs:

  - `labels`: cluster assignment per map.

  - `cluster_centers_indices_`: indices of exemplar maps.

  

### 3.4 Cluster membership ordering

- For each cluster, members are ordered by increasing distance to the center (center first). This gives an intuitive sense of within-cluster variability.

  

## 4. Visualization

All visualizations are Plotly HTML and auto-open in your default browser.

1) **All rainfall maps (`sample_maps.html`)**  

   - Grid layout with up to 6 columns; rows expand as needed.  
   
   - Each subplot: heatmap of one map, titled by its date.  
   
   - Single colorbar shared on the last subplot to reduce clutter.  

   - Purpose: sanity-check the full dataset distribution and spatial patterns.
<img width="686" height="856" alt="image" src="https://github.com/user-attachments/assets/a4e5abb0-d0ee-47fb-b053-7956a9e16678" />


  

2) **Cluster grids (`cluster_centers.html`)**  

   - One row per cluster.  
   
   - Columns span *all* members in that cluster; the center is first.  
   
   - Members are ordered by proximity to the center (closest to farthest).  
   
   - Purpose: assess how representative the exemplar is and how tight/loose the cluster is.

  
<img width="1490" height="838" alt="image" src="https://github.com/user-attachments/assets/b1ace546-aef9-4171-bbd5-38576042a4eb" />


Console output mirrors the cluster composition with timestamps for quick text-based inspection.

  


## 6. Interpreting results

- **Number of clusters**: Driven by AP preferences implicit in the similarity magnitudes; more dispersed data yields more clusters.

- **Cluster centers**: Indices reported correspond to specific maps; visually inspect these in the cluster grid to confirm they are representative.

- **Spread within clusters**: Rows with many columns and visibly diverse shapes suggest AP is grouping broader regimes; consider normalization or alternate similarity metrics if clusters look too loose.

  

## 7. Design choices and rationale

- **Frobenius distance**: Simple, geometry-aware, and scale-sensitive; good for baseline spatial comparisons.

- **Affinity Propagation**: Avoids manual choice of `k`, surfaces exemplars directly; deterministic here via `random_state`.

- **Plotly over Matplotlib**: Browser-native, shareable HTML, no desktop GUI requirement.

- **Full-map displays**: Show every input map to avoid sampling bias; cluster views include all members to make within-cluster diversity explicit.

  

## 8. Limitations and possible extensions

- **Scalability**: Dense $\(N^2\)$ distances limit very large datasets; for larger archives, consider subsampling, sparse k-NN graphs, or approximate neighbors.

- **Metric choice**: Frobenius treats all pixels equally; could experiment with:

    - Normalizing by total rainfall per map.

    - Structural similarity indices.

    - Learned embeddings from autoencoders or CNNs.

- **Clustering alternatives**: k-medoids, hierarchical clustering with cut selection, or HDBSCAN on feature embeddings.

- **Uncertainty and noise**: Current pipeline assumes clean fields; adding noise models or bootstrapping could quantify clustering stability.

- **Temporal context**: Present approach ignores sequence; adding temporal features or seasonal priors could refine grouping.

  
