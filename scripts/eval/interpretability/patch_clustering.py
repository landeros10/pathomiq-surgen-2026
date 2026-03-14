"""UMAP + HDBSCAN patch clustering functions — Phase 7 Step 5.

Clusters top-k patch embeddings from correctly-classified MSI slides.
Extracts actual CZI image patches. SSH orchestration uses utils/gcp_utils.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def run_umap_hdbscan(
    embeddings,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 5,
    random_state: int = 42,
):
    """Run UMAP dimensionality reduction followed by HDBSCAN clustering.

    Args:
        embeddings:       (N, D) float array of patch embeddings.
        n_neighbors:      UMAP n_neighbors parameter.
        min_dist:         UMAP min_dist parameter.
        min_cluster_size: HDBSCAN min_cluster_size parameter.
        random_state:     Random seed for reproducibility.

    Returns:
        dict with keys:
            umap_coords   : (N, 2) float array
            cluster_labels: (N,) int array (-1 = noise)
            n_clusters    : int
    """
    import numpy as np
    try:
        import umap
        import hdbscan
    except ImportError:
        raise ImportError(
            "umap-learn and hdbscan are required. Install with: "
            "pip install umap-learn hdbscan"
        )

    reducer    = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=random_state)
    umap_coords = reducer.fit_transform(embeddings)

    clusterer     = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(umap_coords)
    n_clusters    = int((cluster_labels >= 0).sum() > 0 and cluster_labels.max() + 1)

    return {
        "umap_coords":    umap_coords,
        "cluster_labels": cluster_labels,
        "n_clusters":     n_clusters,
    }


def build_contact_sheet(
    patch_images: list,
    n_cols: int = 4,
    patch_size: int = 224,
):
    """Assemble a grid of patch images into a contact sheet.

    Args:
        patch_images: list of (H, W, 3) uint8 arrays.
        n_cols:       Number of columns in the grid.
        patch_size:   Resize each patch to this square size.

    Returns:
        (H_total, W_total, 3) uint8 numpy array.
    """
    import numpy as np

    if not patch_images:
        return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

    n_rows = (len(patch_images) + n_cols - 1) // n_cols
    canvas = np.zeros((n_rows * patch_size, n_cols * patch_size, 3), dtype=np.uint8)

    for i, img in enumerate(patch_images):
        r, c = divmod(i, n_cols)
        canvas[r * patch_size:(r + 1) * patch_size,
               c * patch_size:(c + 1) * patch_size] = img

    return canvas
