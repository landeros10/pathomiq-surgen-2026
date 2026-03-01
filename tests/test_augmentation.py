"""Augmentation contract tests.

Any transform applied to patch embeddings must pass every test in this file.

INTERFACE
---------
All transforms have the signature:
    transform(features, coords) -> (features, coords)

    features : (N, 1024) float32  — patch embeddings
    coords   : (N, 2)    int64    — (x, y) patch coordinates
    returns  : (M, 1024) float32, (M, 2) int64
               M can differ from N (subsampling/consolidation), but D must == 1024.

HOW TO ADD A NEW TRANSFORM
---------------------------
1. Import or define your transform below.
2. Append it to TRANSFORMS_UNDER_TEST with a descriptive id= string.
3. Run: pytest tests/test_augmentation.py -v
   All universal invariants are checked automatically.
4. If your transform changes patch count (subsampling, consolidation), also
   add it to SUBSAMPLING_TRANSFORMS so the M <= N invariant is verified.
5. If your transform reorders/flips the coordinate space without dropping
   patches, add it to COORD_TRANSFORMS so the feature-preservation invariant
   is verified.
"""

import pytest
import torch


# ── Placeholder identity transforms (stand-ins until real augmentations exist) ─

def identity(features, coords):
    """No-op. Used to verify the contract machinery works."""
    return features, coords


def coord_shuffle(features, coords):
    """Randomly reorder patches by shuffling both features and coords in lock-step."""
    idx = torch.randperm(features.shape[0])
    return features[idx], coords[idx]


def coord_flip_x(features, coords):
    """Mirror patch positions along the x-axis (coords[:, 0] = max_x - x).
    Feature vectors are untouched; only spatial layout changes.
    """
    flipped = coords.clone()
    flipped[:, 0] = coords[:, 0].max() - coords[:, 0]
    return features, flipped


def coord_flip_y(features, coords):
    """Mirror patch positions along the y-axis (coords[:, 1] = max_y - y)."""
    flipped = coords.clone()
    flipped[:, 1] = coords[:, 1].max() - coords[:, 1]
    return features, flipped


def patch_subsample_half(features, coords):
    """Keep a random 50% of patches (minimum 1)."""
    n = max(1, features.shape[0] // 2)
    idx = torch.randperm(features.shape[0])[:n]
    return features[idx], coords[idx]


# ── Transform registries ────────────────────────────────────────────────────────

# Every transform here is checked against ALL universal invariants.
TRANSFORMS_UNDER_TEST = [
    pytest.param(identity,           id="identity"),
    pytest.param(coord_shuffle,      id="coord_shuffle"),
    pytest.param(coord_flip_x,       id="coord_flip_x"),
    pytest.param(coord_flip_y,       id="coord_flip_y"),
    pytest.param(patch_subsample_half, id="patch_subsample_half"),
]

# Transforms that reduce patch count — verified: M <= N.
SUBSAMPLING_TRANSFORMS = [
    pytest.param(patch_subsample_half, id="patch_subsample_half"),
]

# Transforms that reorder/flip coords but keep all patches — verified:
# feature values are exactly preserved (only coords allowed to change).
COORD_TRANSFORMS = [
    pytest.param(coord_shuffle,  id="coord_shuffle"),
    pytest.param(coord_flip_x,   id="coord_flip_x"),
    pytest.param(coord_flip_y,   id="coord_flip_y"),
]

MAX_REASONABLE_ABS = 100.0  # UNI embeddings are ~N(0,1); >100 indicates corruption.


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(params=[1, 50, 300], ids=["n=1", "n=50", "n=300"])
def embedding_batch(request):
    """(features, coords, label) with varying patch counts."""
    n = request.param
    torch.manual_seed(0)
    features = torch.randn(n, 1024, dtype=torch.float32)
    coords   = torch.randint(0, 10000, (n, 2), dtype=torch.int64)
    label    = torch.tensor(1.0)
    return features, coords, label


# ── Universal invariants (every transform must pass these) ─────────────────────

@pytest.mark.parametrize("transform", TRANSFORMS_UNDER_TEST)
class TestUniversalInvariants:

    def test_no_nan_in_features(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        out_f, _ = transform(features, coords)
        assert not torch.isnan(out_f).any(), "NaN introduced in features"

    def test_no_inf_in_features(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        out_f, _ = transform(features, coords)
        assert not torch.isinf(out_f).any(), "Inf introduced in features"

    def test_embedding_dim_preserved(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        out_f, _ = transform(features, coords)
        assert out_f.shape[1] == 1024, (
            f"Embedding dim changed: expected 1024, got {out_f.shape[1]}"
        )

    def test_output_is_nonempty(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        out_f, out_c = transform(features, coords)
        assert out_f.shape[0] >= 1, "Transform returned zero patches"
        assert out_c.shape[0] == out_f.shape[0], (
            "features and coords have different patch counts after transform"
        )

    def test_dtype_float32(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        out_f, _ = transform(features, coords)
        assert out_f.dtype == torch.float32, (
            f"dtype changed: expected float32, got {out_f.dtype}"
        )

    def test_value_range_reasonable(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        out_f, _ = transform(features, coords)
        max_abs = out_f.abs().max().item()
        assert max_abs < MAX_REASONABLE_ABS, (
            f"Feature values out of expected range: max abs = {max_abs:.2f} "
            f"(threshold {MAX_REASONABLE_ABS}). Check for missing normalization "
            "or incorrect scale factor."
        )

    def test_label_not_modified(self, transform, embedding_batch):
        """Transforms must not touch the label — they only receive features+coords."""
        features, coords, label = embedding_batch
        original_label = label.clone()
        transform(features, coords)   # label is intentionally not passed in
        assert label.equal(original_label), "Label tensor was modified as a side effect"


# ── Subsampling-specific invariants ────────────────────────────────────────────

@pytest.mark.parametrize("transform", SUBSAMPLING_TRANSFORMS)
class TestSubsamplingInvariants:

    def test_patch_count_reduced(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        if features.shape[0] == 1:
            pytest.skip("Cannot subsample below 1 patch — skip for n=1")
        out_f, _ = transform(features, coords)
        assert out_f.shape[0] < features.shape[0], (
            "Subsampling transform did not reduce patch count"
        )

    def test_subsampled_patches_are_subset(self, transform, embedding_batch):
        """Every output patch must be an exact row from the input — no invented patches."""
        features, coords, _ = embedding_batch
        out_f, _ = transform(features, coords)
        for row in out_f:
            match = (features == row.unsqueeze(0)).all(dim=1).any()
            assert match, "Output contains a patch not present in the input"


# ── Coordinate transform invariants ────────────────────────────────────────────

@pytest.mark.parametrize("transform", COORD_TRANSFORMS)
class TestCoordTransformInvariants:

    def test_patch_count_unchanged(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        out_f, out_c = transform(features, coords)
        assert out_f.shape[0] == features.shape[0], (
            "Coord transform must not add or remove patches"
        )

    def test_feature_values_unchanged(self, transform, embedding_batch):
        """Coord transforms may reorder or flip coords but must not alter feature values.
        Checks that the multiset of feature rows is preserved (order may differ).
        """
        features, coords, _ = embedding_batch
        out_f, _ = transform(features, coords)
        # Sort both by first column for order-independent comparison
        in_sorted  = features[features[:, 0].argsort()]
        out_sorted = out_f[out_f[:, 0].argsort()]
        assert torch.allclose(in_sorted, out_sorted), (
            "Feature values changed after a coord-only transform"
        )

    def test_coords_are_valid_int64(self, transform, embedding_batch):
        features, coords, _ = embedding_batch
        _, out_c = transform(features, coords)
        assert out_c.dtype == torch.int64, (
            f"Coord dtype changed: expected int64, got {out_c.dtype}"
        )
        assert (out_c >= 0).all(), "Negative coordinate values after transform"
