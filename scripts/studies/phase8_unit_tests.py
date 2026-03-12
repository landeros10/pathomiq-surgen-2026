"""Phase 8 unit tests — MLPRelativePositionBias and MultiMILTransformer.

Standalone script; no embeddings needed. Run from project root:
    python scripts/studies/phase8_unit_tests.py

All 7 tests must pass before running the smoke test.
"""

import sys
import os

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import traceback

from scripts.models.layers import MLPRelativePositionBias
from scripts.models.mil_transformer import MultiMILTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coords(B, N, pixel_scale=2013):
    """Random pixel-space coords (B, N, 2) float."""
    return torch.rand(B, N, 2) * pixel_scale


def _make_embeddings(B, N, D=1024):
    return torch.randn(B, N, D)


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

results = []


def run_test(name, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        results.append((name, True, None))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [{FAIL}] {name}")
        print(f"         {e}")
        if "--verbose" in sys.argv:
            print(tb)
        results.append((name, False, str(e)))


# ---------------------------------------------------------------------------
# Test 1 — MLPRelativePositionBias.forward shape
# ---------------------------------------------------------------------------

def test_rpb_shape():
    B, N, H = 2, 30, 4
    rpb = MLPRelativePositionBias(num_heads=H)
    coords = _make_coords(B, N)
    out = rpb(coords)
    assert out.shape == (B * H, N, N), f"Expected ({B*H},{N},{N}), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# Test 2 — MLPRelativePositionBias gradient flow
# ---------------------------------------------------------------------------

def test_rpb_grad():
    B, N, H = 1, 20, 2
    rpb = MLPRelativePositionBias(num_heads=H)
    coords = _make_coords(B, N)
    out = rpb(coords)
    loss = out.sum()
    loss.backward()
    # Check that at least one gradient is non-zero in rpb.mlp weights
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in rpb.mlp.parameters()
    )
    assert has_grad, "No gradient reached rpb.mlp weights"


# ---------------------------------------------------------------------------
# Test 3 — MLPRelativePositionBias normalization stays in [-1, 1]
# ---------------------------------------------------------------------------

def test_rpb_normalization():
    B, N = 3, 50
    rpb = MLPRelativePositionBias(num_heads=2)
    coords = _make_coords(B, N)

    # Replicate the normalization logic from forward()
    rel = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, 2)
    scale = rel.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
    normalized = rel / scale

    assert normalized.abs().max().item() <= 1.0 + 1e-5, (
        f"Normalized values exceed [-1,1]: max={normalized.abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 4 — MLPRelativePositionBias variable N
# ---------------------------------------------------------------------------

def test_rpb_variable_n():
    rpb = MLPRelativePositionBias(num_heads=2)
    for N in (50, 500):
        coords = _make_coords(B=1, N=N)
        out = rpb(coords)
        assert out.shape == (2, N, N), f"N={N}: expected (2,{N},{N}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 5 — MultiMILTransformer(pe="none") no-coords forward
# ---------------------------------------------------------------------------

def test_multi_mil_no_pe():
    model = MultiMILTransformer(
        input_dim=1024, hidden_dim=128, num_layers=1, num_heads=2,
        ffn_dim=256, dropout=0.0, output_classes=3,
        aggregation="attention", attn_variant="joined",
        positional_encoding="none",
    )
    model.eval()
    x = _make_embeddings(1, 100)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3), f"Expected (1,3), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 6 — MultiMILTransformer(pe="mlp_rpb") with coords, grad to rpb.mlp
# ---------------------------------------------------------------------------

def test_multi_mil_mlp_rpb():
    model = MultiMILTransformer(
        input_dim=1024, hidden_dim=128, num_layers=1, num_heads=2,
        ffn_dim=256, dropout=0.0, output_classes=3,
        aggregation="attention", attn_variant="joined",
        positional_encoding="mlp_rpb",
    )
    x = _make_embeddings(1, 100)
    coords = _make_coords(1, 100)
    out = model(x, coords=coords)
    assert out.shape == (1, 3), f"Expected (1,3), got {out.shape}"

    loss = out.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.rpb.mlp.parameters()
    )
    assert has_grad, "No gradient reached model.rpb.mlp weights"


# ---------------------------------------------------------------------------
# Test 7 — Patch dropout reduces N consistently for both x and coords
# ---------------------------------------------------------------------------

def test_patch_dropout():
    """Simulate the patch-dropout logic from train.py and verify consistency."""
    torch.manual_seed(0)
    N = 100
    drop_rate = 0.25
    x = _make_embeddings(1, N)
    coords = _make_coords(1, N)

    # Replicate the dropout logic used in train.py
    if drop_rate > 0.0:
        keep_n = max(1, int(N * (1.0 - drop_rate)))
        idx = torch.randperm(N)[:keep_n]
        idx_sorted = idx.sort().values
        x_drop = x[:, idx_sorted, :]
        coords_drop = coords[:, idx_sorted, :]
    else:
        x_drop = x
        coords_drop = coords

    kept = x_drop.shape[1]
    expected_approx = int(N * (1.0 - drop_rate))

    assert kept == expected_approx, f"Expected {expected_approx} patches, kept {kept}"
    assert kept >= 1, "All patches were dropped"
    assert x_drop.shape[1] == coords_drop.shape[1], (
        f"x and coords patch dim mismatch: {x_drop.shape[1]} vs {coords_drop.shape[1]}"
    )
    assert x_drop.shape[2] == x.shape[2], "Embedding dim changed after dropout"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 8 Unit Tests")
    print("=" * 60)

    run_test("1. MLPRelativePositionBias.forward shape", test_rpb_shape)
    run_test("2. MLPRelativePositionBias grad flow", test_rpb_grad)
    run_test("3. MLPRelativePositionBias normalization", test_rpb_normalization)
    run_test("4. MLPRelativePositionBias variable N", test_rpb_variable_n)
    run_test("5. MultiMILTransformer(pe=none) no-coords forward", test_multi_mil_no_pe)
    run_test("6. MultiMILTransformer(pe=mlp_rpb) with coords + grad", test_multi_mil_mlp_rpb)
    run_test("7. Patch dropout consistency", test_patch_dropout)

    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\nFailed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
