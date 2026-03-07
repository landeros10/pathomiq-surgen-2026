"""Tests for MILTransformer architecture."""
import pytest
import torch

from scripts.models.mil_transformer import MILTransformer


@pytest.fixture
def model():
    return MILTransformer()


class TestArchitecture:

    def test_two_transformer_layers(self, model):
        """nn.TransformerEncoder must hold exactly 2 independent layer copies."""
        assert len(model.transformer.layers) == 2

    def test_layers_have_independent_weights(self, model):
        """The two layers are cloned — they must not be the same object."""
        assert model.transformer.layers[0] is not model.transformer.layers[1]

    def test_projection_input_dim(self, model):
        """Input projection: 1024 → 512."""
        assert model.input_proj[0].in_features == 1024
        assert model.input_proj[0].out_features == 512

    def test_classifier_dim(self, model):
        """Classifier head: 512 → 1 (binary logit)."""
        assert model.classifier[0].in_features == 512
        assert model.classifier[0].out_features == 1


class TestForwardPass:

    def test_output_shape(self, model):
        """(1, N, 1024) input → (1,) scalar logit output."""
        x = torch.randn(1, 50, 1024)
        assert model(x).shape == (1,)

    def test_output_dtype(self, model):
        x = torch.randn(1, 50, 1024)
        assert model(x).dtype == torch.float32

    def test_variable_patch_counts(self, model):
        """Model must handle any number of patches — WSI sizes vary widely."""
        for n_patches in [1, 50, 500]:
            out = model(torch.randn(1, n_patches, 1024))
            assert out.shape == (1,), f"Wrong shape for N={n_patches}"

    def test_output_is_raw_logit(self, model):
        """Output should be an unbounded logit, not a probability in [0, 1]."""
        torch.manual_seed(0)
        outputs = [model(torch.randn(1, 20, 1024)).item() for _ in range(20)]
        assert any(v < 0 or v > 1 for v in outputs), "All outputs in [0,1] — looks like sigmoid was applied"
