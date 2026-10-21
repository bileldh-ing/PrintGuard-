import pytest
import torch

from printguard.models.temporal import TCN, TemporalTransformer, build_temporal_model


@pytest.mark.parametrize(
    "model", [TCN(12, 2, channels=16, levels=3), TemporalTransformer(12, 2, d_model=16, heads=2, layers=1)]
)
def test_shapes_and_causality(model: torch.nn.Module) -> None:
    torch.manual_seed(0)
    model.eval()
    x = torch.randn(2, 20, 12)
    y = model(x)
    assert y.shape == (2, 20, 2)
    x2 = x.clone()
    x2[:, 15:] = torch.randn(2, 5, 12)  # change the future only
    y2 = model(x2)
    assert torch.allclose(y[:, :15], y2[:, :15], atol=1e-5), "output at t must not depend on frames > t"
    assert not torch.allclose(y[:, 15:], y2[:, 15:])


def test_build_ignores_keys_of_other_architecture() -> None:
    cfg = {"type": "transformer", "channels": 64, "levels": 4, "d_model": 16, "heads": 2, "layers": 1}
    assert isinstance(build_temporal_model(cfg, 8, 2), TemporalTransformer)
    assert isinstance(build_temporal_model({"type": "tcn", "d_model": 99, "channels": 8}, 8, 2), TCN)
    with pytest.raises(ValueError):
        build_temporal_model({"type": "lstm"}, 8, 2)


def test_tcn_receptive_field() -> None:
    assert TCN(4, 1, levels=4, kernel_size=3).receptive_field == 61
