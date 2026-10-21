import numpy as np

from printguard.data.augment import LayerShiftParams, apply_layer_shift, estimate_cut_row, sample_params, shift_region


def _frames(n: int = 20) -> list[np.ndarray]:
    out = []
    for t in range(n):
        img = np.zeros((60, 80, 3), dtype=np.uint8)
        top = 50 - t  # the "part" grows upward over time
        img[top:50, 30:50] = 200
        out.append(img)
    return out


def test_shift_ramp_is_monotonic_and_starts_at_onset() -> None:
    p = LayerShiftParams(onset=5, ramp=4, max_shift_frac=0.1, direction=1, cut_row=30)
    shifts = [p.shift_px(t, 100) for t in range(12)]
    assert shifts[:5] == [0] * 5
    assert shifts[5] > 0 and shifts == sorted(shifts) and max(shifts) == 10


def test_shift_region_only_moves_rows_above_cut() -> None:
    img = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    out = shift_region(img, 20, 5)
    assert np.array_equal(out[20:], img[20:])
    assert np.array_equal(out[:20, 5:], img[:20, :-5])
    assert np.array_equal(shift_region(img, 20, 0), img)


def test_estimate_cut_row_finds_part_top() -> None:
    f = _frames()
    assert estimate_cut_row(f[0], f[10], min_pixels=5) == 40


def test_apply_layer_shift_labels() -> None:
    frames = _frames()
    params = sample_params(frames, np.random.default_rng(1))
    out, labels = apply_layer_shift(frames, params)
    assert len(out) == len(frames)
    assert labels[: params.onset].sum() == 0 and labels[params.onset :].all()
    assert all(np.array_equal(a, b) for a, b in zip(out[: params.onset], frames[: params.onset], strict=True))
