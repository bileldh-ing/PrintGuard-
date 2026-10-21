from printguard.inference.alerts import AlertPolicy, trend


def test_k_of_n_and_cooldown() -> None:
    pol = AlertPolicy(threshold=0.5, k=2, n=3, cooldown_frames=5)
    fired = [pol.update(i, {"stringing": s}) for i, s in enumerate([0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])]
    assert fired[2] == ["stringing"]
    assert all(f == [] for f in fired[3:7])  # muted during cooldown
    assert fired[7] == ["stringing"]


def test_per_label_threshold_and_reset() -> None:
    pol = AlertPolicy(threshold=0.5, k=1, n=1, thresholds={"layer_shift": 0.9})
    assert pol.update(0, {"layer_shift": 0.8, "stringing": 0.8}) == ["stringing"]
    pol.reset()
    assert pol.update(1, {"stringing": 0.8}) == ["stringing"]


def test_trend_slope() -> None:
    assert trend([0.1, 0.2, 0.3]) > 0
    assert trend([0.3, 0.2, 0.1]) < 0
    assert trend([0.5]) == 0.0
