import numpy as np
import pytest

from printguard.evaluation import metrics as M


def test_classification_report() -> None:
    rep = M.classification_report([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], ["a", "b", "c"])
    assert rep["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert rep["accuracy"] == pytest.approx(0.6)
    assert rep["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert rep["per_class"]["b"]["recall"] == 1.0
    assert rep["per_class"]["c"]["f1"] == 0.0


def test_persistent_k_of_n_is_causal() -> None:
    s = [0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1]
    assert M.persistent(s, 0.5, k=3, n=4).tolist() == [False, False, False, True, False, False, False]
    assert M.persistent(s, 0.5, k=1, n=1).tolist() == [x >= 0.5 for x in s]


def test_alarm_onsets_and_delay() -> None:
    state = [False, False, True, True, False, True]
    assert M.alarm_onsets(state) == [2, 5]
    d = M.detection_delay(state, onset=1)
    assert d.first_alarm == 2 and d.delay_frames == 1 and not d.early_alarm
    d = M.detection_delay(state, onset=3)
    assert d.delay_frames == 0 and d.early_alarm
    assert not M.detection_delay([False] * 4, 1).detected


def test_false_alarms_only_count_onsets_on_clean_frames() -> None:
    states = [[True, True, False, False, True], [False] * 5]
    labels = [[0, 0, 0, 0, 1], [0] * 5]
    fa = M.false_alarms(states, labels, seconds_per_frame=36)
    assert fa["false_alarms"] == 1 and fa["clean_frames"] == 9
    assert fa["per_print_hour"] == pytest.approx(1 / (9 * 36 / 3600))


def test_binary_pr_and_ap_ignore_unknown_labels() -> None:
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.99])
    labels = np.array([1, 0, 1, 0, -1])
    pr = M.binary_pr(scores, labels, 0.5)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (1, 1, 1)
    assert M.average_precision(scores, labels) == pytest.approx((1 + 2 / 3) / 2)
    assert M.average_precision([0.5, 0.4], [0, 0]) == 0.0


def test_summarize_delays() -> None:
    res = [M.DelayResult(3, 5, 2, False), M.DelayResult(3, None, None, True)]
    s = M.summarize_delays(res, seconds_per_frame=10)
    assert s["detection_rate"] == 0.5 and s["median_delay_s"] == 20 and s["early_alarm_sequences"] == 1
