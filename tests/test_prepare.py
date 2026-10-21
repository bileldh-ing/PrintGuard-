import pytest

from printguard.data.prepare import _check_no_leakage, split_groups, yolo_strings_split


def test_split_groups_stratified_and_disjoint() -> None:
    groups = {"a": [f"a{i}" for i in range(10)], "b": [f"b{i}" for i in range(5)]}
    assign = split_groups(groups, 0.2, seed=1)
    assert set(assign) == set(groups["a"]) | set(groups["b"])
    assert sum(v == "val" for k, v in assign.items() if k.startswith("a")) == 2
    assert sum(v == "val" for k, v in assign.items() if k.startswith("b")) == 1
    assert split_groups(groups, 0.2, seed=1) == assign  # deterministic


def test_yolo_strings_split_by_sequence() -> None:
    assert yolo_strings_split("data1_micro_18_3_jpg.rf.a.jpg", {"18"}, {"5"}) == "val"
    assert yolo_strings_split("good1_flir_5_3.jpg", {"18"}, {"5"}) == "test"
    assert yolo_strings_split("mic6_3_jpg.rf.a.jpg", {"18"}, {"5"}) == "train"


def test_leakage_check() -> None:
    ok = [{"dataset": "d", "group": "g1", "split": "train"}, {"dataset": "d", "group": "g2", "split": "val"}]
    _check_no_leakage(ok)
    with pytest.raises(RuntimeError, match="Leakage"):
        _check_no_leakage([*ok, {"dataset": "d", "group": "g1", "split": "test"}])
