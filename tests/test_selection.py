from printguard.data.selection import (
    evenly_spaced_blocks,
    frame_index,
    roboflow_group,
    select_sequence_label_members,
    select_sequence_members,
    select_trier_members,
    select_yolo_strings_members,
    sequence_id,
)


def test_frame_index() -> None:
    assert frame_index("base/cropped/x/camera1/camera1_017.jpg") == 17
    assert frame_index("detection_cam3_999.JPG") == 999


def test_evenly_spaced_blocks_are_contiguous_and_spread() -> None:
    idx = list(range(100))
    blocks = evenly_spaced_blocks(idx, 4, 10)
    assert len(blocks) == 4
    assert all(b == list(range(b[0], b[0] + 10)) for b in blocks)
    assert blocks[0][0] == 0 and blocks[-1][-1] == 99


def test_evenly_spaced_blocks_all_mode() -> None:
    blocks = evenly_spaced_blocks(list(range(25)), 0, 10)
    assert [len(b) for b in blocks] == [10, 10, 5]


def test_select_trier_members_groups_by_class_camera_block() -> None:
    members = [
        f"base/cropped/{c}/camera{k}/camera{k}_{i}.jpg"
        for c in ("no_error_print", "layershift_error_print")
        for k in (1, 2)
        for i in range(1, 61)
    ]
    members += ["base/full_res/no_error_print/camera1/camera1_1.jpg", "base/cropped/no_error_print/camera1/"]
    groups = select_trier_members(members, "base", "cropped", ["no_error_print"], ["camera1"], 3, 5)
    assert len(groups) == 3
    assert all(g.startswith("base/no_error_print/camera1/b") for g in groups)
    assert all(len(v) == 5 and all("/cropped/no_error_print/camera1/" in m for m in v) for v in groups.values())


def test_roboflow_group_and_sequence_id() -> None:
    assert roboflow_group("datasets/yolo/train/data1_micro_18_37_jpg.rf.abcdef0123.txt") == "data1_micro_18_37"
    assert roboflow_group("datasets/yolo/train/good1_flir_9_38.jpg") == "good1_flir_9_38.jpg"
    assert sequence_id("data1_micro_18_37_jpg.rf.ab.jpg") == "18"
    assert sequence_id("good1_flir_09_1.jpg") == "9"
    assert sequence_id("mic6_58_jpg.rf.84.jpg") is None


def test_select_yolo_strings_keeps_augmented_copies_together() -> None:
    members = [
        "datasets/yolo/train/data1_flir_1_5_jpg.rf.aa.jpg",
        "datasets/yolo/train/data1_flir_1_5_jpg.rf.aa.txt",
        "datasets/yolo/train/data1_flir_1_5_jpg.rf.bb.jpg",
        "datasets/yolo/train/data1_flir_1_5_jpg.rf.bb.txt",
        "datasets/yolo/train/data1_flir_2_7_jpg.rf.cc.jpg",
        "datasets/yolo/train/_darknet.labels",
    ]
    groups = select_yolo_strings_members(members, 0, seed=0)
    assert set(groups) == {"data1_flir_1_5", "data1_flir_2_7"}
    assert len(groups["data1_flir_1_5"]) == 4
    assert len(select_yolo_strings_members(members, 1, seed=0)) == 1


def test_select_sequence_members_orders_frames_numerically() -> None:
    base = "datasets/sequences/with_stringing/sequence_1/micro/data1_micro_1_{}.jpg"
    members = [base.format(i) for i in (1, 10, 2, 3, 20)] + [
        "datasets/sequences/with_stringing/sequence_1/flir/x_1.jpg"
    ]
    out = select_sequence_members(members, "micro", ["sequence_1"], [], stride=1, max_frames=0)
    assert out["with_stringing/sequence_1"] == [base.format(i) for i in (1, 2, 3, 10, 20)]
    out = select_sequence_members(members, "micro", "all", "all", stride=2, max_frames=2)
    assert out["with_stringing/sequence_1"] == [base.format(1), base.format(3)]


def test_select_sequence_label_members_filters_camera_and_sequence() -> None:
    members = [
        "datasets/yolo/valid/data1_micro_18_3_jpg.rf.aa.txt",
        "datasets/yolo/valid/data1_flir_18_3_jpg.rf.aa.txt",
        "datasets/yolo/train/data1_micro_4_3_jpg.rf.aa.txt",
        "datasets/yolo/train/data1_micro_18_3_jpg.rf.aa.jpg",
    ]
    assert select_sequence_label_members(members, "micro", ["sequence_18"]) == [members[0]]
