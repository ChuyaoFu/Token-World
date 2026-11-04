import copy
import json
from pathlib import Path
import pytest
from tokenworld.data.audit import audit_manifest
from tokenworld.evaluation.aggregate import macro_average


def episode(task, ident, value, rows):
    return {"task": task, "episode_id": ident,
            "qwen_feature_metrics": {"nmse": value, "rows": rows}, "action_metrics": {}}


def test_macro_weights_tasks_and_episodes_equally_not_frames():
    rows = [episode("a", "1", 0, 100), episode("a", "2", 2, 1), episode("b", "3", 5, 1000)]
    result = macro_average(rows)
    assert result["qwen_feature_metrics"]["nmse"] == 3
    assert result["per_task"]["a"]["qwen_feature_metrics"]["nmse"] == 1


def test_duplicate_or_incomplete_results_are_rejected():
    row = episode("a", "1", 1, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        macro_average([row, row])
    missing = episode("a", "2", None, 1)
    with pytest.raises(ValueError, match="Missing"):
        macro_average([row, missing])


def test_fixed_split_and_overlap_rejection():
    path = Path(__file__).parents[1] / "configs/robotwin/splits/seed3047.json"
    manifest = json.loads(path.read_text())
    report = audit_manifest(manifest)
    assert report["episodes"] == 4999
    assert report["splits"] == {"success_train": 2250, "success_val": 250, "failure_train": 2249, "failure_val": 250}
    assert not report["paper_count_matches"]
    assert report["paper_count_deviations"] == [{"dataset": "robotwin_fail", "task": "place_a2b_left", "train": 44,
                                                "val": 5, "unused": 0, "expected_train": 45, "expected_val": 5}]
    broken = copy.deepcopy(manifest)
    split = next(iter(next(iter(broken["datasets"].values()))["tasks"].values()))
    split["val"][0] = split["train"][0]
    with pytest.raises(ValueError, match="Repeated"):
        audit_manifest(broken)
