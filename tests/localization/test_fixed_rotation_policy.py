from collections import Counter
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[2] / "scripts/match_rotated_queries.py"
spec = importlib.util.spec_from_file_location("match_rotated_queries", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_user_winter50_policy_covers_all_84_files_without_search():
    policy = module.parse_fixed_rotations(
        "4-5:270,6-15:90,16-22:180,23-65:270,66-75:90,76-87:270"
    )
    assert set(policy) == {f"{i:06d}" for i in range(4, 88)}
    assert Counter(policy.values()) == {270.0: 57, 90.0: 20, 180.0: 7}
    assert policy["000005"] == 270 and policy["000006"] == 90
    assert policy["000065"] == 270 and policy["000066"] == 90
    assert policy["000075"] == 90 and policy["000076"] == 270


def test_counterclockwise_angles_are_normalized_but_conflicts_fail_closed():
    assert module.parse_fixed_rotations("4-5:-90") == {"000004": 270.0, "000005": 270.0}
    assert module.parse_fixed_rotations("") == {}
    for invalid in ("4-5:90,5-6:180", "5-4:90", "4:nan"):
        with pytest.raises(ValueError):
            module.parse_fixed_rotations(invalid)
