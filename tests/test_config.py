import os

import pytest

from molsd.config import load_config

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_default_composes_groups():
    cfg = load_config(os.path.join(CONFIGS, "default.yaml"))
    assert cfg.objective.type == "flow_matching"
    assert list(cfg.data.tasks) == ["canny"]
    assert cfg.training.grad_accum == 4


def test_experiment_inherits_and_overrides():
    cfg = load_config(os.path.join(CONFIGS, "mol.yaml"), overrides=["objective.shift=3.0"])
    assert cfg.experiment_name == "tes-mol"
    assert list(cfg.data.tasks) == ["canny", "depth"]
    assert cfg.data.tasks.canny.weight == 0.2 and cfg.data.tasks.depth.weight == 0.8
    assert cfg.data.tasks.canny.target_dir == "data/canny/targets"  # untouched keys come from the group
    assert cfg.objective.shift == 3.0
    assert "defaults" not in cfg


def test_every_config_file_is_valid():
    for dirpath, _, files in os.walk(CONFIGS):
        for f in files:
            if f.endswith(".yaml"):
                load_config(os.path.join(dirpath, f), root=CONFIGS)


def test_missing_group_and_unknown_key_raise(tmp_path):
    (tmp_path / "bad.yaml").write_text("defaults:\n  - objective/nope\n")
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "bad.yaml"))
    (tmp_path / "typo.yaml").write_text("training:\n  learning_rat: 1.0e-4\n")
    with pytest.raises(Exception):
        load_config(str(tmp_path / "typo.yaml"))
