import pytest

from petcls.config import load_config


def test_yaml_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run_name: x\ntrain: {lr: 0.01, epochs: 3}\n")
    cfg = load_config(p, ["train.lr=3e-4", "model.name=scratch_cnn", "train.epochs=5"])
    assert cfg.run_name == "x"
    assert cfg.train.lr == pytest.approx(3e-4)
    assert cfg.train.epochs == 5
    assert cfg.model.name == "scratch_cnn"
    assert cfg.flat()["train.lr"] == pytest.approx(3e-4)


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(KeyError):
        load_config(None, ["train.not_a_key=1"])
    p = tmp_path / "bad.yaml"
    p.write_text("train: {lrr: 1}\n")
    with pytest.raises(KeyError):
        load_config(p)


@pytest.mark.parametrize("name", ["resnet50.yaml", "scratch_cnn.yaml", "smoke.yaml"])
def test_shipped_configs_parse(name):
    cfg = load_config(f"configs/{name}")
    assert cfg.train.epochs > 0
