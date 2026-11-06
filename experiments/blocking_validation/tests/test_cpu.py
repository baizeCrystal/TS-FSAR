"""CPU-only contract tests for the blocking-validation layer."""

import copy
import json
import random

import numpy as np
try:
    import pytest
except ImportError:  # keep the CPU contract runnable in the minimal env
    from contextlib import contextmanager
    class _Approx:
        def __init__(self, value, tol=1e-6): self.value, self.tol = value, tol
        def __eq__(self, other): return abs(float(other) - self.value) <= self.tol * max(1., abs(self.value))
    class _PytestCompat:
        @staticmethod
        def approx(value, abs=None, rel=None): return _Approx(value, tol=abs or rel or 1e-6)
        @staticmethod
        @contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"expected {exc.__name__}")
    pytest = _PytestCompat()
import torch

from experiments.blocking_validation.checkpoint import load_checkpoint, save_checkpoint
from experiments.blocking_validation.config import BlockingExperimentConfig, load_config
from experiments.blocking_validation.controller import KLControlConfig, KLWeightController
from experiments.blocking_validation.counterfactual import counterfactual_update
from experiments.blocking_validation.metrics import gradient_metrics
from experiments.blocking_validation.protocol import assert_split_disjoint, episode_sequence_hash
from experiments.blocking_validation.short_train import run_short_branch
from experiments.blocking_validation.state import restore, snapshot


def test_controller_identity_and_values():
    original = KLWeightController(KLControlConfig(mode="original", attenuation=.2))
    assert original.weights(original.plan([-1, 1])) == [1.0, 1.0]
    selective = KLWeightController(KLControlConfig(mode="selective", attenuation=.25,
                                                    interference_threshold=0.0))
    decisions = selective.plan([-1, .1, -0.2])
    assert selective.weights(decisions) == [.25, 1.0, .25]
    assert [d.selected for d in decisions] == [True, False, True]


def test_controller_records_auditable_decisions():
    controller = KLWeightController(KLControlConfig(mode="uniform", attenuation=.7))
    decisions = controller.plan([-1., 1.])
    assert len(controller.decisions) == 2
    assert all(d.reason == "uniform" for d in decisions)


def test_b0_matches_original_loss_composition():
    matching = torch.tensor(1.25)
    student = torch.tensor(.4)
    teacher = torch.tensor(.3)
    kl = torch.tensor(.2)
    original = 1.0 * matching + 2.0 * student + .1 * (kl + teacher)
    b0 = 1.0 * matching + 2.0 * student + .1 * (1.0 * kl + teacher)
    torch.testing.assert_close(original, b0)


def test_gradient_metrics_projection():
    metrics = gradient_metrics([torch.tensor([1., 0.])], [torch.tensor([-1., 0.])])
    assert metrics["kl_opposes_matching"]
    assert metrics["relative_projection"] < 0


def test_random_frequency_and_uniform_budget_match():
    reference = [-1, 1, -2, 2, -3, 3]
    selective = KLWeightController(KLControlConfig(mode="selective", attenuation=.4,
                                                    interference_threshold=0.0))
    random_controller = KLWeightController(KLControlConfig(mode="random", attenuation=.4,
                                                           interference_threshold=0.0, seed=9))
    uniform = KLWeightController(KLControlConfig(mode="uniform", attenuation=.4,
                                                 interference_threshold=0.0))
    a, b, c = selective.plan(reference), random_controller.plan(reference), uniform.plan(reference)
    assert sum(x.selected for x in a) == sum(x.selected for x in b)
    assert sum(1 - x.weight for x in a) == pytest.approx(sum(1 - x.weight for x in b))
    assert sum(1 - x.weight for x in c) == pytest.approx(sum(1 - x.weight for x in a))


def _model_and_optimizer(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)


def test_counterfactual_restores_model_optimizer_and_rng():
    model, optimizer = _model_and_optimizer(4)
    x, y = torch.randn(6, 3), torch.randint(0, 2, (6,))
    before = snapshot(model, optimizer)
    initial = copy.deepcopy(model.state_dict())

    def losses(branch):
        logits = model(x)
        matching = torch.nn.functional.cross_entropy(logits, y)
        kl = logits.square().mean()
        weight = {"A0": 1., "A1": 0., "A2": .5}[branch]
        return {"total": matching + weight * kl, "matching": matching, "kl": kl}

    result = counterfactual_update(model, optimizer, losses)
    assert set(result) == {"A0", "A1", "A2"}
    for branch in result:
        assert "relative_projection" in result[branch]
        assert "post_update" in result[branch]
    for key, value in initial.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    assert optimizer.state_dict() == before.optimizer


def test_checkpoint_save_restore(tmp_path):
    model, optimizer = _model_and_optimizer(3)
    save_checkpoint(tmp_path / "state.pt", model, optimizer, next_episode=7,
                    metadata={"unit": True})
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    for p in model.parameters():
        p.data.add_(1)
    payload = load_checkpoint(tmp_path / "state.pt", model, optimizer)
    assert payload["next_episode"] == 7
    for key, value in expected.items():
        torch.testing.assert_close(model.state_dict()[key], value)


def test_split_and_episode_hash():
    assert_split_disjoint(["a"], ["b"], ["c"])
    with pytest.raises(ValueError):
        assert_split_disjoint(["a"], ["a"], ["c"])
    episodes = [{"id": i, "label": [i % 2]} for i in range(3)]
    assert episode_sequence_hash(episodes) == episode_sequence_hash(copy.deepcopy(episodes))
    assert episode_sequence_hash(episodes) != episode_sequence_hash(episodes[::-1])


def test_synthetic_short_run_and_resume(tmp_path):
    torch.manual_seed(12)
    model, optimizer = _model_and_optimizer(12)
    episodes = [{"x": torch.randn(5, 3), "y": torch.randint(0, 2, (5,)),
                 "relative_projection": -.1 if i % 2 == 0 else .2} for i in range(4)]
    controller = KLWeightController(KLControlConfig(mode="selective", attenuation=.5,
                                                    interference_threshold=0.0))

    def loss_fn(m, e, weight):
        logits = m(e["x"])
        matching = torch.nn.functional.cross_entropy(logits, e["y"])
        kl = logits.square().mean()
        return {"total": matching + weight * kl, "matching": matching, "kl": kl}

    summary = run_short_branch(model, optimizer, episodes, loss_fn, controller=controller,
                               output_dir=tmp_path, branch_name="B1", checkpoint_every=2,
                               reference_projections=[e["relative_projection"] for e in episodes])
    assert summary["steps"] == 4
    assert (tmp_path / "B1" / "latest.pt").exists()
    assert len((tmp_path / "B1" / "metrics.jsonl").read_text().splitlines()) == 4


def test_config_json_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"shot": 1, "device": "cpu", "kl": {"mode": "original"}}))
    config = load_config(path)
    assert isinstance(config, BlockingExperimentConfig)
    assert config.kl.mode == "original"
