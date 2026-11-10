import json

import numpy as np
import pytest

from experiments.ts_dcm_static_shortcut.metrics import episode_bootstrap, score_margin
from experiments.ts_dcm_static_shortcut.protocol import validate_protocol
from experiments.ts_dcm_static_shortcut.transforms import apply_condition


def sample_video():
    rng = np.random.default_rng(3)
    frames = rng.integers(0, 256, size=(4, 12, 16, 3), dtype=np.uint8)
    mask = np.zeros((4, 12, 16), dtype=bool)
    mask[:, 3:9, 5:11] = True
    return frames, mask


def test_e0_is_bitwise_identity_and_frame_count_preserved():
    frames, mask = sample_video()
    out, rec = apply_condition(frames, mask, "E0", seed=4)
    assert np.array_equal(out, frames)
    assert rec["frame_count_preserved"] is True
    assert rec["changed_pixels"] == 0


@pytest.mark.parametrize("condition", ["E1_background_blur", "E2_background_texture"])
def test_background_conditions_preserve_foreground(condition):
    frames, mask = sample_video()
    out1, rec1 = apply_condition(frames, mask, condition, seed=4)
    out2, rec2 = apply_condition(frames, mask, condition, seed=4)
    assert np.array_equal(out1, out2)
    assert np.array_equal(out1[mask], frames[mask])
    assert rec1["foreground_changed_pixels"] == 0
    assert rec1["background_changed_pixels"] > 0
    assert rec1["output_sha256"] == rec2["output_sha256"]


def test_foreground_control_preserves_background():
    frames, mask = sample_video()
    out, rec = apply_condition(frames, mask, "E3_foreground_blur", seed=4)
    assert np.array_equal(out[~mask], frames[~mask])
    assert rec["background_changed_pixels"] == 0
    assert rec["foreground_changed_pixels"] > 0


def test_bad_mask_shape_is_rejected():
    frames, mask = sample_video()
    with pytest.raises(ValueError, match="foreground_mask"):
        apply_condition(frames, mask[:-1], "E1_background_blur")


def test_protocol_rejects_unreviewed_masks_for_formal_run():
    payload = {
        "dataset": "ssv2-small", "checkpoint": "x", "episode_manifest": "e",
        "mask_manifest": "m", "mask_type": "per_frame_binary", "mask_review_status": "pending",
        "query_only": True, "conditions": ["E0", "E1_background_blur", "E2_background_texture", "E3_foreground_blur"],
        "seed": 1, "max_runtime_minutes": 55,
    }
    with pytest.raises(ValueError, match="approved"):
        validate_protocol(payload)


def test_episode_bootstrap_is_clustered_and_reproducible():
    r1 = episode_bootstrap([0, 0, 1, 1], [1, 0, 1, 1], seed=9, repeats=100)
    r2 = episode_bootstrap([0, 0, 1, 1], [1, 0, 1, 1], seed=9, repeats=100)
    assert r1 == r2
    assert r1["episodes"] == 2
    assert r1["queries"] == 4


def test_score_margin():
    scores = np.array([[.8, .2, .4], [.1, .7, .6]])
    assert np.allclose(score_margin(scores, np.array([0, 2])), [.4, -.1])

