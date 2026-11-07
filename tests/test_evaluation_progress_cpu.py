"""Regression for the campus failure at the first 1000-episode progress log."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from datasets.base_dataset import BaseVideoDataset
from experiments.dto_kd.episodes import loader
from experiments.dto_kd.evaluation import evaluate
from tests.test_official_restart_cpu import stream_config
from tests import test_rigorous_protocol_cpu as fixtures


class EvaluationProgressTest(unittest.TestCase):
    def test_actual_official_evaluation_crosses_default_1000_episode_boundary(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', fixtures.read_fake_video):
            root = Path(directory)
            cfg = stream_config(root)
            cfg.TRAIN.NUM_TEST_TASKS = 1001
            model = fixtures.RigorousTrainingTest().make_model(cfg)
            embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                result = evaluate(loader(cfg, 'test', 'test'), model, embeddings,
                                  cfg, 'test', 500, torch.device('cpu'))
            rows = [json.loads(s[len('EVAL_PROGRESS '):]) for s in console.getvalue().splitlines()
                    if s.startswith('EVAL_PROGRESS ')]
            self.assertEqual(rows, [{'split': 'test', 'step': 500, 'episodes': 1000, 'total': 1001}])
            self.assertEqual(result['episodes'], 1001)
            self.assertEqual(result['queries'], 2002)
            self.assertEqual(len(Path(result['predictions']).read_text().splitlines()), 1001)
            self.assertEqual(len(model.ts_dcm._forward_hooks), 1)  # the adapter's own hook


if __name__ == '__main__':
    unittest.main()
