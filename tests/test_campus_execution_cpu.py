"""CPU regression tests; no model loading, CUDA calls, or real training jobs."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from experiments.dto_kd import train as dto_train
from scripts import launch_campus_suite as suite


class LabelTransferTest(unittest.TestCase):
    def test_episode_labels_follow_images_through_cuda_transfer(self):
        moved = []

        def cpu_transfer(tensor, *args, **kwargs):
            copy = tensor.clone()
            moved.append(copy)
            return copy

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, images, support, query, real_labels, embeddings):
                # A CPU spy replaces CUDA; check both index/CE labels were moved.
                for labels in (support, query):
                    if not any(labels is tensor for tensor in moved):
                        raise AssertionError("episode labels did not follow images to CUDA")
                return {"loss": self.weight.square() * images.mean(), "acc": 0.5}

        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batch = {
            "support_set": torch.ones(1, 1, 2, 3, 4, 4),
            "target_set": torch.ones(1, 1, 2, 3, 4, 4),
            "support_labels": torch.tensor([[[0, 1]]]),
            "target_labels": torch.tensor([[[0, 1]]]),
            "real_support_labels": torch.tensor([[[2, 3]]]),
            "real_target_labels": torch.tensor([[[2, 3]]]),
        }
        cfg = types.SimpleNamespace(
            TRAIN=types.SimpleNamespace(NUM_TRAIN_TASKS=1, VAL_FRE_ITER=500, BATCH_SIZE_PER_TASK=1),
            DTO_KD=types.SimpleNamespace(GRADIENT_SAMPLE_INTERVAL=0))
        baseline = types.ModuleType("main")
        baseline.scaler = types.SimpleNamespace(scale=lambda loss: loss,
                                                step=lambda opt: opt.step(), update=lambda: None)
        with mock.patch.dict(sys.modules, {"main": baseline}), \
                mock.patch.object(torch.Tensor, "cuda", cpu_transfer), \
                mock.patch.object(dto_train, "autocast", contextlib.nullcontext):
            dto_train.train_epoch([batch], [], model, None, None, optimizer,
                                  lambda step: None, mock.Mock(), cfg)
        self.assertLess(model.weight.item(), 1.0)
        self.assertEqual(len(moved), 6)


class QueueTest(unittest.TestCase):
    def test_queue_reuses_gpu_after_completion_and_records_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [{"name": name, "output": str(root / name), "status": "queued",
                     "console": str(root / name / "console.log")}
                    for name in ("first", "second", "fails")]

            def command(job, gpu):
                # Exercise real subprocess exit codes and GPU slot reuse, without training.
                rc = 7 if job["name"] == "fails" else 0
                return [sys.executable, "-c", f"import time; time.sleep(0.01); raise SystemExit({rc})"]

            with mock.patch.object(suite, "job_command", command):
                rc = suite.run_queue(jobs, [2], root / "status.json", launch_delay=0, poll=0.01)
            self.assertEqual(rc, 1)
            self.assertEqual([j["status"] for j in jobs], ["completed", "completed", "failed"])
            self.assertEqual([j["gpu"] for j in jobs], [2, 2, 2])
            self.assertEqual(json.loads((root / "status.json").read_text())["jobs"][-1]["returncode"], 7)

    def test_failure_prevents_unstarted_jobs_from_launching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [{"name": name, "output": str(root / name), "status": "queued",
                     "console": str(root / name / "console.log")}
                    for name in ("fails", "must_not_start")]
            with mock.patch.object(suite, "job_command", return_value=[sys.executable, "-c", "raise SystemExit(3)"]):
                self.assertEqual(suite.run_queue(jobs, [0], root / "status.json", 0, poll=0.01), 1)
            self.assertEqual(jobs[1]["status"], "cancelled_after_failure")
            self.assertFalse((root / "must_not_start").exists())

    def test_dry_run_plan_has_all_ten_controls_and_can_exclude_running_e1(self):
        jobs = suite.build_jobs([1, 5], 0, Path("/unused"), set())
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len({j["output"] for j in jobs}), 10)
        reduced = suite.build_jobs([1, 5], 0, Path("/unused"), {"E1_1shot"})
        self.assertEqual(len(reduced), 9)
        self.assertNotIn("E1_1shot", {j["name"] for j in reduced})

    def test_smoke_config_changes_only_generated_file(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = suite.ROOT / "configs/dto_kd/ssv2_small_1shot_E1.yaml"
            before = source.read_bytes()
            out = root / "smoke" / "E1"
            subprocess.run([sys.executable, str(suite.ROOT / "scripts/prepare_run_config.py"),
                            "--config", str(source), "--output", str(out / "config.yaml"),
                            "--output-dir", str(out), "--smoke"], check=True)
            cfg = yaml.safe_load((out / "config.yaml").read_text())
            self.assertEqual(cfg["TRAIN"]["NUM_TRAIN_TASKS"], 2)
            self.assertEqual(cfg["TRAIN"]["NUM_TEST_TASKS"], 2)
            self.assertEqual(cfg["DTO_KD"]["GRADIENT_SAMPLE_INTERVAL"], 1)
            self.assertEqual(len(cfg["TEST"]["CLASS_NAME"]), 12)
            ann = Path(cfg["DATA"]["ANNO_DIR"])
            expected = (suite.ROOT / 'configs/ssv2_small/val_few_shot.txt').read_text().splitlines()
            aliased = (ann / 'test_few_shot.txt').read_text().splitlines()
            self.assertEqual([line[len('test'):] for line in aliased],
                             [line[len('val'):] for line in expected])
            self.assertEqual(json.loads((ann / 'protocol.json').read_text())["current_evaluation_split"], 'val')
            self.assertEqual(source.read_bytes(), before)

    def test_final_evaluation_restores_test_metadata(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = suite.ROOT / 'configs/dto_kd/ssv2_small_1shot_E1.yaml'
            checkpoint = root / 'placeholder.pth'
            checkpoint.touch()  # This test never loads weights.
            out = root / 'evaluation'
            subprocess.run([sys.executable, str(suite.ROOT / 'scripts/prepare_run_config.py'),
                            '--config', str(source), '--output', str(out / 'config.yaml'),
                            '--output-dir', str(out), '--mode', 'eval', '--checkpoint', str(checkpoint)], check=True)
            cfg = yaml.safe_load((out / 'config.yaml').read_text())
            self.assertEqual(len(cfg['TEST']['CLASS_NAME']), 24)
            self.assertFalse(cfg['TRAIN']['ENABLE'])
            self.assertTrue(cfg['TEST']['ENABLE'])
            actual = (Path(cfg['DATA']['ANNO_DIR']) / 'test_few_shot.txt').read_text()
            self.assertEqual(actual, (suite.ROOT / 'configs/ssv2_small/test_few_shot.txt').read_text())


if __name__ == "__main__":
    unittest.main()
