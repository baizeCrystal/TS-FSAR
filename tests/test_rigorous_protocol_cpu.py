"""Protocol-v2 integration tests: real sampler/losses, fake video/LSN, CPU only."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock
import subprocess
import sys
import weakref

import numpy as np
import torch
import yaml

from datasets.base_dataset import BaseVideoDataset
from experiments.dto_kd.controller import DTOController
from experiments.dto_kd.episodes import AuditedEpisodes, loader
from experiments.dto_kd.model import DTOTSFSAR
from experiments.dto_kd.rigorous import train
from experiments.dto_kd.state import EpisodeLedger, cpu_state, restore_rng, rng_state
from scripts.audit_campus_protocol import ROOT
from scripts.calibrate_rigorous_scale import calibrate
from scripts.compare_rigorous_results import compare
from scripts.launch_campus_suite import build_jobs, job_command
from tests.test_campus_protocol_cpu import TinyLSN, namespace
from utils.functions import cosine_lr
from utils.optimizer import construct_optimizer


class FakeFrames:
    def __init__(self, value):
        self.value = value

    def to_dlpack(self):
        return torch.utils.dlpack.to_dlpack(self.value)


class FakeVideo:
    def __init__(self, path):
        self.value = int(Path(path).stem) % 200

    def __len__(self):
        return 12

    def get_avg_fps(self):
        return 12.

    def get_batch(self, indices):
        # Actual decoding interface, frame selection, transforms and sampler run.
        frames = torch.stack([torch.full((10, 12, 3), self.value + int(i), dtype=torch.uint8)
                              for i in indices])
        return FakeFrames(frames)


def read_fake_video(self, path, index):
    return FakeVideo(path), [], True


def configuration(root, shot=1):
    root.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load((ROOT / f'configs/dto_kd/ssv2_small_{shot}shot_E0.yaml').read_text())
    raw['TRAIN'].update(WAY=2, NUM_CLASSES=3, QUERY_PER_CLASS=1, NUM_TRAIN_TASKS=4,
                        NUM_TEST_TASKS=2, VAL_FRE_ITER=2, RANDOM_SEED=19)
    raw['DATA'].update(DATA_ROOT_DIR=str(root), ANNO_DIR=str(root / 'ann'),
                       NUM_INPUT_FRAMES=2, TRAIN_JITTER_SCALES=[8, 8], TRAIN_CROP_SIZE=8,
                       TEST_SCALE=8, TEST_CROP_SIZE=8)
    raw['DATA_LOADER']['NUM_WORKERS'] = 0
    raw['LSN'].update(EMBED_DIM=4, OUTPUT_DIM=4)
    raw['DTO_KD'].update(MODE='dynamic', UPDATE_RULE='paper_pi_softmax', TRACE_INTERVAL=0,
                         GRADIENT_SAMPLE_INTERVAL=2, ACTIVATION_OFFLOAD='diagnostics')
    raw['OUTPUT_DIR'] = raw['OUTPUT_DIR_SAVE'] = str(root)
    raw['LOG_PERIOD'] = 1
    ann = root / 'ann'
    ann.mkdir(exist_ok=True)
    for split in ('train', 'test'):
        (ann / f'{split}_few_shot.txt').write_text(''.join(
            f'{split}{c}/{100 + c * 20 + i}\n' for c in range(2) for i in range(8)))
    cfg = namespace(raw)
    cfg.cfg_dict = copy.deepcopy(raw)
    return cfg


class PaperControllerTest(unittest.TestCase):
    def test_direct_probability_update_matches_paper_equation_and_differs_from_legacy(self):
        controller = DTOController(eta=0.03, init_distill=0.3, update_rule='paper_pi_softmax')
        legacy = DTOController(eta=0.03, init_distill=0.3)
        p = torch.tensor([0.3, 0.7], dtype=torch.float64)
        logs = torch.tensor([4., 0.5]).log().double()
        expected = (p - 0.03 * p.dot(logs) * logs).softmax(0)
        controller.update(torch.tensor(4.), torch.tensor(0.5))
        legacy.update(torch.tensor(4.), torch.tensor(0.5))
        torch.testing.assert_close(torch.tensor(controller.weights).double(), expected, rtol=1e-7, atol=1e-7)
        self.assertNotEqual(controller.weights, legacy.weights)

    def test_paper_state_resume_and_extreme_losses(self):
        controller = DTOController(update_rule='paper_pi_softmax')
        for d, t in ((0., 0.), (1e-30, 1e20), (3., 0.25)):
            controller.update(torch.tensor(d), torch.tensor(t))
            self.assertAlmostEqual(sum(controller.weights), 1.)
            self.assertTrue(all(np.isfinite(controller.weights)))
        restored = DTOController()
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.update_rule, 'paper_pi_softmax')
        for item in (controller, restored):
            item.update(torch.tensor(0.7), torch.tensor(1.5))
        self.assertEqual(controller.weights, restored.weights)


class EpisodeProtocolTest(unittest.TestCase):
    def test_actual_sampler_has_same_order_frames_augmentation_across_workers(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', read_fake_video):
            cfg = configuration(Path(directory))
            dataset = AuditedEpisodes(cfg, 'train')
            first = dataset[0]
            random.seed(991)
            np.random.seed(992)
            torch.manual_seed(993)
            previous = rng_state()
            replay = dataset[0]
            self.assertEqual(first['episode_record'], replay['episode_record'])
            torch.testing.assert_close(previous['torch'], torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(previous['python'], random.getstate())
            np.testing.assert_equal(previous['numpy'], np.random.get_state())
            record = json.loads(first['episode_record'])
            self.assertEqual(len(record['support']), 2)
            self.assertEqual(len(record['query']), 2)
            self.assertTrue(all(len(item['reads'][0]['frames']) == 2 for item in record['support']))
            cfg.DATA_LOADER.NUM_WORKERS = 2
            worker = next(iter(loader(cfg, 'train')))
            self.assertEqual(worker['episode_record'][0], first['episode_record'])
            self.assertNotEqual(first['episode_record'], dataset[1]['episode_record'])

    def test_val_test_are_distinct_and_five_shot_order_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', read_fake_video):
            cfg = configuration(Path(directory), shot=5)
            val = json.loads(AuditedEpisodes(cfg, 'test', 'val')[0]['episode_record'])
            test = json.loads(AuditedEpisodes(cfg, 'test', 'test')[0]['episode_record'])
            self.assertEqual(len(val['support']), 10)
            self.assertNotEqual(val['seed'], test['seed'])
            self.assertNotEqual(val['episode_sha256'], test['episode_sha256'])

    def test_ledger_rejects_changed_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger.jsonl'
            ledger = EpisodeLedger(path)
            ledger.observe({'episode': 0, 'episode_sha256': 'a'})
            digest = ledger.sequence_hash(1)
            replay = EpisodeLedger(path)
            replay.observe({'episode': 0, 'episode_sha256': 'a'})
            self.assertEqual(digest, replay.sequence_hash(1))
            with self.assertRaises(RuntimeError):
                replay.observe({'episode': 0, 'episode_sha256': 'b'})


class AdmissionAndAnalysisTest(unittest.TestCase):
    def test_v2_plan_uses_shot_specific_half_double_and_avoids_duplicate_center(self):
        jobs = build_jobs([1, 5], 0, Path('/unused'), set(), protocol='v2')
        self.assertEqual(len(jobs), 8)
        self.assertEqual({j['name'] for j in jobs if j['experiment'] == 'E2'},
                         {'E2_1shot_kl0.05', 'E2_1shot_kl0.2', 'E2_5shot_kl0.005', 'E2_5shot_kl0.02'})
        jobs[0].update(protocol='v2', acceptance_report='/unused/acceptance.json')
        command = job_command(jobs[0], 4)
        self.assertIn('--acceptance-report', command)
        self.assertEqual(command[command.index('--protocol') + 1], 'v2')

    def test_v2_formal_config_refuses_absent_or_failed_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            command = [sys.executable, str(ROOT / 'scripts/prepare_run_config.py'),
                       '--config', str(ROOT / 'configs/dto_kd/ssv2_small_5shot_E0.yaml'),
                       '--output', str(out / 'config.yaml'), '--output-dir', str(out), '--protocol', 'v2']
            absent = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(absent.returncode, 0)
            certificate = out / 'failed.json'
            certificate.write_text('{"passed": false}')
            failed = subprocess.run(command + ['--acceptance-report', str(certificate)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((out / 'config.yaml').exists())

    def test_bootstrap_requires_exact_episode_inputs_and_labels(self):
        reference, candidate = {}, {}
        for i in range(4):
            row = {'episode': i, 'episode_sha256': str(i), 'labels': [0, 1],
                   'predictions': {b: [1, 0] for b in ('matching', 'clip', 'side', 'fusion')}}
            reference[i] = row
            candidate[i] = {**copy.deepcopy(row), 'predictions': {b: [0, 1] for b in row['predictions']}}
        result = compare(reference, candidate)
        self.assertEqual(result['branches']['fusion']['ci95_pp'], [100., 100.])
        self.assertEqual(result['branches']['fusion']['query_counts']['corrected'], 8)
        candidate[0]['episode_sha256'] = 'different frames'
        with self.assertRaises(ValueError):
            compare(reference, candidate)

    def test_scale_calibration_requires_twenty_shared_lsn_samples_from_dynamic(self):
        rows = [{'parameter_group': 'lsn', 'step': i * 25, 'mode': 'dynamic', 'finite': True,
                 'raw_distill_norm': 2., 'raw_task_norm': 4.,
                 'weighted_norm_ratio_distill_over_task': 1., 'task_coefficient': .2}
                for i in range(1, 21)]
        self.assertEqual(calibrate(rows)['static_kl_alpha'], 2.)
        with self.assertRaises(ValueError):
            calibrate(rows[:1])
        with self.assertRaises(ValueError):
            calibrate(rows + [rows[0]])


class RigorousTrainingTest(unittest.TestCase):
    def make_model(self, cfg):
        with mock.patch('trainers.ts_fsar.LSN', side_effect=lambda *args: TinyLSN()):
            return DTOTSFSAR(cfg, torch.nn.Identity())

    def test_training_images_are_released_before_validation_with_gradients_preserved(self):
        from experiments.dto_kd import rigorous
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', read_fake_video):
            cfg = configuration(Path(directory))
            cfg.TRAIN.NUM_TRAIN_TASKS = cfg.TRAIN.VAL_FRE_ITER = 1
            model = self.make_model(cfg)
            optimizer = construct_optimizer(model, cfg)
            embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
            original_unpack, original_evaluate = rigorous.unpack, rigorous.evaluate
            images = []

            def unpack(*args):
                values = original_unpack(*args)
                images.append(weakref.ref(values[0]))
                return values

            def evaluate(*args):
                self.assertTrue(all(value() is None for value in images))
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                return original_evaluate(*args)

            with mock.patch.object(rigorous, 'unpack', side_effect=unpack), mock.patch.object(
                    rigorous, 'evaluate', side_effect=evaluate):
                train(loader(cfg, 'train'), loader(cfg, 'test'), model, embeddings, embeddings,
                      optimizer, cosine_lr(optimizer, 0, 1), torch.cuda.amp.GradScaler(enabled=False),
                      cfg, torch.device('cpu'), {'configuration_signature': 'release-inputs',
                                                 'source_sha256': 'release-inputs'})

    def test_uninterrupted_and_full_state_resume_are_exactly_equal(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                self.check_full_resume(workers)

    def check_full_resume(self, workers):
        class StopAfterCheckpoint(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', read_fake_video):
            root = Path(directory)
            full_cfg, resumed_cfg = configuration(root / 'full'), configuration(root / 'resumed')
            full_cfg.DATA_LOADER.NUM_WORKERS = resumed_cfg.DATA_LOADER.NUM_WORKERS = workers
            torch.manual_seed(144)
            initial = self.make_model(full_cfg)
            initial_state = cpu_state(initial)
            embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
            initial_rng = rng_state()
            metadata = {'configuration_signature': 'cpu-integration-v2', 'source_sha256': 'cpu-integration-v2'}

            def run(cfg, resume=None, interrupt=False):
                model = self.make_model(cfg)
                model.load_state_dict(initial_state)
                optimizer = construct_optimizer(model, cfg)
                scheduler = cosine_lr(optimizer, 0, cfg.TRAIN.NUM_TRAIN_TASKS)
                data = loader(cfg, 'train', start=resume['completed_episodes'] if resume else 0)
                if interrupt:
                    def interrupted():
                        for index, batch in enumerate(data):
                            if index == 2:
                                raise StopAfterCheckpoint
                            yield batch
                    train_data = interrupted()
                else:
                    train_data = data
                restore_rng(initial_rng)
                summary = train(train_data, loader(cfg, 'test'), model, embeddings, embeddings,
                                optimizer, scheduler, torch.cuda.amp.GradScaler(enabled=False), cfg,
                                torch.device('cpu'), metadata, resume)
                return model, optimizer, summary

            full_model, full_optimizer, full_summary = run(full_cfg)
            with self.assertRaises(StopAfterCheckpoint):
                run(resumed_cfg, interrupt=True)
            saved = torch.load(root / 'resumed/resume_latest.pth', map_location='cpu')
            self.assertEqual(saved['completed_episodes'], 2)
            resumed_model, resumed_optimizer, resumed_summary = run(resumed_cfg, saved)
            for name, value in full_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(full_model.dto_controller.weights, resumed_model.dto_controller.weights)
            self.assertEqual(full_summary['train_sequence_sha256'], resumed_summary['train_sequence_sha256'])
            self.assertEqual(full_summary['successful_updates'], 4)
            traces = [json.loads(line) for line in (root / 'full/train_trace.jsonl').read_text().splitlines()]
            self.assertTrue(all('teacher_entropy' in row['global_responses'] for row in traces))
            for key, value in full_optimizer.state_dict()['state'].items():
                for field, actual in value.items():
                    torch.testing.assert_close(actual, resumed_optimizer.state_dict()['state'][key][field], rtol=0, atol=0)
            rows = [json.loads(line) for line in (root / 'full/gradient_groups.jsonl').read_text().splitlines()]
            self.assertTrue(any(row['parameter_group'] == 'lsn' and row['finite'] for row in rows))
            self.assertTrue(all('weighted_norm_ratio_distill_over_task' in row for row in rows if row['finite']))
            evaluations = [json.loads(line) for line in (root / 'full/validation.jsonl').read_text().splitlines()]
            self.assertEqual(evaluations[0]['successful_updates_before_validation'], 1)
            for evaluation in evaluations:
                self.assertEqual(set(evaluation['accuracy_percent']), {'matching', 'clip', 'side', 'fusion'})
                first = json.loads(Path(evaluation['predictions']).read_text().splitlines()[0])
                self.assertEqual(set(first['predictions']), {'matching', 'clip', 'side', 'fusion'})

    def test_amp_skipped_update_rolls_back_controller_and_preserves_parameters(self):
        class SkipScaler:
            def __init__(self):
                self.value = 1.
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def step(self, optimizer):
                pass
            def update(self):
                self.value *= .5
            def get_scale(self):
                return self.value
            def state_dict(self):
                return {'scale': self.value}

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', read_fake_video):
            cfg = configuration(Path(directory))
            cfg.TRAIN.NUM_TRAIN_TASKS = cfg.TRAIN.VAL_FRE_ITER = 1
            cfg.DTO_KD.GRADIENT_SAMPLE_INTERVAL = 1
            model = self.make_model(cfg)
            initial = cpu_state(model)
            before = model.dto_controller.weights
            embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
            optimizer = construct_optimizer(model, cfg)
            result = train(loader(cfg, 'train'), loader(cfg, 'test'), model, embeddings, embeddings,
                           optimizer, cosine_lr(optimizer, 0, 1), SkipScaler(), cfg, torch.device('cpu'),
                           {'configuration_signature': 'skip-cpu', 'source_sha256': 'skip-cpu'})
            self.assertEqual(result['successful_updates'], 0)
            self.assertEqual(model.dto_controller.weights, before)
            self.assertEqual(model.dto_controller.step_index, 0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
