"""Official Test selection and worker-stream tests, fake videos/LSN only."""
import copy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

from datasets.base_dataset import BaseVideoDataset
from datasets.builder import build_loader
from experiments.dto_kd.episodes import loader, OfficialStreamEpisodes
from experiments.dto_kd.rigorous import train
from experiments.dto_kd.state import rng_state, restore_rng
from scripts.prepare_run_config import prepare_protocol
from scripts.audit_campus_protocol import ROOT, compare_config
from scripts.launch_campus_suite import build_jobs, job_command
from scripts.calibrate_rigorous_scale import calibrate
from tests.test_rigorous_protocol_cpu import configuration, read_fake_video
from tests import test_rigorous_protocol_cpu as fixtures
from tests.test_campus_protocol_cpu import namespace
from utils.functions import cosine_lr
from utils.optimizer import construct_optimizer


def stream_config(root):
    cfg = configuration(root)
    cfg.RIGOROUS = namespace({'SELECTION_PROTOCOL': 'official-test'})
    cfg.TRAIN.NUM_SAMPLES = 12
    cfg.cfg_dict['RIGOROUS'] = {'SELECTION_PROTOCOL': 'official-test'}
    cfg.cfg_dict['TRAIN']['NUM_SAMPLES'] = 12
    return cfg


class OfficialRestartTest(unittest.TestCase):
    def test_official_protocol_preparation_and_audit_uses_24_test_classes(self):
        for shot in (1, 5):
            with self.subTest(shot=shot), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg = yaml.safe_load((ROOT / f'configs/dto_kd/ssv2_small_{shot}shot_E0.yaml').read_text())
                cfg['RIGOROUS'] = {'VERSION': 2, 'SELECTION_PROTOCOL': 'official-test'}
                cfg['OUTPUT_DIR'] = cfg['OUTPUT_DIR_SAVE'] = str(root)
                prepare_protocol(cfg, root / 'config.yaml', 'train')
                (root / 'config.yaml').write_text(yaml.safe_dump(cfg))
                self.assertEqual(len(cfg['TEST']['CLASS_NAME']), 24)
                self.assertEqual((Path(cfg['DATA']['ANNO_DIR']) / 'test_few_shot.txt').read_bytes(),
                                 (ROOT / 'configs/ssv2_small/test_few_shot.txt').read_bytes())
                report = compare_config(root / 'config.yaml', generated=True)
                self.assertTrue(report['configuration_checks_pass'], report)
                self.assertEqual(report['protocol']['checkpoint_selection_split'], 'Novel-Test')
                self.assertEqual(report['protocol']['rng_policy'], 'official-worker-stream')

    def test_observer_preserves_official_loader_inputs_rng_and_sampler(self):
        # Execute the actual original sampler/frame transforms through both
        # loaders. Only video decoding is replaced with synthetic frames.
        for workers in (0, 2):
            for split in ('train', 'test'):
                with self.subTest(workers=workers, split=split), tempfile.TemporaryDirectory() as directory, mock.patch.object(
                        BaseVideoDataset, '_read_video', read_fake_video):
                    cfg = stream_config(Path(directory))
                    cfg.DATA_LOADER.NUM_WORKERS = workers
                    random.seed(443)
                    np.random.seed(444)
                    torch.manual_seed(445)
                    initial = rng_state()
                    original = build_loader(cfg, split)
                    expected = list(itertools.islice(original, 2))
                    expected_rng = rng_state()
                    restore_rng(initial)
                    observed = loader(cfg, split, 'test')
                    actual = list(itertools.islice(observed, 2))
                    self.assertEqual(type(original.sampler), type(observed.sampler))
                    self.assertIsNone(observed.generator)
                    self.assertEqual(len(original), len(observed))
                    self.assertEqual(original.drop_last, observed.drop_last)
                    self.assertEqual(expected_rng['python'], random.getstate())
                    np.testing.assert_equal(expected_rng['numpy'], np.random.get_state())
                    torch.testing.assert_close(expected_rng['torch'], torch.get_rng_state(), rtol=0, atol=0)
                    for left, right in zip(expected, actual):
                        self.assertEqual(set(left), set(right) - {'episode_record'})
                        for key in left:
                            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_official_stream_advances_rng_and_refuses_false_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', read_fake_video):
            cfg = stream_config(Path(directory))
            dataset = OfficialStreamEpisodes(cfg, 'test', 'test')
            before = random.getstate()
            first = dataset[0]
            self.assertNotEqual(before, random.getstate())
            second = dataset[0]
            self.assertNotEqual(first['episode_record'], second['episode_record'])
            with self.assertRaisesRegex(ValueError, 'cannot resume'):
                loader(cfg, 'train', start=2)

    def test_official_training_truncates_random_sampler_and_resamples_test(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', read_fake_video):
            cfg = stream_config(Path(directory))
            model = fixtures.RigorousTrainingTest().make_model(cfg)
            optimizer = construct_optimizer(model, cfg)
            embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
            data = loader(cfg, 'train')
            self.assertGreater(len(data), cfg.TRAIN.NUM_TRAIN_TASKS)
            result = train(data, loader(cfg, 'test', 'test'), model, embeddings, embeddings,
                           optimizer, cosine_lr(optimizer, 0, 4), torch.cuda.amp.GradScaler(enabled=False),
                           cfg, torch.device('cpu'), {'configuration_signature': 'official-cpu',
                                                      'source_sha256': 'official-cpu'})
            self.assertEqual(result['completed_episodes'], 4)
            self.assertEqual(result['successful_updates'], 4)
            self.assertIn('best_novel_test_percent', result)
            self.assertNotIn('best_meta_val_percent', result)
            self.assertFalse(result['exact_resume_supported'])
            rows = [json.loads(row) for row in (Path(directory) / 'train_episodes.jsonl').read_text().splitlines()]
            self.assertEqual([row['episode'] for row in rows], list(range(4)))
            evaluations = [json.loads(row) for row in (Path(directory) / 'validation.jsonl').read_text().splitlines()]
            self.assertEqual([r['split'] for r in evaluations], ['test', 'test'])
            self.assertEqual([r['successful_updates_before_validation'] for r in evaluations], [1, 3])
            self.assertNotEqual(evaluations[0]['episode_sequence_sha256'], evaluations[1]['episode_sequence_sha256'])
            ledgers = list((Path(directory) / 'predictions').glob('*.episodes.jsonl'))
            self.assertEqual(len(ledgers), 2)
            best = json.loads((Path(directory) / 'best_result.json').read_text())
            self.assertEqual(best['accuracy_percent']['fusion'], max(r['accuracy_percent']['fusion'] for r in evaluations))

    def test_instrumented_test_accuracy_matches_original_test_reduction(self):
        from main import test as reference_test
        from experiments.dto_kd.evaluation import evaluate
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                BaseVideoDataset, '_read_video', read_fake_video):
            cfg = stream_config(Path(directory))
            model = fixtures.RigorousTrainingTest().make_model(cfg)
            embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
            initial = rng_state()
            # CPU transfer replacement only; the actual original test(), real
            # loss/fusion, sampler and frame augmentation execute unchanged.
            with mock.patch.object(torch.Tensor, 'cuda', lambda value, *a, **k: value):
                expected, loss = reference_test(build_loader(cfg, 'test'), model, embeddings, cfg)
            restore_rng(initial)
            actual = evaluate(loader(cfg, 'test', 'test'), model, embeddings, cfg, 'test', 0, torch.device('cpu'))
            self.assertEqual(actual['accuracy_percent']['fusion'], expected)
            self.assertAlmostEqual(actual['loss'], loss, places=7)

    def test_confirmed_queue_has_exactly_eight_jobs_and_propagates_protocol(self):
        jobs = [j for j in build_jobs([1, 5], 0, Path('/fresh'), set(), 'v2', True)
                if j['experiment'] in ('E0', 'E1_safe', 'E2')]
        self.assertEqual(len(jobs), 8)
        self.assertFalse(any(j['experiment'] in ('E1', 'E3') for j in jobs))
        for job in jobs:
            job.update(protocol='v2', selection_protocol='official-test', acceptance_report='/new/report.json')
            command = job_command(job, 0)
            self.assertEqual(command[command.index('--selection-protocol') + 1], 'official-test')

    def test_e3_accepts_safe_dynamic_but_rejects_unrelated_modes(self):
        rows = [{'parameter_group': 'lsn', 'step': i * 25, 'mode': 'safe_dynamic', 'finite': True,
                 'raw_distill_norm': 2., 'raw_task_norm': 4.,
                 'weighted_norm_ratio_distill_over_task': .25, 'task_coefficient': .04}
                for i in range(1, 21)]
        self.assertEqual(calibrate(rows)['static_kl_alpha'], .5)
        bad = copy.deepcopy(rows)
        bad[0]['mode'] = 'static_kl'
        with self.assertRaises(ValueError):
            calibrate(bad)

    def test_safe_e3_freezes_teacher_target_and_records_train_only_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gradients = root / 'gradient_groups.jsonl'
            rows = [{'parameter_group': 'lsn', 'step': i * 25, 'mode': 'safe_dynamic', 'finite': True,
                     'raw_distill_norm': 2., 'raw_task_norm': 4.,
                     'weighted_norm_ratio_distill_over_task': .25, 'task_coefficient': .04}
                    for i in range(1, 21)]
            gradients.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            cfg, report = root / 'E3.yaml', root / 'calibration.json'
            subprocess.run([sys.executable, str(ROOT / 'scripts/calibrate_rigorous_scale.py'),
                            '--gradients', str(gradients), '--config',
                            str(ROOT / 'configs/dto_kd/ssv2_small_1shot_E3.yaml'),
                            '--output-config', str(cfg), '--output-report', str(report)],
                           check=True, capture_output=True, text=True)
            result = yaml.safe_load(cfg.read_text())
            self.assertTrue(result['DTO_KD']['DETACH_TEACHER'])
            self.assertEqual(result['DTO_KD']['STATIC_KL_ALPHA'], .5)
            provenance = json.loads(report.read_text())
            self.assertEqual(provenance['selection_split'], 'Meta-Train only')
            self.assertEqual(provenance['source_mode'], 'safe_dynamic')
            self.assertIn('source_gradients_sha256', provenance)

    def test_restart_never_signals_unverified_launcher_or_escaped_job(self):
        from scripts import restart_campus_official_test as restart
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / 'old_project'
            root = project / 'runs/old_queue'
            root.mkdir(parents=True)
            (root / 'launcher.pid').write_text('12345')
            job = {'config': str(project / 'configs/dto_kd/E0.yaml'),
                   'output': str(root / 'E0_1shot'), 'status': 'running', 'pid': 12346}
            (root / 'status.json').write_text(json.dumps({'jobs': [job]}))
            with mock.patch.object(restart, 'validate_launcher', side_effect=RuntimeError('PID mismatch')), \
                    mock.patch.object(restart, 'inspect_running', return_value=None), \
                    mock.patch.object(restart.os, 'kill') as kill:
                info = restart.queue_info(root)
                self.assertIsNone(info['launcher'])
                kill.assert_not_called()
            job['output'] = str(project / 'outside_queue')
            (root / 'status.json').write_text(json.dumps({'jobs': [job]}))
            with self.assertRaisesRegex(RuntimeError, 'outside'):
                restart.queue_info(root)

    def test_bad_initialization_hash_stops_before_any_old_queue_signal(self):
        from scripts import restart_campus_official_test as restart
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / 'previous/shot1'
            previous.mkdir(parents=True)
            (previous / 'initialization.pth').write_bytes(b'changed initial weights')
            (previous / 'acceptance.json').write_text(json.dumps({'passed': True, 'shot': 1, 'seed': 0,
                'initialization_file': 'initialization.pth', 'initialization_file_sha256': 'wrong'}))
            argv = ['restart', '--old-run-root', str(root / 'old'),
                    '--previous-acceptance-root', str(previous.parent), '--output-root', str(root / 'new'), '--apply']
            with mock.patch.object(restart.sys, 'argv', argv), \
                    mock.patch.object(restart.shutil, 'disk_usage', return_value=namespace({'free': 40 * 1024**3})), \
                    mock.patch.object(restart, 'queue_info', return_value={}), \
                    mock.patch.object(restart, 'stop_queues') as stop, \
                    mock.patch.object(restart.subprocess, 'check_output') as query:
                with self.assertRaises(SystemExit):
                    restart.main()
                stop.assert_not_called()
                query.assert_not_called()
                self.assertFalse((root / 'new').exists())

    def test_background_driver_preserves_ignored_sigint_for_tail_ctrl_c(self):
        from scripts import restart_campus_official_test as restart
        with mock.patch.object(restart.signal, 'getsignal', return_value=restart.signal.SIG_IGN), \
                mock.patch.object(restart.signal, 'signal') as install:
            restart.install_interrupt_handlers([])
            self.assertEqual([call.args[0] for call in install.call_args_list], [restart.signal.SIGTERM])


if __name__ == '__main__':
    unittest.main()
