"""Gradient-path regression tests; tiny CPU features, no real model inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from experiments.dto_kd.acceptance import require_adaptation_certificate
from experiments.dto_kd.diagnostics import diagnose
from experiments.dto_kd.model import DTOTSFSAR
from scripts.audit_campus_protocol import ROOT
from scripts.launch_campus_suite import build_jobs
from scripts.stop_campus_jobs import inspect_job
from tests.test_campus_protocol_cpu import TinyLSN, namespace


class AdaptationGradientTest(unittest.TestCase):
    def fixture(self, variant, shot=1):
        raw = yaml.safe_load((ROOT / f'configs/dto_kd/ssv2_small_{shot}shot_{variant}.yaml').read_text())
        raw['TRAIN'].update(WAY=2, QUERY_PER_CLASS=1, NUM_CLASSES=3)
        raw['DATA']['NUM_INPUT_FRAMES'] = 2
        raw['LSN'].update(EMBED_DIM=4, OUTPUT_DIM=4)
        raw['DTO_KD'].update(TRACE_INTERVAL=0, UPDATE_RULE='paper_pi_softmax')
        cfg = namespace(raw)
        torch.manual_seed(47)
        with mock.patch('trainers.ts_fsar.LSN', side_effect=lambda *args: TinyLSN()):
            model = DTOTSFSAR(cfg, torch.nn.Identity()).train()
        support = torch.arange(2).repeat_interleave(shot)
        query = torch.arange(2)
        labels = torch.cat([support, query])
        inputs = (torch.randn(len(labels) * 2, 3, 2, 2), support, query, labels,
                  torch.nn.functional.normalize(torch.randn(3, 4), dim=1))
        return cfg, model, inputs

    def test_only_kl_teacher_path_is_stopped_and_teacher_ce_still_trains(self):
        for shot in (1, 5):
            for variant in ('E0', 'E1', 'E0_detach', 'E1_detach', 'E1_safe'):
                with self.subTest(shot=shot, variant=variant):
                    cfg, model, inputs = self.fixture(variant, shot)
                    result = model(*inputs)
                    task, kl, student, teacher, side = model._components(inputs[2])
                    params = list(model.model.Adapter.parameters())
                    kl_grad = torch.autograd.grad(kl, params, retain_graph=True, allow_unused=True)
                    ce_grad = torch.autograd.grad(teacher, params, retain_graph=True)
                    self.assertTrue(any(float(g.norm()) > 0. for g in ce_grad))
                    if variant in ('E0', 'E1'):
                        self.assertTrue(any(g is not None and float(g.norm()) > 0. for g in kl_grad))
                    else:
                        self.assertTrue(all(g is None for g in kl_grad))
                        total_grad = torch.autograd.grad(result['loss'], params, retain_graph=True)
                        weight = cfg.TRAIN.KD_ALPHA
                        if variant == 'E1_safe':
                            weight *= model.dto_last_metrics['auxiliary_coefficient']
                        for actual, expected in zip(total_grad, ce_grad):
                            torch.testing.assert_close(actual, weight * expected, rtol=1e-5, atol=1e-7)
                    self.assertIsNotNone(torch.autograd.grad(kl, model.model.Linear.weight, retain_graph=True)[0])

    def test_detach_controls_preserve_forward_values_predictions_and_rng(self):
        for original, detached in [('E0', 'E0_detach'), ('E1', 'E1_detach')]:
            _, left, inputs = self.fixture(original)
            _, right, _ = self.fixture(detached)
            right.load_state_dict(left.state_dict(), strict=True)
            rng = torch.get_rng_state()
            a = left(*inputs)
            after = torch.get_rng_state()
            torch.set_rng_state(rng)
            b = right(*inputs)
            torch.testing.assert_close(a['loss'], b['loss'], rtol=0, atol=0)
            self.assertEqual(float(a['acc']), float(b['acc']))
            torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)

    def test_safe_small_kl_has_bounded_derivative_and_all_ce_in_supervised_objective(self):
        for shot in (1, 5):
            for value in (0., 1e-12, 1e-4, 1., 100.):
                cfg, model, inputs = self.fixture('E1_safe', shot)
                leaf = torch.tensor(value, requires_grad=True)
                with mock.patch.object(model.kl, 'forward', return_value=leaf):
                    result = model(*inputs)
                task, kl, student, teacher, side = model._components(inputs[2])
                supervised = (cfg.TRAIN.VIS_ALPHA * task + cfg.TRAIN.CLS_ALPHA * student
                              + cfg.TRAIN.KD_ALPHA * teacher + cfg.TRAIN.SIDE_CLS_ALPHA * side)
                expected = .5 * torch.log1p(supervised) + .5 * torch.log1p(cfg.TRAIN.KD_ALPHA * leaf)
                torch.testing.assert_close(result['loss'], expected)
                derivative, = torch.autograd.grad(result['loss'], leaf, retain_graph=True)
                self.assertTrue(bool(torch.isfinite(derivative)))
                self.assertLessEqual(float(derivative), cfg.TRAIN.KD_ALPHA)
                self.assertAlmostEqual(float(derivative), model.dto_last_metrics['distill_coefficient'], places=7)
                # Each CE keeps its released ratio relative to DCM inside S.
                self.assertGreater(model.dto_last_metrics['auxiliary_coefficient'], 0.)

    def test_safe_parameter_group_probes_report_the_actual_weighted_gradient(self):
        cfg, model, inputs = self.fixture('E1_safe')
        result = model(*inputs)
        rows = diagnose(model, cfg, inputs[2])
        teacher = next(row for row in rows if row['parameter_group'] == 'teacher_adapter')
        self.assertEqual(teacher['raw_distill_norm'], 0.)
        self.assertGreater(teacher['auxiliary_distill_norm'], 0.)
        self.assertFalse(teacher['both_objective_gradients_nonzero'])
        task, kl, student, teacher_ce, side = model._components(inputs[2])
        params = [p for p in model.parameters() if p.requires_grad]
        auxiliary = cfg.TRAIN.CLS_ALPHA * student + cfg.TRAIN.KD_ALPHA * teacher_ce + cfg.TRAIN.SIDE_CLS_ALPHA * side
        m = model.dto_last_metrics
        reconstructed = m['task_coefficient'] * task + m['distill_coefficient'] * kl + m['auxiliary_coefficient'] * auxiliary
        actual = torch.autograd.grad(result['loss'], params, retain_graph=True, allow_unused=True)
        expected = torch.autograd.grad(reconstructed, params, allow_unused=True)
        for a, b in zip(actual, expected):
            if a is None:
                self.assertIsNone(b)
            else:
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        self.assertEqual(set(model.dto_response_metrics), {
            n + '_' + s for n in ('student', 'teacher')
            for s in ('entropy', 'max_probability', 'global_accuracy')})

    def test_full_state_resume_remains_exact_for_safe_adaptation(self):
        from tests import test_rigorous_protocol_cpu as suite
        original = suite.configuration

        def safe_configuration(*args, **kwargs):
            cfg = original(*args, **kwargs)
            cfg.DTO_KD.MODE = 'safe_dynamic'
            cfg.DTO_KD.DETACH_TEACHER = True
            return cfg

        with mock.patch.object(suite, 'configuration', side_effect=safe_configuration):
            suite.RigorousTrainingTest().check_full_resume(workers=0)


class AdmissionAndStoppingTest(unittest.TestCase):
    def test_legacy_certificate_cannot_admit_new_path(self):
        for mode in ('off', 'dynamic', 'safe_dynamic'):
            with self.assertRaises(RuntimeError):
                require_adaptation_certificate({'passed': True}, mode, True)
        good = {'smokes': [{'mode': 'safe_dynamic', 'detach_teacher': True,
                            'adaptation_check': {'passed': True}, 'successful_updates': 3,
                            'schedule_check': {'verified': True}}]}
        require_adaptation_certificate(good, 'safe_dynamic', True)
        with self.assertRaises(RuntimeError):
            require_adaptation_certificate(good, 'dynamic', True)

    def test_new_queue_variants_are_explicit_and_use_independent_configs(self):
        old = build_jobs([1, 5], 0, Path('/unused'), set(), protocol='v2')
        new = build_jobs([1, 5], 0, Path('/unused'), set(), protocol='v2', include_adaptations=True)
        self.assertEqual(len(old), 8)
        self.assertEqual(len(new), 14)
        for job in new[8:]:
            cfg = yaml.safe_load(Path(job['config']).read_text())
            self.assertTrue(cfg['DTO_KD']['DETACH_TEACHER'])

    def test_stop_refuses_reused_pid_and_wrong_session_without_signalling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / 'proc' / '12345'
            proc.mkdir(parents=True)
            (proc / 'cwd').symlink_to(root)
            output = root / 'E1_1shot'
            job = {'pid': 12345, 'name': 'E1_1shot', 'output': str(output), 'gpu': 2}
            args = ['python', '-m', 'experiments.dto_kd.rigorous', '--cfg', str(output / 'config.yaml')]
            (proc / 'cmdline').write_bytes('\0'.join(args).encode())
            with mock.patch('os.getpgid', return_value=12345), mock.patch('os.getsid', return_value=12345):
                self.assertEqual(inspect_job(job, root / 'proc')['pid'], 12345)
                wrong = dict(job, output=str(root / 'E0_1shot'))
                with self.assertRaises(RuntimeError):
                    inspect_job(wrong, root / 'proc')
            with mock.patch('os.getpgid', return_value=123):
                with self.assertRaises(RuntimeError):
                    inspect_job(job, root / 'proc')
            self.assertIsNone(inspect_job(dict(job, pid=54321), root / 'proc'))


if __name__ == '__main__':
    unittest.main()
