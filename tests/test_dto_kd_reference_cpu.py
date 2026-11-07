"""Paper equations and genuine TS-FSAR loss paths, synthetic CPU features only."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from experiments.dto_kd.exact import log_loss, shared_lsn_weights, solve_gram
from experiments.dto_kd.diagnostics import diagnose, parameter_group
from experiments.dto_kd.acceptance import require_adaptation_certificate
from experiments.dto_kd.rigorous import train
from experiments.dto_kd.episodes import loader
from datasets.base_dataset import BaseVideoDataset
from scripts.launch_campus_suite import build_jobs
from tests import test_dto_kd_adaptation_cpu as adaptation
from tests import test_rigorous_protocol_cpu as fixtures
from utils.functions import cosine_lr
from utils.optimizer import construct_optimizer


class ExactSolverTest(unittest.TestCase):
    def test_real_lsn_names_and_shared_graph_filter_match_direct_log_gradients(self):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        lsn = model.model
        lsn.side_downsample = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        lsn.side_transformer = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        lsn.side_LN = torch.nn.ModuleList([torch.nn.LayerNorm(2)])
        lsn.side_gate_params = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))])
        lsn.side_last_LN = torch.nn.LayerNorm(2)
        lsn.side_proj_downsample = torch.nn.Linear(2, 2)
        lsn.side_post_LN = torch.nn.LayerNorm(2)
        lsn.side_reduce = torch.nn.Linear(2, 2)
        shared = list(model.named_parameters())
        lsn.side_frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        lsn.side_distill_only = torch.nn.Parameter(torch.ones(2))
        lsn.side_task_only = torch.nn.Parameter(torch.ones(2))
        lsn.side_unused = torch.nn.Parameter(torch.ones(2))
        # These are graph-connected but outside the shared LSN scope.
        lsn.sideways = torch.nn.Linear(2, 2)
        lsn.Adapter = torch.nn.Linear(2, 2)
        lsn.Linear = torch.nn.Linear(2, 2)
        lsn.up_linear = torch.nn.Linear(2, 2)
        model.ts_dcm = torch.nn.Module()
        model.ts_dcm.generator = torch.nn.Linear(2, 2)
        private = [p for n, p in model.named_parameters()
                   if n.startswith(('model.sideways.', 'model.Adapter.', 'model.Linear.',
                                    'model.up_linear.', 'ts_dcm.'))]
        # Include a graph-connected numerical zero gradient; it must be retained.
        a = sum((i * p).sum() for i, (_, p) in enumerate(shared))
        b = sum(((i % 3 - 1) * p).sum() for i, (_, p) in enumerate(shared))
        extra = sum(p.sum() for p in private) + lsn.side_frozen.sum()
        d = (a * .01 + extra * .01 + lsn.side_distill_only.sum() * .01).exp()
        t = (b * .02 + extra * .01 + lsn.side_task_only.sum() * .01).exp()
        params = [p for _, p in shared]
        gd = torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(d.log(), params, retain_graph=True)])
        gt = torch.cat([g.reshape(-1).double() for g in torch.autograd.grad(t.log(), params, retain_graph=True)])
        expected = solve_gram(float(gd.dot(gd)), float(gd.dot(gt)), float(gt.dot(gt)),
                              float((gd-gt).square().sum()))
        result = shared_lsn_weights(model, d, t, 1e-8)
        self.assertEqual(result['solver_parameter_names'], [n for n, _ in shared])
        self.assertEqual(result['solver_numel'], sum(p.numel() for p in params))
        self.assertAlmostEqual(result['pi_distill'], expected['pi_distill'], places=6)
        self.assertTrue(result['pair_kkt_verified'])
        self.assertTrue(all(parameter_group(n) == 'lsn' for n, _ in shared))
        self.assertEqual(parameter_group('model.sideways.weight'), 'other')
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_simplex_solver_matches_dense_search_and_kkt_at_extreme_scales(self):
        pairs = [([2., 0.], [-1., 1.]), ([1., 0.], [2., 0.]),
                 ([2., 0.], [1., 0.]), ([1., 0.], [-1., 0.]),
                 ([0., 0.], [1., 0.]), ([1., 2.], [1., 2.])]
        for scale in (1e-20, 1., 1e20):
            for a, b in pairs:
                with self.subTest(scale=scale, a=a, b=b):
                    a, b = torch.tensor(a, dtype=torch.float64) * scale, torch.tensor(b, dtype=torch.float64) * scale
                    row = solve_gram(float(a.dot(a)), float(a.dot(b)), float(b.dot(b)),
                                     float((a-b).square().sum()))
                    p = row['pi_distill']
                    actual = ((p*a + (1-p)*b) / scale).square().sum()
                    grid = torch.linspace(0, 1, 10001, dtype=torch.float64).unsqueeze(1)
                    best_grid = ((grid*a + (1-grid)*b) / scale).square().sum(1).min()
                    self.assertLessEqual(float(actual), float(best_grid) + 1e-12)
                    self.assertTrue(row['pair_kkt_verified'])

    def test_shared_solver_uses_actual_log_gradients_without_rng_or_grad_mutation(self):
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.side = torch.nn.Linear(2, 1, bias=False).double()
        model.private = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        with torch.no_grad():
            model.model.side.weight.zero_()
        theta = model.model.side.weight.reshape(-1)
        d = (2 * theta[0]).exp() * model.private
        t = (-theta[0] + theta[1]).exp()
        model.model.side.weight.grad = torch.ones_like(model.model.side.weight)
        rng = torch.get_rng_state()
        result = shared_lsn_weights(model, d, t, 1e-8)
        self.assertAlmostEqual(result['pi_distill'], .4, places=7)
        self.assertEqual(result['solver_numel'], 2)
        self.assertEqual(result['solver_parameter_names'], ['model.side.weight'])
        torch.testing.assert_close(model.model.side.weight.grad, torch.ones(1, 2, dtype=torch.float64))
        self.assertIsNone(model.private.grad)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_numerical_floor_reports_zero_gradient_stationarity(self):
        result = solve_gram(0., 0., 1., 1.)
        self.assertEqual(result['pi_distill'], 1.)
        self.assertTrue(result['pair_stationary'])
        leaf = torch.tensor(1e-12, requires_grad=True)
        derivative, = torch.autograd.grad(log_loss(leaf, 1e-8), leaf)
        self.assertEqual(float(derivative), 0.)
        with self.assertRaises(FloatingPointError):
            log_loss(torch.tensor(-1.), 1e-8)
        with self.assertRaises(FloatingPointError):
            solve_gram(float('inf'), 0., 1., 1.)


class ReferenceLossTest(unittest.TestCase):
    def test_both_reference_variants_keep_auxiliary_ce_and_stop_only_kl_target(self):
        for shot in (1, 5):
            for variant in ('E1_exact', 'E1_amortized'):
                with self.subTest(shot=shot, variant=variant):
                    cfg, model, inputs = adaptation.AdaptationGradientTest().fixture(variant, shot)
                    output = model(*inputs)
                    task, kl, student, teacher, side = model._components(inputs[2])
                    m = model.dto_last_metrics
                    auxiliary = cfg.TRAIN.CLS_ALPHA * student + cfg.TRAIN.KD_ALPHA * teacher + cfg.TRAIN.SIDE_CLS_ALPHA * side
                    expected = auxiliary + m['pi_distill'] * log_loss(kl, cfg.DTO_KD.EPS) + m['pi_task'] * log_loss(task, cfg.DTO_KD.EPS)
                    torch.testing.assert_close(output['loss'], expected)
                    params = list(model.model.Adapter.parameters())
                    self.assertTrue(all(g is None for g in torch.autograd.grad(kl, params, retain_graph=True, allow_unused=True)))
                    ce = torch.autograd.grad(teacher, params, retain_graph=True)
                    total = torch.autograd.grad(output['loss'], params, retain_graph=True)
                    for a, b in zip(total, ce):
                        torch.testing.assert_close(a, cfg.TRAIN.KD_ALPHA * b, rtol=1e-5, atol=1e-6)
                    self.assertTrue(any(float(g.norm()) > 0 for g in ce))
                    reconstructed = auxiliary + m['task_coefficient'] * task + m['distill_coefficient'] * kl
                    params = [p for p in model.parameters() if p.requires_grad]
                    actual = torch.autograd.grad(output['loss'], params, retain_graph=True, allow_unused=True)
                    expected_grads = torch.autograd.grad(reconstructed, params, retain_graph=True, allow_unused=True)
                    for a, b in zip(actual, expected_grads):
                        if a is None:
                            self.assertIsNone(b)
                        else:
                            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                    rows = diagnose(model, cfg, inputs[2])
                    shared = next(r for r in rows if r['parameter_group'] == 'lsn')
                    self.assertIn('total_gradient_alignment', shared)
                    if variant == 'E1_exact':
                        self.assertTrue(m['pair_kkt_verified'])
                        # The pair's descent property is checked separately from
                        # auxiliary CE: no claim about complete AdamW alignment.
                        self.assertGreaterEqual(shared['pair_gradient_alignment']['dot_raw_kl'], -1e-5)
                        self.assertGreaterEqual(shared['pair_gradient_alignment']['dot_raw_dcm'], -1e-5)
                        self.assertIsNone(model.dto_controller)
                    else:
                        logs = torch.tensor([float(kl.detach()), float(task.detach())], dtype=torch.float64).log()
                        p = torch.tensor([.5, .5], dtype=torch.float64)
                        expected_pi = (p - cfg.DTO_KD.ETA * p.dot(logs) * logs).softmax(0)
                        torch.testing.assert_close(torch.tensor(model.dto_controller.weights).double(), expected_pi,
                                                   rtol=1e-6, atol=1e-7)

    def test_exact_reference_runs_common_training_loop_and_saves_without_controller(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(BaseVideoDataset, '_read_video', fixtures.read_fake_video):
            cfg = fixtures.configuration(Path(directory))
            cfg.DTO_KD.MODE, cfg.DTO_KD.DETACH_TEACHER = 'paper_exact', True
            model = fixtures.RigorousTrainingTest().make_model(cfg)
            optimizer = construct_optimizer(model, cfg)
            embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
            result = train(loader(cfg, 'train'), loader(cfg, 'test', 'val'), model, embeddings, embeddings,
                           optimizer, cosine_lr(optimizer, 0, 4), torch.cuda.amp.GradScaler(enabled=False),
                           cfg, torch.device('cpu'), {'configuration_signature': 'exact', 'source_sha256': 'cpu'})
            self.assertEqual(result['successful_updates'], 4)
            checkpoint = torch.load(Path(directory) / 'resume_latest.pth', map_location='cpu')
            self.assertIsNone(checkpoint['controller'])
            rows = [json.loads(s) for s in (Path(directory) / 'train_trace.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['dto']['pair_kkt_verified'] for r in rows))

    def test_reference_certificates_and_queue_paths_are_explicit(self):
        for mode in ('paper_exact', 'paper_amortized'):
            with self.assertRaises(RuntimeError):
                require_adaptation_certificate({'passed': True}, mode, True)
            good = {'smokes': [{'mode': mode, 'detach_teacher': True,
                'adaptation_check': {'passed': True, 'original_auxiliary_ce_coefficients': True,
                                     'shared_lsn_pair_kkt_every_episode': True},
                'schedule_check': {'verified': True}, 'successful_updates': 3}]}
            require_adaptation_certificate(good, mode, True)
            with self.assertRaises(RuntimeError):
                require_adaptation_certificate(good, mode, False)
        jobs = build_jobs([1, 5], 0, Path('/fresh'), set(), 'v2', include_references=True)
        reference = [j for j in jobs if j['experiment'] in ('E1_exact', 'E1_amortized')]
        self.assertEqual(len(reference), 4)
        self.assertTrue(all(Path(j['config']).is_file() for j in reference))


class ReferenceDriverTest(unittest.TestCase):
    def previous(self, root):
        previous = root / 'previous'
        for shot in (1, 5):
            directory = previous / f'shot{shot}'
            directory.mkdir(parents=True)
            bank = directory / 'initialization.pth'
            bank.write_bytes(b'initial-model-not-a-trained-checkpoint')
            (directory / 'acceptance.json').write_text(json.dumps({
                'passed': True, 'shot': shot, 'seed': 0,
                'initialization_file': bank.name,
                'initialization_file_sha256': hashlib.sha256(bank.read_bytes()).hexdigest()}))
        return previous

    def test_even_train_plan_is_file_only_until_apply(self):
        from scripts import run_campus_dto_reference as driver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = self.previous(root)
            args = ['reference', '--previous-acceptance-root', str(previous),
                    '--output-root', str(root / 'new'), '--train',
                    '--experiments', 'E0,E0_detach,E1_amortized,E1_exact']
            with mock.patch.object(driver.sys, 'argv', args), \
                    mock.patch.object(driver.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 40*1024**3})()), \
                    mock.patch.object(driver.subprocess, 'Popen') as launch, \
                    mock.patch.object(driver.subprocess, 'check_output') as query:
                driver.main()
                launch.assert_not_called()
                query.assert_not_called()
                self.assertFalse((root / 'new').exists())

    def test_busy_gpu_aborts_before_any_acceptance_or_output(self):
        from scripts import run_campus_dto_reference as driver
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = self.previous(root)
            args = ['reference', '--previous-acceptance-root', str(previous),
                    '--output-root', str(root / 'new'), '--apply']
            with mock.patch.object(driver.sys, 'argv', args), \
                    mock.patch.object(driver.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 40*1024**3})()), \
                    mock.patch.object(driver.subprocess, 'check_output', return_value='0'), \
                    mock.patch.object(driver, 'gpu_idle', return_value=False), \
                    mock.patch.object(driver.subprocess, 'Popen') as launch:
                with self.assertRaises(SystemExit):
                    driver.main()
                launch.assert_not_called()
                self.assertFalse((root / 'new').exists())


if __name__ == '__main__':
    unittest.main()
