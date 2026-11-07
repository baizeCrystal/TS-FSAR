"""Opt-in campus CUDA checks: tiny synthetic tensors, no CLIP/PCA/video loading.

Run only on an allocated campus GPU:
TSFSAR_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 python -m unittest tests.test_dto_kd_cuda -v
"""
import io
import os
import unittest

import torch

from experiments.dto_kd.controller import DTOController
from experiments.dto_kd.gradient_utils import activation_context, loss_gradients
from experiments.dto_kd.acceptance import one_successful_update


@unittest.skipUnless(os.environ.get("TSFSAR_RUN_CUDA_TESTS") == "1",
                     "CUDA checks require explicit campus opt-in")
class DTOControllerCudaTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.fail("CUDA checks requested, but CUDA is unavailable")

    def check_loss_dtype(self, dtype, update_rule='legacy_logits'):
        controller = DTOController(eta=0.03, update_rule=update_rule)
        reference = DTOController(eta=0.03, update_rule=update_rule)
        for d, t in ((4.0, 0.5), (0.25, 2.0), (1.5, 0.75)):
            distill = torch.tensor(d, device="cuda", dtype=dtype, requires_grad=True)
            task = torch.tensor(t, device="cuda", dtype=dtype, requires_grad=True)
            current_weights = controller.weights
            with torch.cuda.amp.autocast():
                objective = controller.weighted_log_objective(distill, task)
                state = controller.update(distill, task)
            reference.update(torch.tensor(d), torch.tensor(t))
            self.assertEqual(objective.device.type, "cuda")
            self.assertEqual(objective.dtype, torch.float32)
            self.assertIsNone(distill.grad)
            self.assertIsNone(task.grad)
            objective.backward()
            self.assertTrue(bool(torch.isfinite(distill.grad)))
            self.assertTrue(bool(torch.isfinite(task.grad)))
            torch.testing.assert_close(distill.grad.float().cpu(),
                                       torch.tensor(current_weights[0] / d), rtol=1e-3, atol=1e-5)
            for got, want in zip(controller.weights, reference.weights):
                self.assertAlmostEqual(got, want, places=6)
            self.assertEqual(controller.state_dict()["logits"].device.type, "cpu")
            self.assertTrue(state.finite)

    def test_cuda_losses_cpu_controller_fp32(self):
        self.check_loss_dtype(torch.float32)

    def test_cuda_losses_cpu_controller_amp_half(self):
        self.check_loss_dtype(torch.float16)

    def test_paper_probability_update_with_cuda_losses(self):
        self.check_loss_dtype(torch.float32, update_rule='paper_pi_softmax')
        self.check_loss_dtype(torch.float16, update_rule='paper_pi_softmax')

    def test_acceptance_backs_off_real_amp_and_compares_one_effective_update(self):
        model = torch.nn.Linear(2, 2, bias=False).cuda()
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        rng = torch.cuda.get_rng_state()
        inputs = torch.ones(2, 2, device='cuda')

        def forward():
            with torch.cuda.amp.autocast():
                # Finite forward, but default-scaled FP16 backward overflows.
                loss = (model(inputs).float() * 1000.).mean()
            return {'loss': loss, 'acc': 0.}

        def run(scale=None, max_attempts=17):
            events = []
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scaler = torch.cuda.amp.GradScaler(**({'init_scale': scale} if scale is not None else {}))
            result = one_successful_update(model, initial, optimizer, scaler, forward,
                                           lambda: torch.cuda.set_rng_state(rng), events.append, max_attempts)
            return result, events

        (gradients, parameters, summary), events = run()
        self.assertGreater(summary['skipped_attempts'], 0)
        self.assertEqual(sum(row['update_applied'] for row in events), 1)
        (grads2, params2, _), events2 = run(summary['successful_scale'], max_attempts=1)
        self.assertEqual(len(events2), 1)
        for name in gradients:
            torch.testing.assert_close(gradients[name], grads2[name], rtol=0, atol=0)
            torch.testing.assert_close(parameters[name], params2[name], rtol=0, atol=0)

    def test_checkpoint_loaded_onto_cuda_restores_cpu_controller(self):
        source = DTOController(eta=0.03)
        source.update(torch.tensor(1.5), torch.tensor(0.75))
        buffer = io.BytesIO()
        torch.save(source.state_dict(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location="cuda", weights_only=True)
        self.assertEqual(state["logits"].device.type, "cuda")
        restored = DTOController()
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict()["logits"].device.type, "cpu")
        source.update(torch.tensor(0.25), torch.tensor(2.0))
        restored.update(torch.tensor(0.25, device="cuda"), torch.tensor(2.0, device="cuda"))
        for got, want in zip(restored.weights, source.weights):
            self.assertAlmostEqual(got, want, places=6)

    def test_safe_adaptation_real_loss_path_stops_target_under_amp(self):
        from tests.test_dto_kd_adaptation_cpu import AdaptationGradientTest
        for shot in (1, 5):
            cfg, model, inputs = AdaptationGradientTest().fixture('E1_safe', shot)
            model.cuda()
            inputs = tuple(value.cuda() for value in inputs)
            with torch.cuda.amp.autocast():
                output = model(*inputs)
            _, kl, _, ce, _ = model._components(inputs[2])
            params = list(model.model.Adapter.parameters())
            self.assertTrue(all(g is None for g in torch.autograd.grad(
                kl, params, allow_unused=True, retain_graph=True)))
            ce_grad = torch.autograd.grad(ce, params, retain_graph=True)
            self.assertTrue(any(float(g.norm()) > 0. for g in ce_grad))
            self.assertLessEqual(model.dto_last_metrics['distill_coefficient'], cfg.TRAIN.KD_ALPHA)
            self.assertEqual(model.dto_controller.state_dict()['probabilities'].device.type, 'cpu')
            output['loss'].backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in params))

    def test_offloaded_diagnostic_step_preserves_warmed_adam_updates(self):
        def run(mode):
            param = torch.nn.Parameter(torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4) / 16)
            optimizer = torch.optim.AdamW([param], lr=0.01)
            controller = DTOController()
            history = []
            for step in range(3):
                with activation_context(mode, sampled=step == 1), torch.cuda.amp.autocast():
                    pred = param @ torch.ones(4, 4, device='cuda')
                    task = (pred.float().square() + 0.5).mean()
                    distill = ((pred.float() - 1).square() + 0.5).mean()
                    loss = controller.weighted_log_objective(distill, task)
                    controller.update(distill, task)
                if step == 1:
                    vectors = loss_gradients({'distill': distill, 'task': task}, [param])
                    self.assertTrue(all(g.device.type == 'cpu' for g in vectors.values()))
                    self.assertIsNone(param.grad)
                    del vectors
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                history.append(param.detach().cpu().clone())
                del loss, task, distill, pred
            return history

        reference = run('off')
        for mode in ('diagnostics', 'all'):
            for got, want in zip(run(mode), reference):
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
