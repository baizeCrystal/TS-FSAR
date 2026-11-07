"""Opt-in campus CUDA checks: tiny synthetic tensors, no CLIP/PCA/video loading.

Run only on an allocated campus GPU:
TSFSAR_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 python -m unittest tests.test_dto_kd_cuda -v
"""
import io
import os
import unittest

import torch

from experiments.dto_kd.controller import DTOController


@unittest.skipUnless(os.environ.get("TSFSAR_RUN_CUDA_TESTS") == "1",
                     "CUDA checks require explicit campus opt-in")
class DTOControllerCudaTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.fail("CUDA checks requested, but CUDA is unavailable")

    def check_loss_dtype(self, dtype):
        controller = DTOController(eta=0.03)
        reference = DTOController(eta=0.03)
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


if __name__ == "__main__":
    unittest.main()
