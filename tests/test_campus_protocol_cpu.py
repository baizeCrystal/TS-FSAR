"""Released-recipe guard and actual loss-path checks with a tiny CPU LSN.

These exercise TS_FSAR/DTOTSFSAR and the actual TS_DCM/GLAC losses, but replace
CLIP/PCA/LSN features. They are not full-model CUDA equivalence certification.
"""
import copy
from pathlib import Path
import types
import unittest
from unittest import mock

import torch
import yaml

from experiments.dto_kd.model import DTOTSFSAR
from trainers.ts_fsar import TS_FSAR
from scripts.audit_campus_protocol import ROOT, compare_config


def namespace(value):
    if isinstance(value, dict):
        return types.SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    return value


class TinyLSN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Match the real LSN namespace so integration tests exercise parameter
        # selection used by the exact solver, not an invented model.side tree.
        self.side_downsample = torch.nn.ModuleList([torch.nn.Linear(3, 4)])
        self.Adapter = torch.nn.Linear(4, 4)
        self.up_linear = torch.nn.Linear(4, 4, bias=False)
        self.up_ln = torch.nn.LayerNorm(4)

    def forward(self, images):
        features = self.side_downsample[0](images.mean((-1, -2)))
        offsets = features.new_tensor([0., 0.1, 0.3, 0.7]).reshape(1, 4, 1)
        return features[:, None, :] + offsets, features


class CampusProtocolTest(unittest.TestCase):
    def test_all_configs_retain_released_scientific_recipe(self):
        for path in sorted((ROOT / 'configs/dto_kd').glob('ssv2_small_*.yaml')):
            with self.subTest(config=path.name):
                report = compare_config(path)
                self.assertEqual(report['released_recipe_differences'], {})
                self.assertTrue(all(report['checks'].values()))

    def test_audit_detects_previous_five_shot_learning_rate_error(self):
        import tempfile
        source = ROOT / 'configs/dto_kd/ssv2_small_5shot_E0.yaml'
        cfg = yaml.safe_load(source.read_text())
        cfg['SOLVER']['LSN_LR'] = 0.00025
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.yaml'
            path.write_text(yaml.safe_dump(cfg))
            report = compare_config(path)
        self.assertFalse(report['configuration_checks_pass'])
        self.assertIn('SOLVER.LSN_LR', report['released_recipe_differences'])

    def test_real_loss_paths_and_adam_step_agree_when_objective_is_unchanged(self):
        for shot in (1, 5):
            for mode in ('off', 'static_kl'):
                with self.subTest(shot=shot, mode=mode):
                    self.check_update(shot, mode)

    def check_update(self, shot, mode):
        cfg_dict = yaml.safe_load((ROOT / f'configs/dto_kd/ssv2_small_{shot}shot_E0.yaml').read_text())
        cfg_dict['TRAIN'].update(WAY=2, QUERY_PER_CLASS=1, NUM_CLASSES=3)
        cfg_dict['DATA']['NUM_INPUT_FRAMES'] = 2
        cfg_dict['LSN'].update(EMBED_DIM=4, OUTPUT_DIM=4)
        cfg_dict['DTO_KD'].update(MODE=mode, TRACE_INTERVAL=0)
        cfg = namespace(cfg_dict)
        torch.manual_seed(43)
        with mock.patch('trainers.ts_fsar.LSN', side_effect=lambda *args: TinyLSN()):
            original = TS_FSAR(copy.deepcopy(cfg), torch.nn.Identity())
            adapter = DTOTSFSAR(copy.deepcopy(cfg), torch.nn.Identity())
        adapter.load_state_dict(original.state_dict(), strict=True)
        original.train()
        adapter.train()
        support_labels = torch.arange(2).repeat_interleave(shot)
        target_labels = torch.arange(2)
        labels = torch.cat((support_labels, target_labels))
        images = torch.randn(len(labels) * 2, 3, 2, 2)
        embeddings = torch.nn.functional.normalize(torch.randn(3, 4), dim=1)
        initial_rng = torch.get_rng_state()
        parameters, results, gradients = [], [], []
        for model in (original, adapter):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, eps=1e-6, weight_decay=0.1)
            torch.set_rng_state(initial_rng)
            output = model(images, support_labels, target_labels, labels, embeddings)
            results.append(output['loss'].detach())
            output['loss'].backward()
            gradients.append({name: None if p.grad is None else p.grad.detach().clone()
                              for name, p in model.named_parameters()})
            optimizer.step()
            parameters.append({name: p.detach().clone() for name, p in model.named_parameters()})
        torch.testing.assert_close(results[0], results[1], rtol=1e-6, atol=1e-6)
        self.assertIsNotNone(gradients[1]['model.Adapter.weight'])
        self.assertGreater(gradients[1]['model.Adapter.weight'].norm().item(), 0.)
        for name in gradients[0]:
            left, right = gradients[0][name], gradients[1][name]
            if left is None:
                self.assertIsNone(right)
            else:
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(parameters[0][name], parameters[1][name], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
