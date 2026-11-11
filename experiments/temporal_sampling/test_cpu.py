"""Run with unittest. Default suite needs only numpy; model parity needs torch
and the project's already installed model dependencies (TEST_MODEL_PARITY=1).
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from .core import (SCHEMES, cache_key, compare_features, margin, paired_bootstrap,
                   sample_indices, validate_episode, validate_output, write_json)
from .run import HERE, evaluate, load_protocol, parser, sampling_for, verify_historical


def video(n=43, video_id='one', label=0):
    hi = n // 8 * 8 - 1 if n >= 8 else n - 1
    return {'video_id': video_id, 'source_num_frames': n, 'split': 'test',
            'frame_indices': np.rint(np.linspace(0, hi, 8)).astype(int).tolist(),
            'class_id': label, 'episode_label': label}


def episode(shot=1):
    return {'episode': 0,
            'support': [video(video_id=f's{c}_{s}', label=c) for c in range(5) for s in range(shot)],
            'query': [video(video_id=f'q{c}', label=c) for c in range(5)]}


def output(correct=True):
    labels = np.arange(5)
    scores = np.eye(5, dtype=np.float32)
    if not correct:
        scores = np.roll(scores, 1, axis=1)
    out = {branch: scores.copy() for branch in ('ts_dcm', 'adapter_text', 'lsn_text', 'fusion')}
    out.update(labels=labels, R=np.ones((5, 5, 8, 8), dtype=np.float32),
               W=np.ones((5, 5, 8, 8), dtype=np.float32))
    return out


class SamplingTests(unittest.TestCase):
    def test_identity_exact_for_locked_official_indices(self):
        # Non-equidistant historical positions must NOT turn into linspace.
        v = video()
        v['frame_indices'] = [0, 6, 12, 15, 23, 25, 34, 36]
        self.assertEqual(sample_indices(v, 'e0')['indices'], v['frame_indices'])

    def test_gamma_one_uses_original_not_quantized_transform(self):
        for n in range(1, 70):
            v = video(n)
            self.assertEqual(sample_indices(v, 'e0')['indices'], v['frame_indices'])

    def test_monotone_bounded_eight_for_all_conditions(self):
        for n in range(1, 240):
            for s in SCHEMES:
                x = sample_indices(video(n), s)
                self.assertEqual(len(x['indices']), 8)
                self.assertTrue(np.all(np.diff(x['indices']) >= 0))
                self.assertGreaterEqual(min(x['indices']), 0)
                self.assertLessEqual(max(x['indices']), x['eligible_range'][1])

    def test_short_videos_keep_repeats(self):
        for n in (1, 2, 7):
            for scheme in SCHEMES:
                result = sample_indices(video(n), scheme)
                self.assertLessEqual(result['unique_frames'], n)
                self.assertGreaterEqual(result['duplicate_fraction'], 1 - n/8)

    def test_no_trailing_remainder_added(self):
        self.assertEqual(sample_indices(video(43), 'p075')['eligible_range'], [0, 39])

    def test_power_directions(self):
        v = video(128)
        a, b = sample_indices(v, 'p075'), sample_indices(v, 'p133')
        self.assertTrue(np.all(np.array(a['actual_u']) >= a['u']))
        self.assertTrue(np.all(np.array(b['actual_u']) <= b['u']))

    def test_random_is_reproducible_and_seed_sensitive(self):
        v = video(1000)
        self.assertEqual(sample_indices(v, 'r075', 1), sample_indices(v, 'r075', 1))
        self.assertNotEqual(sample_indices(v, 'r075', 1)['indices'], sample_indices(v, 'r075', 2)['indices'])

    def test_input_and_episode_identity_untouched(self):
        ep = episode(5)
        old = copy.deepcopy(ep)
        for scope in ('query', 'support', 'both'):
            samples = sampling_for(ep, 'p075', scope, 1)
            for i, s in enumerate(samples):
                role = 'support' if i < 25 else 'query'
                self.assertEqual(s['scheme'] != 'e0', scope in (role, 'both'))
        self.assertEqual(ep, old)
        self.assertEqual(validate_episode(ep, 5), list(range(5)))

    def test_label_tampering_rejected(self):
        ep = episode()
        ep['query'][0]['class_id'] = 99
        with self.assertRaises(ValueError):
            validate_episode(ep, 1)

    def test_cache_key_separates_sampling_and_weights(self):
        v = video()
        b, a = sample_indices(v), sample_indices(v, 'p075')
        k = cache_key(v, b, {'weights': 'A'})
        self.assertNotEqual(k, cache_key(v, a, {'weights': 'A'}))
        self.assertNotEqual(k, cache_key(v, b, {'weights': 'B'}))
        self.assertEqual(k, cache_key(v, b, {'weights': 'A'}))

    def test_invalid_indices_fail_closed(self):
        for bad in ([0]*7, list(range(7, -1, -1)), [99]*8):
            v = video()
            v['frame_indices'] = bad
            with self.assertRaises(ValueError):
                sample_indices(v)


class StatisticsTests(unittest.TestCase):
    def test_margin_direction(self):
        np.testing.assert_allclose(margin(np.array([[2., 1.], [2., 1.]]), [0, 1]), [1, -1])

    def test_episode_bootstrap_not_query_bootstrap(self):
        b = paired_bootstrap([1, 1, 1, 0], [0, 0, 0, 1], repeats=100)
        self.assertEqual(b['estimate'], .5)
        self.assertEqual(b['episodes'], 2)
        self.assertEqual(b, paired_bootstrap([1, 1, 1, 0], [0, 0, 0, 1], repeats=100))

    def test_known_classification_and_margin_deltas(self):
        from .analyze import summarize
        b, a = output(), output(False)
        r = summarize([(0, b, a), (1, b, a)], repeats=30)
        self.assertEqual(r['fusion']['accuracy_delta']['estimate'], -1)
        self.assertEqual(r['fusion']['accuracy_delta']['ci95'], [-1, -1])
        self.assertEqual(r['fusion']['correct_to_wrong_count'], 10)
        self.assertEqual(r['fusion']['wrong_to_correct_count'], 0)
        self.assertEqual(r['attribution_transition_counts'][3][0], 10)

    def test_time_interpolation_not_index_comparison(self):
        ta, tb = np.linspace(0, 1, 8), np.linspace(0, 1, 8)**.75
        features_a = np.stack([ta, 1-ta], axis=-1)
        features_b = np.stack([tb, 1-tb], axis=-1)
        out = compare_features(features_a, features_b, ta, tb)
        self.assertGreater(out['index_cosine_distance'], .001)
        self.assertAlmostEqual(out['time_interpolated_cosine_distance'], 0, places=12)

    def test_duplicate_times_and_disjoint_coverage(self):
        feature = np.ones((8, 2))
        out = compare_features(feature, feature, np.zeros(8), np.ones(8))
        self.assertIsNone(out['time_interpolated_cosine_distance'])
        self.assertEqual(out['shared_frames'], 0)

    def test_output_structure_and_nonfinite_rejection(self):
        out = output()
        validate_output(out)
        out['R'][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_output(out)

    def test_bundled_protocol_hashes_and_shots(self):
        for shot in (1, 5):
            cfg, eps, ref = load_protocol(HERE / 'data' / f'{shot}shot', 200)
            self.assertEqual(cfg['TRAIN']['SHOT'], shot)
            self.assertEqual(len(eps), len(ref['pixel_hashes']))

    def test_historical_pixel_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            verify_historical(0, {}, 'wrong', {'pixel_hashes': ['original']})

    def test_offline_cli_and_mock_files(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            root = Path(task_tmp)
            run = root / 'run'
            run.mkdir()
            write_json(run / 'status.json', {'status': 'complete'})
            write_json(run / 'e0_verification.json', {'status': 'passed'})
            write_json(run / 'manifest.json', {'schemes': ['e0', 'p075'], 'scopes': ['query']})
            ep = episode()
            (run / 'episodes.jsonl').write_text(json.dumps(ep) + '\n')
            np.savez(run / 'e0_00000.npz', **output())
            np.savez(run / 'p075_query_00000.npz', **output(False))
            write_json(run / 'p075_query_00000.json', {'feature_changes': []})
            result = subprocess.run([sys.executable, '-m', 'experiments.temporal_sampling.analyze',
                '--run', str(run), '--output', str(root/'analysis'), '--repeats', '20'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root/'analysis'/'summary.json').read_text())
            self.assertEqual(summary['conditions']['p075_query']['fusion']['accuracy_delta']['estimate'], -1)

    def test_mock_evaluation_writes_complete_protocol_and_paired_outputs(self):
        from unittest.mock import patch
        class MockEncoder:
            def __init__(self, *args):
                self.identity = {'mock': True}
                self.native_check = {'status': 'mock_only'}
                self.device = 'mock_no_gpu'
            def baseline(self, ep):
                return self.changed(ep['support']+ep['query'], sampling_for(ep, 'e0', 'both', 1), 2, lambda: None), 'pixels'
            def changed(self, videos, samplings, batch, check):
                check()
                return [{k: np.ones((8, 2), dtype=np.float32) * (1 if s['scheme'] == 'e0' else 2)
                         for k in ('clip', 'lsn', 'adc')} for s in samplings]
            def score(self, ep, chunks):
                out = output(all(float(v['clip'][0, 0]) == 1 for v in chunks))
                out['fusion'] = out['fusion'] * .5 + .1
                out['adapter_probability'] = out['adapter_text'] * .5 + .1
                out['lsn_probability'] = out['lsn_text'] * .5 + .1
                out['vision_probability'] = out['ts_dcm'] * .5 + .1
                return out
        with tempfile.TemporaryDirectory() as task_tmp:
            root = Path(task_tmp)
            checkpoint = root / 'mock.pth'
            checkpoint.write_text('mock checkpoint, not model weights')
            args = parser().parse_args(['evaluate', '--protocol', str(root), '--output', str(root/'run'),
                '--cache', str(root/'cache'), '--checkpoint', str(checkpoint), '--video-root', str(root),
                '--limit', '2', '--schemes', 'e0', 'p075', '--scopes', 'query', 'support', 'both'])
            eps = [episode(), episode()]
            eps[1]['episode'] = 1
            base = MockEncoder().score(eps[0], MockEncoder().baseline(eps[0])[0])
            reference = {'pixel_hashes': ['pixels']*2,
                'matching_score': [base['ts_dcm'].tolist()]*2,
                'clip_probability': [base['adapter_probability'].tolist()]*2,
                'side_probability': [base['lsn_probability'].tolist()]*2,
                'full_log_score': [np.log(base['fusion']).tolist()]*2,
                'targets': [base['labels'].tolist()]*2}
            with patch('experiments.temporal_sampling.runtime.Encoder', MockEncoder):
                evaluate(args, {'TEST': {}, 'DATA': {}}, eps, reference)
            status = json.loads((root/'run'/'status.json').read_text())
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['e0_completed'], 2)
            self.assertEqual(status['conditions_completed'], 6)
            self.assertTrue((root/'run'/'e0_verification.json').exists())
            meta = json.loads((root/'run'/'p075_support_00001.json').read_text())
            self.assertTrue(meta['episode_identity_unchanged'])
            with self.assertRaises(ValueError):
                evaluate(args, {'TEST': {}, 'DATA': {}}, eps, reference)


class CacheTests(unittest.TestCase):
    def test_disk_and_memory_budget_and_roundtrip(self):
        from .runtime import FeatureCache
        with tempfile.TemporaryDirectory() as task_tmp:
            root = Path(task_tmp)
            data = {k: np.ones((8, 4), dtype=np.float32) for k in ('clip', 'lsn', 'adc')}
            cache = FeatureCache(root/'cached', {}, memory_mb=1, disk_mb=1)
            cache.put('a', data)
            disk_cache = FeatureCache(root/'cached', {}, memory_mb=1, disk_mb=1)
            for k, array in disk_cache.get('a').items():
                np.testing.assert_array_equal(array, data[k])
            tiny = FeatureCache(root/'no_disk', {}, memory_mb=0, disk_mb=0)
            tiny.put('a', data)
            self.assertIsNone(tiny.get('a'))
            self.assertEqual(list((root/'no_disk').iterdir()), [])


@unittest.skipUnless(os.environ.get('TEST_MODEL_PARITY') == '1', 'enable optional CPU native-equation parity')
class NativeModelParityTests(unittest.TestCase):
    def test_cached_readout_matches_original_model_1_and_5_shot(self):
        import torch
        import torch.nn as nn
        from .frozen_ts_fsar import TS_FSAR, TS_DCM
        from .runtime import Encoder, namespace
        torch.set_num_threads(2)
        torch.manual_seed(42)
        for shot in (1, 5):
            cfg = namespace({'TRAIN': {'WAY': 5, 'SHOT': shot, 'QUERY_PER_CLASS_TEST': 1, 'DC_ALPHA': .4},
                'DATA': {'NUM_INPUT_FRAMES': 8}, 'TEST': {'FUSE_VISION': .4, 'FUSE_CLIP': .2, 'FUSE_SIDE': .4},
                'TEMP_CLIP': 50., 'TEMP_SIDE': 50.})
            n = 5 * shot + 5
            features, clip = torch.randn(n*8, 7, 8), torch.randn(n*8, 16)
            model = TS_FSAR.__new__(TS_FSAR)
            nn.Module.__init__(model)
            model.cfg = cfg
            model.ts_dcm = TS_DCM(cfg, 8)
            model.model = nn.Module()
            model.model.Adapter = nn.Identity()
            model.model.up_ln = nn.LayerNorm(8)
            model.model.up_linear = nn.Linear(8, 16, bias=False)
            model.eval().requires_grad_(False)
            ep = episode(shot)
            # Non-class-major support order verifies label-based prototype means.
            ep['support'] = [ep['support'][i] for i in np.random.default_rng(shot).permutation(5*shot)]
            enc = Encoder.__new__(Encoder)
            enc.torch, enc.cfg, enc.model, enc.device = torch, cfg, model, 'cpu'
            enc.text = torch.randn(5, 16)
            enc.native_check = None
            with torch.inference_mode():
                context = torch.tensor([v['episode_label'] for v in ep['support']])
                _, adc, cls = model.ts_dcm(features, context, 5, shot, 1)
                chunks = [{k: value[i*8:(i+1)*8].numpy() for k, value in {'clip': clip, 'lsn': cls, 'adc': adc}.items()}
                          for i in range(n)]
                enc.verify_native(ep, chunks, features, clip, torch.zeros(n*8, 3, 2, 2))
            self.assertEqual(enc.native_check['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
