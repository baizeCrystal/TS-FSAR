"""Synthetic CPU tests: no CLIP weights, videos, CUDA or training."""
import copy
import unittest
from types import SimpleNamespace as NS
import torch
from research.ps_ds_subspace_experiment.subspace import (
    Settings, svd_subspace, wsd, cayley_descent, fit_templates,
    orth_error, ps_objective, ds_objective)
from research.ps_ds_subspace_experiment.matching import matching, match_from_templates, project


class Numerics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(6)

    def test_svd_equivalence_and_rank_failure(self):
        h = torch.randn(3, 16, 32, dtype=torch.double)
        u, w, rank = svd_subspace(h, 5)
        a, s, _ = torch.linalg.svd(h, full_matrices=False)
        torch.testing.assert_close(u@u.transpose(-1, -2), a[..., :5]@a[..., :5].transpose(-1, -2))
        torch.testing.assert_close(w, s[..., :5]/s[..., :5].sum(-1, keepdim=True))
        self.assertEqual(rank.tolist(), [16]*3)
        with self.assertRaises(ValueError):
            svd_subspace(torch.ones(16, 32), 5)

    def test_wsd_against_direct_paper_sum_and_gradient(self):
        a, wa, _ = svd_subspace(torch.randn(2, 12, 20), 3)
        b, wb, _ = svd_subspace(torch.randn(4, 12, 20), 3)
        got = wsd(a, wa, b, wb)
        for n in range(2):
            for k in range(4):
                term = sum((wa[n,i]*wb[k,j]).sqrt() * torch.dot(a[n,:,i], b[k,:,j])**2
                           for i in range(3) for j in range(3))
                torch.testing.assert_close(got[n,k], (1-term).sqrt())
        self.assertTrue(torch.autograd.gradcheck(lambda x: wsd(x, wa, b, wb), (a.requires_grad_(),)))

    def test_cayley_dense_equivalence_descent_and_constraint(self):
        u = torch.linalg.qr(torch.randn(5, 24, 5, dtype=torch.double))[0]
        z = torch.randn_like(u)
        ut, zt = u.transpose(-1,-2), z.transpose(-1,-2)
        w = z@ut-u@zt-.5*u@(ut@z-zt@u)@ut
        eye = torch.eye(24, dtype=torch.double)
        expected = torch.linalg.solve(eye+.05*w, (eye-.05*w)@u)
        got = cayley_descent(u, z, .1)
        torch.testing.assert_close(got, expected)
        self.assertLess(float(orth_error(got)), 1e-12)
        # Directional derivative of a local linear objective must be negative.
        delta = cayley_descent(u, z, 1e-6)-u
        self.assertLess(float((z*delta).sum()), 0.)

    def test_ps_ds_full_steps_finite_and_order_invariant(self):
        for shot in (1, 5):
            support = torch.randn(5*shot, 8, 24, 16)
            labels = torch.arange(5).repeat_interleave(shot)
            u, report = fit_templates(support, labels, 5)
            self.assertEqual(tuple(u.shape), (5, 24, 5))
            self.assertFalse(u.requires_grad)
            self.assertFalse(report['query_labels_used'])
            self.assertEqual(report['ps_steps'], 50)
            self.assertEqual(report['ds_steps'], 50)
            self.assertLessEqual(report['ps_final'], report['ps_initial']+1e-9)
            self.assertTrue(torch.isfinite(torch.tensor(report['ds_final'])))
            # WSD has a cusp at identical subspaces (notably 1-shot). Fixed-step
            # Cayley SGD is not a line-search solver; do not assume monotonicity.
            if shot == 5:
                self.assertLess(report['ds_final'], report['ds_initial'])
            self.assertLess(report['max_orthogonality_error'], 1e-10)
            # Shuffling support/video order preserves class projection, not SVD signs.
            idx = torch.randperm(5*shot)
            v, _ = fit_templates(support[idx], labels[idx], 5)
            torch.testing.assert_close(u@u.transpose(-1,-2), v@v.transpose(-1,-2), atol=1e-9, rtol=1e-9)

    def test_ds_small_step_descends_away_from_wsd_cusp(self):
        a, wa, _ = svd_subspace(torch.randn(10,24,30),5)
        u, w, _ = svd_subspace(torch.randn(5,24,30),5)
        y = torch.arange(5).repeat_interleave(2)
        u.requires_grad_()
        old = ds_objective(u,w,a,wa,y,1e-12)
        g, = torch.autograd.grad(old,(u,))
        new = ds_objective(cayley_descent(u,g,1e-4),w,a,wa,y,1e-12)
        self.assertLess(float(new),float(old))

    def make_dcm(self):
        from trainers.ts_fsar import TS_DCM
        cfg = NS(DATA=NS(NUM_INPUT_FRAMES=8), TRAIN=NS(DC_ALPHA=.4))
        return TS_DCM(cfg, 256).eval().requires_grad_(False)

    def test_disabled_is_bitwise_e0_and_preserves_parameters(self):
        dcm = self.make_dcm()
        features = torch.randn(80, 197, 256)
        labels = torch.arange(5)
        before = copy.deepcopy(dcm.state_dict())
        with torch.no_grad():
            baseline = dcm(features, labels, 5, 1, 1)[0]
            score, report = matching(dcm, features, labels, 5, 1, 1, enabled=False)
        self.assertTrue(torch.equal(score, baseline))
        for key, value in before.items():
            self.assertTrue(torch.equal(dcm.state_dict()[key], value))
        self.assertFalse(report['enabled'])

    def test_full_channel_identity_projection_matches_original_tsm(self):
        dcm = self.make_dcm()
        for shot in (1, 5):
            features = torch.randn((5*shot+5)*8, 197, 256)
            labels = torch.arange(5).repeat_interleave(shot)
            perm = torch.randperm(5*shot)
            support = features[:5*shot*8].reshape(5*shot,8,197,256)[perm]
            features = torch.cat((support.flatten(0,1), features[5*shot*8:]))
            labels = labels[perm]
            with torch.no_grad():
                baseline = dcm(features, labels, 5, shot, 1)[0]
                score, _ = match_from_templates(dcm, features, labels, 5, shot, 1,
                                                torch.eye(256).repeat(5,1,1))
            torch.testing.assert_close(score, baseline, atol=2e-5, rtol=2e-5)

    def test_actual_256_196_shape_and_query_independence(self):
        import inspect
        self.assertNotIn('query', inspect.signature(fit_templates).parameters)
        dcm = self.make_dcm()
        for shot in (1, 5):
            features = torch.randn((5*shot+5)*8, 197, 256)
            labels = torch.arange(5).repeat_interleave(shot)
            before = copy.deepcopy(dcm.state_dict())
            score, report = matching(dcm, features, labels, 5, shot, 1)
            self.assertEqual(tuple(score.shape), (5,5))
            self.assertTrue(torch.isfinite(score).all())
            self.assertEqual(report['template_shape'], [5,256,5])
            self.assertEqual(report['projected_patch_count'], 196)
            self.assertEqual(report['ps_steps']+report['ds_steps'],100)
            for k, v in before.items():
                self.assertTrue(torch.equal(v,dcm.state_dict()[k]))
            self.assertTrue(all(p.grad is None for p in dcm.parameters()))
            # Alter query features: fitted support bases and losses remain identical.
            changed = features.clone()
            changed[5*shot*8:] *= -2
            _, other = matching(dcm, changed, labels, 5, shot, 1)
            for key in ['ps_initial','ps_final','ds_initial','ds_final','template_singular_weights']:
                self.assertEqual(report[key], other[key])


if __name__ == '__main__':
    unittest.main()
