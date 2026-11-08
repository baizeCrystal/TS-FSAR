import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from research.cls_ctm_experiment.common import atomic_save, identity_file, output_lock
from research.cls_ctm_experiment.launch import allocations, check_gpu
from research.cls_ctm_experiment.train import preserve_prefix
from research.cls_ctm_experiment.analyze import summarize
from research.ps_ds_subspace_experiment.protocol import sha_file


class IOTests(unittest.TestCase):
    def test_resume_preserves_post_checkpoint_evidence_and_rejects_missing_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'train_trace.jsonl'
            original = '{"step":1}\n{"step":2}\n{"step":3}\n{"step":'
            path.write_text(original)
            preserve_prefix(path,2)
            self.assertEqual([json.loads(s)['step'] for s in path.read_text().splitlines()],[1,2])
            backups = list(Path(temp).glob('train_trace.jsonl.before_resume_*'))
            self.assertEqual(len(backups),1)
            self.assertEqual(backups[0].read_text(),original)
            with self.assertRaises(RuntimeError):
                preserve_prefix(path,3)

    def test_output_identity_and_live_lock_prevent_conflicting_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity_file(root/'identity.json',dict(shot=1))
            identity_file(root/'identity.json',dict(shot=1))
            with self.assertRaises(RuntimeError):
                identity_file(root/'identity.json',dict(shot=5))
            with output_lock(root):
                with self.assertRaises(RuntimeError):
                    with output_lock(root):
                        pass

    def test_checkpoint_weights_only_roundtrip_and_unfinished_file_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'resume_latest.pth'
            value = dict(module={'weight':torch.tensor([1.,2.])},completed_steps=500,
                         identity=dict(settings=dict(alpha=.5,beta=.5)))
            atomic_save(value,path)
            loaded = torch.load(path,map_location='cpu',weights_only=True)
            torch.testing.assert_close(loaded['module']['weight'],value['module']['weight'])
            partial = path.with_name(path.name+'.partial')
            partial.write_text('interrupted write')
            with self.assertRaises(RuntimeError):
                atomic_save(value,path)
            self.assertEqual(partial.read_text(),'interrupted write')

    def test_single_shot_allocation_never_requires_other_shot_gpu(self):
        self.assertEqual(allocations('1','4'),[(1,4)])
        self.assertEqual(allocations('5','5'),[(5,5)])
        self.assertEqual(allocations('1,5','4,5'),[(1,4),(5,5)])
        for shots,gpus in [('1','4,5'),('1,5','4'),('1,5','4,4'),
                           ('1,1','4,5'),('2','4'),('1','-1'),('','4')]:
            with self.assertRaises(ValueError):
                allocations(shots,gpus)

    def test_launcher_refuses_busy_gpu_without_stopping_processes(self):
        with patch('subprocess.check_output',return_value='12345\n') as query:
            with self.assertRaises(RuntimeError):
                check_gpu(4)
            self.assertEqual(query.call_count,1)
        with patch('subprocess.check_output',side_effect=['','238']):
            with self.assertRaises(RuntimeError):
                check_gpu(7)

    def test_paired_report_checks_hashes_and_counts_actual_transitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = dict(shot=1,episodes=3,implementation_sha256='fixture',method='CLS-CTM-discriminative',preflight_module=False,
                          method_disabled=False)
            (root/'run_identity.json').write_text(json.dumps(header))
            rows = []
            for ep in range(3):
                target = list(range(5)); e0 = [1,1,2,3,4]; e1 = [0,1,2,3,4]
                row = dict(episode_index=ep,episode_id=str(ep),original_forward_parity=True,
                    target=target,class_order=target,query_video_ids=[str(i) for i in target],
                    e0_seconds=.1,diagnostics=dict(total_additional_seconds=.02,gate_min=.4,
                        gate_max=1.6,gate_mean=1.,gate_std=.1,gate_mean_absolute_deviation_from_one=.1,nearest_gap_mean=.2),
                    scale=dict(e0_logits=dict(std=1.),e1_logits=dict(std=1.)))
                for branch,pred in [('e0_matching',e0),('e0_fusion',e0),('e1_matching',e1),('e1_fusion',e1)]:
                    scores = [[float(i==p) for i in target] for p in pred]
                    row[branch] = dict(scores=scores,prediction=pred,correct=[a==b for a,b in zip(pred,target)])
                rows.append(row)
            path = root/'paired_predictions.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'completion.json').write_text(json.dumps(dict(passed=True,predictions_sha256=sha_file(path))))
            result,_ = summarize(root,5000)
            self.assertEqual(result['matching']['corrected'],3)
            self.assertEqual(result['matching']['introduced'],0)
            self.assertAlmostEqual(result['matching']['delta_pp'],20.)
            self.assertEqual(result['matching']['ci95_pp'],[19.999999999999996]*2)
            with patch('sys.argv',['analyze','--shot1',str(root),
                       '--bootstrap','5000','--output',str(root/'single_analysis')]):
                from research.cls_ctm_experiment.analyze import main
                main()
            self.assertTrue((root/'single_analysis/shot1_statistics.json').is_file())
            self.assertFalse((root/'single_analysis/shot5_statistics.json').exists())
            path.write_text(path.read_text()+'\n')
            with self.assertRaises(ValueError):
                summarize(root,5000)


if __name__ == '__main__':
    unittest.main()
