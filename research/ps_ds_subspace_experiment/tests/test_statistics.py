import copy
import io
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research.ps_ds_subspace_experiment.analyze import bootstrap, paired_statistics, hard_negatives, main
from research.ps_ds_subspace_experiment.protocol import canonical,sha_file,write_new


class PairedStatistics(unittest.TestCase):
    def rows(self):
        result=[]
        for a,b in [([1,0,0,1,0],[1,1,0,0,0]),([0,0,0,0,0],[1,1,1,1,1])]:
            row={'target':list(range(5)),'class_order':list(range(5))}
            for name,correct in [('e0_matching',a),('e1_matching',b)]:
                score=[]
                for q,c in enumerate(correct):
                    s=[0.]*5; s[q if c else (q+1)%5]=1.; score.append(s)
                row[name]={'correct':[bool(v) for v in correct],'scores':score}
            result.append(row)
        return result

    def test_episode_unit_and_transitions(self):
        rows=self.rows(); stats=paired_statistics(rows,'matching',5000)
        self.assertEqual(stats['corrected'],6)
        self.assertEqual(stats['introduced'],1)
        self.assertEqual(stats['both_correct'],1)
        self.assertEqual(stats['both_wrong'],2)
        self.assertEqual(stats['delta_pp'],50.)
        # Two episode deltas are 0 and 1. CI must include 0 and 100 pp;
        # treating the ten queries as independent would give a different CI.
        self.assertEqual(stats['ci95_pp'],[0.,100.])

    def test_zero_delta_reproducible_ci_and_minimum_repetitions(self):
        a=bootstrap(np.zeros(10),5000); b=bootstrap(np.zeros(10),5000)
        self.assertEqual(a,b)
        self.assertEqual(a['ci95_pp'],[0.,0.])
        with self.assertRaises(ValueError):
            bootstrap(np.zeros(10),4999)

    def test_hard_negative_keeps_e0_confuser(self):
        stats=hard_negatives(self.rows(),'matching')
        self.assertEqual(stats['e0_errors'],8)
        self.assertEqual(stats['corrected'],6)
        self.assertEqual(stats['beaten_original_confuser'],6)

    def test_complete_report_pipeline_on_explicit_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for shot in (1,5):
                folder=root/f'shot{shot}';folder.mkdir()
                rows=self.rows()
                for i,r in enumerate(rows):
                    r.update(episode_index=i,episode_id=f'synthetic-{i}',original_forward_parity=True,
                             e0_seconds=.2,query_video_ids=[f'synthetic-q{q}' for q in range(5)],
                             scale={'e0_logits':{'std':.4},'e1_logits':{'std':.4}})
                    for branch in ['e0_matching','e1_matching']:
                        r[branch]['prediction']=np.array(r[branch]['scores']).argmax(1).tolist()
                    r['e0_fusion']=copy.deepcopy(r['e0_matching'])
                    r['e1_fusion']=copy.deepcopy(r['e1_matching'])
                predictions=folder/'paired_predictions.jsonl'
                predictions.write_text(''.join(canonical(r)+'\n' for r in rows))
                write_new(folder/'run_identity.json',dict(episodes=2,shot=shot,method_disabled=True,
                           implementation_sha256='synthetic-code-fixture'))
                write_new(folder/'completion.json',dict(passed=True,predictions_sha256=sha_file(predictions)))
            with patch('sys.argv',['analyze','--shot1',str(root/'shot1'),'--shot5',str(root/'shot5'),
                                  '--output',str(root/'report'),'--bootstrap','5000']), patch('sys.stdout',new=io.StringIO()):
                main()
            self.assertIn('探索结果',(root/'report/experiment_report.md').read_text())
            self.assertEqual(len((root/'report/shot1_prediction_changes.csv').read_text().splitlines()),21)


if __name__=='__main__':
    unittest.main()
