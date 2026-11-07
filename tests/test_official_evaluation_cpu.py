"""File-only guards: official evaluation must not inherit the Meta-Val recipe."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts import run_official_evaluation as official


class OfficialEvaluationTest(unittest.TestCase):
    def test_only_runtime_paths_change_for_both_shots(self):
        for shot in (1, 5):
            with self.subTest(shot=shot):
                path = official.ROOT / f'configs/ssv2_small/Metatest_{shot}shot.yaml'
                original = yaml.safe_load(path.read_text())
                expected = copy.deepcopy(original)
                expected['DATA']['DATA_ROOT_DIR'] = '/dataset'
                expected['TEST']['PRETRAIN_MODEL_PATH'] = '/checkpoint.pth'
                expected['OUTPUT_DIR'] = expected['OUTPUT_DIR_SAVE'] = '/evaluation'
                cfg = official.evaluation_config(official.ROOT, shot, Path('/dataset'),
                                                 Path('/checkpoint.pth'), Path('/evaluation'))
                self.assertEqual(cfg, expected)
                self.assertFalse(cfg['TRAIN']['ENABLE'])
                self.assertEqual(cfg['TRAIN']['NUM_TEST_TASKS'], 10000)
                self.assertEqual(len(cfg['TEST']['CLASS_NAME']), 24)
                self.assertEqual(cfg['TEST']['LLM_test_json'], './prompts/Ssv2small_test.json')
                self.assertTrue(cfg['TEST']['LOAD_PRETRAIN'])

    def test_refuses_training_config_in_evaluation_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            path = source / 'configs/ssv2_small/Metatest_1shot.yaml'
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump({'TRAIN': {'ENABLE': True}}))
            with self.assertRaisesRegex(RuntimeError, 'evaluation mode'):
                official.evaluation_config(source, 1, Path('/d'), Path('/c'), Path('/o'))

    def test_source_lock_rejects_modified_or_missing_official_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            evaluator = source / 'main.py'
            evaluator.write_text('original evaluator\n')
            lock = source / 'lock.json'
            lock.write_text(json.dumps({'repository': official.UPSTREAM, 'commit': official.COMMIT,
                                        'files': {'main.py': official.sha256(evaluator)}}))
            with patch.object(official, 'LOCK', lock):
                official.verify_source(source)
                evaluator.write_text('changed evaluator\n')
                with self.assertRaisesRegex(RuntimeError, 'main.py'):
                    official.verify_source(source)
                evaluator.unlink()
                with self.assertRaisesRegex(RuntimeError, 'main.py'):
                    official.verify_source(source)

    def test_lock_covers_actual_evaluation_and_training_selection_dependencies(self):
        lock = json.loads(official.LOCK.read_text())
        self.assertEqual(lock['commit'], official.COMMIT)
        required = {'main.py', 'datasets/builder.py', 'datasets/video_few_shot.py',
                    'datasets/base_dataset.py', 'trainers/model.py', 'trainers/ts_fsar.py',
                    'utils/functions.py', 'configs/base.yaml', 'prompts/Ssv2small_test.json',
                    'configs/ssv2_small/test_few_shot.txt'}
        for shot in (1, 5):
            required.update({f'configs/ssv2_small/Metatest_{shot}shot.yaml',
                             f'configs/ssv2_small/Metatrain_{shot}shot.yaml'})
        self.assertTrue(required <= lock['files'].keys())


if __name__ == '__main__':
    unittest.main()
