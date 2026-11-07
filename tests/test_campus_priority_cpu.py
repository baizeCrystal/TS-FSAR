"""CPU/std-lib checks for the campus E1-only queue transition; never trains."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from scripts import prioritize_campus_e1 as priority


class PriorityQueueTest(unittest.TestCase):
    def fixture(self, directory):
        project = Path(directory) / 'project'
        root = project / 'runs/campus/old'
        (root / 'formal').mkdir(parents=True)
        (root / 'acceptance').mkdir()
        (root / 'launcher.pid').write_text('12345')
        jobs = []
        for experiment in ('E0_detach', 'E1_safe'):
            for shot in (1, 5):
                certificate = root / 'acceptance' / f'shot{shot}.json'
                certificate.write_text(json.dumps({'passed': True, 'shot': shot, 'seed': 0,
                    'smokes': [{'mode': 'safe_dynamic', 'detach_teacher': True,
                                'adaptation_check': {'passed': True}}]}))
                jobs.append({'name': f'{experiment}_{shot}shot', 'experiment': experiment,
                             'shot': shot, 'seed': 0, 'status': 'queued' if experiment == 'E1_safe' else 'running',
                             'gpu': None if experiment == 'E1_safe' else (2 if shot == 1 else 3),
                             'config': str(project / f'configs/dto_kd/ssv2_small_{shot}shot_{experiment}.yaml'),
                             'acceptance_report': str(certificate)})
        status = {'jobs': jobs}
        (root / 'formal/status.json').write_text(json.dumps(status))
        return project, root, project / 'runs/campus/new', status

    def test_exact_launcher_identity_and_output_root_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            project, root, output, _ = self.fixture(directory)
            proc_root = Path(directory) / 'proc'
            proc = proc_root / '12345'
            proc.mkdir(parents=True)
            (proc / 'cwd').symlink_to(project)
            args = ['python', '-u', 'scripts/launch_campus_suite.py', '--output-root', str(root / 'formal')]
            (proc / 'cmdline').write_bytes('\0'.join(args).encode())
            self.assertTrue(priority.validate_launcher(12345, project, root / 'formal', proc_root))
            with self.assertRaises(RuntimeError):
                priority.validate_launcher(12345, project, output / 'formal', proc_root)
            (proc / 'cmdline').write_bytes(b'python\0other-script.py')
            with self.assertRaises(RuntimeError):
                priority.validate_launcher(12345, project, root / 'formal', proc_root)
            self.assertFalse(priority.validate_launcher(54321, project, root / 'formal', proc_root))

    def test_plan_reuses_certificates_and_launches_only_safe_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            project, root, output, status = self.fixture(directory)
            before = (root / 'formal/status.json').read_bytes()
            with mock.patch.object(priority, 'validate_launcher', return_value=True):
                pid, command, _ = priority.plan_transition(project, root, output)
                self.assertEqual(pid, 12345)
                self.assertEqual(command[command.index('--experiments') + 1], 'E1_safe')
                self.assertEqual(command[command.index('--acceptance-1shot') + 1],
                                 status['jobs'][2]['acceptance_report'])
                output.mkdir()
                with self.assertRaises(RuntimeError):
                    priority.plan_transition(project, root, output)
            self.assertEqual(before, (root / 'formal/status.json').read_bytes())

    def test_already_started_e1_or_unrelated_queue_is_not_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            project, root, output, status = self.fixture(directory)
            for change in ('started', 'unrelated'):
                altered = copy.deepcopy(status)
                if change == 'started':
                    altered['jobs'][2]['status'] = 'running'
                else:
                    altered['jobs'][0]['name'] = 'E0_1shot'
                (root / 'formal/status.json').write_text(json.dumps(altered))
                with self.assertRaises(RuntimeError), mock.patch('os.kill') as kill:
                    priority.plan_transition(project, root, output)
                kill.assert_not_called()

    def test_dry_run_never_signals_queries_gpu_or_starts_process(self):
        with tempfile.TemporaryDirectory() as directory:
            project, root, output, _ = self.fixture(directory)
            argv = ['helper', '--project-root', str(project), '--run-root', str(root), '--output-root', str(output)]
            with mock.patch('sys.argv', argv), mock.patch.object(priority, 'validate_launcher', return_value=True), \
                    mock.patch('os.kill') as kill, mock.patch('os.execv') as execute, \
                    mock.patch.object(priority, 'resources_idle') as gpu:
                priority.main()
            kill.assert_not_called()
            execute.assert_not_called()
            gpu.assert_not_called()
            self.assertFalse(output.exists())

    def test_apply_signals_only_verified_parent_and_retains_old_results(self):
        with tempfile.TemporaryDirectory() as directory:
            project, root, output, status = self.fixture(directory)
            marker = root / 'formal/preserved-result.txt'
            marker.write_text('preserve me')
            argv = ['helper', '--project-root', str(project), '--run-root', str(root),
                    '--output-root', str(output), '--apply']
            with mock.patch('sys.argv', argv), \
                    mock.patch.object(priority, 'validate_launcher', side_effect=[True, True, False]), \
                    mock.patch.object(priority, 'resources_idle', return_value=True), \
                    mock.patch('os.kill') as kill, mock.patch('os.execv') as execute, \
                    mock.patch('os.chdir'), mock.patch.dict(os.environ):
                priority.main()
            kill.assert_called_once_with(12345, signal.SIGTERM)
            self.assertEqual(execute.call_args[0][1][execute.call_args[0][1].index('--experiments') + 1], 'E1_safe')
            self.assertEqual(marker.read_text(), 'preserve me')
            self.assertEqual(json.loads((root / 'formal/status.json').read_text()), status)
            self.assertEqual((output / 'acceptance').resolve(), root / 'acceptance')
            self.assertEqual((project / 'runs/campus/latest_adaptation_root.txt').read_text().strip(), str(output))


if __name__ == '__main__':
    unittest.main()
