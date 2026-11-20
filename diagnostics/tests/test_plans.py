"""Validate the actual archived filename-based plans, including permuted rows."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('suite_saved_loop_plan',ROOT/'tools/saved_loop_plan.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Plans(unittest.TestCase):
    def test_exact_archived_plans(self):
        expected={'0bfdd020cf475b9c':([17,35,52,70,87,105,122,140],[0,37,57,79,101,121],[6,46,72,91,107,124]),
                  'ff592398657b3dfe':([23,46,70,93,116,139,163,186],[0,36,71,105,140,175],[3,29,56,104,155,184])}
        for path in (ROOT/'diagnostics/plans').glob('*.json'):
            saved=json.loads(path.read_text());ids=np.array(saved['source_frame_ids_chronological'])
            with self.subTest(plan=path.name):
                result=module.load_saved_loop_plan(saved,ids,saved['scene_id'])
                targets,trajectory,fps=expected[path.stem[:16]]
                self.assertEqual(result['target_indices'],targets)
                self.assertEqual(result['methods']['trajectory']['primary'],trajectory)
                self.assertEqual(result['methods']['fps']['primary'],fps)
    def test_mapping_survives_reversed_dataset(self):
        for path in (ROOT/'diagnostics/plans').glob('*.json'):
            saved=json.loads(path.read_text());ids=np.array(saved['source_frame_ids_chronological'])[::-1]
            result=module.load_saved_loop_plan(saved,ids,saved['scene_id'])
            for method in result['methods'].values():
                self.assertEqual([int(ids[r]) for r in method['primary']],
                                 [int(f[6:-4]) for f in method['primary_files']])


if __name__=='__main__':unittest.main()
