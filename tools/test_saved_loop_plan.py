import copy
import json
from pathlib import Path
import unittest
import numpy as np
from saved_loop_plan import load_saved_loop_plan

BASE=Path(__file__).resolve().parents[1]


class SavedLoopPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((BASE/'assets/first_loop_scene_selection.json').read_text())
        self.ids=np.array(self.plan['source_frame_ids_chronological'],dtype=np.int64)
    def load(self,p=None,ids=None):
        return load_saved_loop_plan(self.plan if p is None else p,self.ids if ids is None else ids,self.plan['scene_id'])
    def test_exact_selection(self):
        result=self.load()
        self.assertEqual(result['methods']['trajectory']['primary'],[0,27,48,69,89,108])
        self.assertEqual(result['methods']['fps']['primary'],[27,57,72,86,101,122])
        self.assertEqual(result['target_indices'],[15,29,44,58,73,87,102,116])
    def test_permuted_dataset_rows(self):
        ids=self.ids[::-1].copy(); result=self.load(ids=ids)
        for name,method in result['methods'].items():
            for row,filename in zip(method['primary'],method['primary_files']):
                self.assertEqual(ids[row],int(filename[6:-4]))
        self.assertEqual(result['target_indices'][0],332)
    def test_frame_gap_preserved(self):
        self.assertEqual(self.ids[284],290)
        self.assertEqual(self.load()['frame_count'],348)
    def test_missing_frame_rejected(self):
        with self.assertRaises(ValueError):self.load(ids=self.ids[:-1])
    def test_wrong_filename_rejected(self):
        self.plan['methods']['Uniform distance within loop']['primary_files'][0]='frame_00002.png'
        with self.assertRaises(ValueError):self.load()
    def test_secondary_target_overlap_rejected(self):
        m=self.plan['methods']['Uniform distance within loop']
        m['secondary_rows'][0][0]=15;m['secondary_files'][0][0]='frame_00016.png'
        with self.assertRaises(ValueError):self.load()
    def test_wrong_scene_rejected(self):
        with self.assertRaises(ValueError):load_saved_loop_plan(self.plan,self.ids,'wrong')
    def test_closing_row_rejected_as_input(self):
        m=self.plan['methods']['Uniform distance within loop']
        m['primary_rows'][0]=132;m['primary_files'][0]='frame_00133.png'
        with self.assertRaises(ValueError):self.load()


if __name__=='__main__':unittest.main()
