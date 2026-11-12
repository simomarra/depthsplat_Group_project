"""CPU-only checks: python tools/test_selection_protocol.py."""
import unittest
import numpy as np
from selection_protocol import make_plan, trajectory, secondary, fps_numpy


class SelectionProtocol(unittest.TestCase):
    def positions(self):
        t = np.linspace(0, 4 * np.pi, 120)
        return np.column_stack((np.cos(t), np.sin(t), t / 10))

    def test_reproducibility_and_disjointness(self):
        a = make_plan(self.positions())
        self.assertEqual(a, make_plan(self.positions()))
        for method in a['methods'].values():
            p = np.array(method['primary'])
            s = np.array(method['secondary'])
            self.assertEqual(s.shape, (6, 2))
            self.assertEqual(len(np.unique(np.concatenate((p, s.ravel())))), 18)
            self.assertEqual(np.intersect1d(np.concatenate((p,s.ravel())), a['target_indices']).size, 0)

    def test_twelve_primaries(self):
        plan = make_plan(self.positions(), count=12)
        self.assertEqual(plan['num_secondary'], 24)

    def test_fps_original_initialization(self):
        xyz = np.array([[0,0,0],[1,0,0],[3,0,0],[10,0,0]], dtype=np.float32)
        # Mean is 3.5; first chosen camera is x=10, then x=0.
        self.assertEqual(fps_numpy(xyz, 2).tolist(), [0,3])

    def test_path_not_shortcut_when_frames_excluded(self):
        # Travel lengths [0,10,20,21,22]; removing row 1 must not replace
        # that excursion by the direct distance between rows 0 and 2.
        xyz = np.array([[0,0,0],[0,10,0],[0,0,0],[1,0,0],[2,0,0]])
        got = trajectory(xyz, 3, eligible_indices=np.array([0,2,3,4]))
        self.assertEqual(got.tolist(), [0,2,4])

    def test_secondary_exclusion_and_capacity(self):
        x = np.arange(18)
        xyz = np.column_stack((x,x*0,x*0))
        with self.assertRaises(ValueError):
            secondary(xyz, np.array([0,3,6,9,12,17]), excluded_indices=np.array([1]))

    def test_insufficient_frames(self):
        with self.assertRaises(ValueError):
            make_plan(self.positions()[:20], count=6, target_count=8)

    def test_stationary_trajectory(self):
        with self.assertRaises(ValueError):
            make_plan(np.zeros((100,3)))

    def test_endpoint_supports_point_inward(self):
        xyz = np.column_stack((np.arange(100),np.zeros(100),np.zeros(100)))
        p = trajectory(xyz,6)
        s = secondary(xyz,p)
        self.assertTrue(np.all(s[0] > p[0]))
        self.assertTrue(np.all(s[-1] < p[-1]))


if __name__ == '__main__':
    unittest.main()
