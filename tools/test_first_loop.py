import unittest
from pathlib import Path
import runpy
import numpy as np
m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/dataset/view_sampler/first_loop.py'))


def circle(turns=2,steps=240):
    a=np.linspace(0,2*np.pi*turns,steps*turns+1)
    return np.column_stack((np.cos(a),np.sin(a),np.zeros(len(a))))


class FirstLoopTests(unittest.TestCase):
    def test_first_of_two_laps(self):
        self.assertEqual(m['detect_first_loop'](circle())['end_row'],240)
    def test_closed_selection_avoids_closing_duplicate(self):
        p=m['closed_trajectory_indices'](circle(),240,6)
        np.testing.assert_array_equal(p,[0,40,80,120,160,200])
    def test_unique_supports_inside_loop(self):
        xyz=circle(); p=m['closed_trajectory_indices'](xyz,240,6)
        s=m['closed_secondary_indices'](xyz,240,p)
        self.assertEqual(len(np.unique(np.r_[p,s.ravel()])),18)
        self.assertTrue(np.all(s<240))
        self.assertGreater(s[0,0],200)  # wrap-around support at start
    def test_scale_translation_rotation_invariance(self):
        xyz=circle(); r=np.array([[0,0,1],[1,0,0],[0,1,0]])
        self.assertEqual(m['detect_first_loop']((xyz@r)*71+[8,4,20])['end_row'],240)
    def test_straight_out_and_back_rejected(self):
        x=np.r_[np.linspace(0,5,100),np.linspace(5,0,100)]
        with self.assertRaises(ValueError): m['detect_first_loop'](np.column_stack((x,x*0,x*0)))
    def test_open_path_rejected(self):
        with self.assertRaises(ValueError): m['detect_first_loop'](circle()[:120])
    def test_stationary_rejected(self):
        with self.assertRaises(ValueError): m['detect_first_loop'](np.zeros((50,3)))
    def test_nearly_complete_loop(self):
        xyz=circle()[:235]
        self.assertEqual(m['detect_first_loop'](xyz)['end_row'],234)
    def test_targets_excluded_from_both_roles(self):
        xyz=circle(); targets=np.array([40,80,200])
        p=m['closed_trajectory_indices'](xyz,240,6,targets)
        s=m['closed_secondary_indices'](xyz,240,p,excluded_indices=targets)
        self.assertEqual(np.intersect1d(np.r_[p,s.ravel()],targets).size,0)
    def test_rotated_return_rejected_when_requested(self):
        xyz=circle(turns=1); a=np.linspace(0,np.pi,len(xyz)); r=np.zeros((len(xyz),3,3))
        r[:,0,0]=np.cos(a);r[:,0,1]=-np.sin(a);r[:,1,0]=np.sin(a);r[:,1,1]=np.cos(a);r[:,2,2]=1
        with self.assertRaises(ValueError):m['detect_first_loop'](xyz,rotations=r,max_rotation_deg=30)


if __name__=='__main__':unittest.main()
