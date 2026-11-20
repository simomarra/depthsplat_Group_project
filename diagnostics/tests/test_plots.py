"""Exercise all offline plotting entrypoints with tiny synthetic temporary data."""
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from diagnostics.conditions import MAIN_CONDITIONS
from diagnostics.io import write_csv
from diagnostics.tests.test_suite import inputs


class PlotSmoke(unittest.TestCase):
    def test_offline_entrypoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); exported=root/'export'; run=root/'run'
            a,b=inputs()
            for suffix,data in (('primary_only',a),('support',b)):
                folder=exported/('trajectory_'+suffix);folder.mkdir(parents=True)
                np.savez_compressed(folder/'diagnostic_inputs.npz',**data)
            rows=[]
            for name in MAIN_CONDITIONS:
                for frame in (100,101):
                    for folder in (run/name/'rendered',run/'ground_truth'):
                        folder.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(np.zeros((8,8,3),np.uint8)).save(folder/f'{frame:06d}.png')
                    rows.append(dict(scene='synthetic',method='trajectory',condition=name,target=frame,
                                     psnr=20.,ssim=.8,lpips=.1))
            write_csv(run/'per_target.csv',rows)
            data=dict(primary_indices=np.arange(6),rgb=np.zeros((6,3,8,8)),
                      baseline_confidence=np.full((6,4,4),.25),secondary_confidence=np.full((6,4,4),.5),
                      baseline_mv_level0=np.ones((6,3,4,4)),secondary_mv_level0=np.ones((6,3,4,4))*2)
            np.savez_compressed(run/'inspection.npz',**data)
            commands=[('diagnostics.reproject',['--run',str(exported),'--method','trajectory'],'reprojection'),
                      ('diagnostics.inspect',['--run',str(run)],'inspection'),
                      ('diagnostics.plot_renderings',['--run',str(run)],'renderings')]
            for module,args,folder in commands:
                destination=root/folder
                proc=subprocess.run([sys.executable,'-m',module,*args,'--out',str(destination)],
                                    capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                self.assertTrue(list(destination.glob('*.png')))
                self.assertTrue(list(destination.glob('*.csv')))


if __name__=='__main__':unittest.main()
