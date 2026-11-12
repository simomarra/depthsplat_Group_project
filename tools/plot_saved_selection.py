"""Plot actual HPC selections: python tools/plot_saved_selection.py --run-dir PATH."""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from compare_primary_selection import plot_positions

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', required=True)
args = parser.parse_args()
from pathlib import Path
root = Path(args.run_dir)
plan = json.loads((root / 'selection.json').read_text())
with np.load(root / 'full_scene_cameras.npz') as data:
    positions = data['extrinsics'][:, :3, 3]
selections = [(name, np.array(method['primary']), np.array(method['secondary']))
              for name, method in plan['methods'].items()]
plot_positions(positions, selections)
plt.show()
