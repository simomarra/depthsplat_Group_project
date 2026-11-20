"""Regenerate the small historical per-target report figure from tracked CSV."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=Path(__file__).parent/'results/legacy/per_target.csv')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with args.csv.open() as f:rows=[r for r in csv.DictReader(f) if r['source_run']=='ablation_uniform_scene1_772082']
    labels={'baseline':'Baseline','secondary_depth_only':'Secondary depth',
            'secondary_depth_confidence':'Secondary depth + confidence',
            'secondary_features_depth':'Secondary features + depth','full_secondary':'Full secondary'}
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for name,label in labels.items():
        subset=sorted([r for r in rows if r['condition']==name],key=lambda r:int(r['target']))
        for ax,metric in zip(axes,('psnr','ssim','lpips')):
            ax.plot([int(r['target']) for r in subset],[float(r[metric]) for r in subset],marker='o',markersize=3,label=label)
            ax.set(xlabel='Held-out target ID',ylabel=metric.upper(),title=metric.upper()+(' (lower better)' if metric=='lpips' else ' (higher better)'))
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Historical scene 1 / uniform primaries: five measured conditions\nFeatures-only with baseline depth was not run in this experiment')
    args.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.out,dpi=130);plt.close(fig)


if __name__=='__main__':main()
