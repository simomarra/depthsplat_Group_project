"""Matched crops, common-scale RGB errors, per-target metrics and scene tables."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .conditions import MAIN_CONDITIONS
from .io import write_csv, write_json


def rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB'),np.float32)/255


def crop_box(value, h, w):
    if value is None:
        return (w//4,h//4,max(1,w//2),max(1,h//2))
    x,y,cw,ch=value
    if min(x,y)<0 or min(cw,ch)<=0 or x+cw>w or y+ch>h:
        raise ValueError('Crop must lie inside the saved image: x y width height.')
    return x,y,cw,ch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--targets',nargs='+',type=int)
    p.add_argument('--all-figures',action='store_true')
    p.add_argument('--crop',nargs=4,type=int,metavar=('X','Y','WIDTH','HEIGHT'))
    p.add_argument('--error-max',type=float,default=.25)
    args=p.parse_args()
    if args.error_max<=0:raise ValueError('Positive error scale required.')
    with (args.run/'per_target.csv').open() as f:rows=list(csv.DictReader(f))
    available={r['condition'] for r in rows}
    if not set(MAIN_CONDITIONS)<=available:
        raise ValueError('Rendering panel requires all six main conditions; run diagnostics.run first.')
    targets=sorted({int(r['target']) for r in rows})
    chosen=targets if args.all_figures else (args.targets or targets[:1])
    if not set(chosen)<=set(targets):raise ValueError('Unknown target ID.')
    args.out.mkdir(parents=True,exist_ok=True)
    short={
        'baseline':'Baseline','secondary_depth_only':'Secondary depth',
        'secondary_depth_confidence':'Secondary depth + confidence',
        'secondary_features_only':'Secondary features only',
        'secondary_features_depth':'Secondary features + depth',
        'full_secondary':'Full secondary'}
    for target in chosen:
        gt=rgb(args.run/'ground_truth'/f'{target:06}.png')
        images=[gt]+[rgb(args.run/name/'rendered'/f'{target:06}.png') for name in MAIN_CONDITIONS]
        if any(im.shape!=gt.shape for im in images):raise ValueError('Rendering/GT sizes differ.')
        h,w=gt.shape[:2];x,y,cw,ch=crop_box(args.crop,h,w)
        fig,axes=plt.subplots(3,7,figsize=(23,10),constrained_layout=True)
        labels=['GT']+[short[n] for n in MAIN_CONDITIONS]
        for j,(im,label) in enumerate(zip(images,labels)):
            axes[0,j].imshow(im);axes[0,j].set_title(label,fontsize=9)
            from matplotlib.patches import Rectangle
            axes[0,j].add_patch(Rectangle((x,y),cw,ch,fill=False,edgecolor='yellow',linewidth=1))
            axes[1,j].imshow(im[y:y+ch,x:x+cw]);axes[1,j].set_title('Identical crop',fontsize=9)
            error=np.abs(im-gt).mean(-1)
            hm=axes[2,j].imshow(error,cmap='magma',vmin=0,vmax=args.error_max)
            axes[2,j].set_title('RGB MAE from PNG',fontsize=9)
        for ax in axes.flat:ax.axis('off')
        fig.colorbar(hm,ax=axes[2].tolist(),shrink=.6,extend='max')
        fig.suptitle(f'Target {target} | same cameras, crop and error scale across six conditions')
        fig.savefig(args.out/f'target{target:06}_six_conditions.png',dpi=125);plt.close(fig)
        write_json(args.out/f'target{target:06}_crop.json',{'target':target,'x':x,'y':y,'width':cw,'height':ch})
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    summary=[]
    for condition in MAIN_CONDITIONS:
        subset=sorted([r for r in rows if r['condition']==condition],key=lambda r:int(r['target']))
        if [int(r['target']) for r in subset]!=targets:raise ValueError('Mismatched target lists in scores.')
        record={'scene':subset[0]['scene'],'method':subset[0]['method'],'condition':condition}
        for ax,metric in zip(axes,('psnr','ssim','lpips')):
            values=[float(r[metric]) for r in subset]
            record[metric]=float(np.mean(values))
            ax.plot(range(len(targets)),values,marker='o',markersize=3,label=short[condition])
            ax.set_xticks(range(len(targets)),labels=targets)
            ax.set(xlabel='Held-out target ID',ylabel=metric.upper(),
                   title=f'{metric.upper()} ({"lower" if metric=="lpips" else "higher"} is better)')
            ax.grid(alpha=.2)
        summary.append(record)
    axes[0].legend(fontsize=7)
    fig.savefig(args.out/'per_target_metrics.png',dpi=150);plt.close(fig)
    write_csv(args.out/'scene_averages.csv',summary)
    lines=['| Condition | PSNR ↑ | SSIM ↑ | LPIPS ↓ |','|---|---:|---:|---:|']
    lines += [f"| {r['condition']} | {r['psnr']:.5f} | {r['ssim']:.5f} | {r['lpips']:.5f} |" for r in summary]
    (args.out/'scene_averages.md').write_text('\n'.join(lines)+'\n')
    print('Wrote:',args.out.resolve())


if __name__=='__main__':main()
