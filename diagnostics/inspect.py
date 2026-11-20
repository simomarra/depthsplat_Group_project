"""CPU confidence and feature maps from a completed diagnostics.run output."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .feature_maps import feature_comparison
from .io import load_npz, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--primaries', nargs='+', type=int, help='Default: first primary')
    p.add_argument('--all-figures', action='store_true')
    args = p.parse_args()
    data = load_npz(args.run/'inspection.npz')
    args.out.mkdir(parents=True, exist_ok=True)
    ids = data['primary_indices'].tolist()
    selected = ids if args.all_figures else (args.primaries or ids[:1])
    if not set(selected) <= set(ids): raise ValueError('Unknown primary ID.')
    rgb = data['rgb'].transpose(0,2,3,1)
    a,b = data['baseline_confidence'],data['secondary_confidence']
    if a.shape != b.shape or a.shape[0] != len(ids): raise ValueError('Confidence shape mismatch.')
    bins=np.linspace(0,1,51)
    fig, axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True,sharex=True,sharey=True)
    stats=[]
    for j,frame in enumerate(ids):
        ax=axes.flat[j]
        for values,name in ((a[j],'Baseline'),(b[j],'Secondary pass')):
            ax.hist(values.ravel(),bins=bins,density=True,histtype='step',label=name)
        ax.set(title=f'Primary {frame}',xlabel='Max depth-candidate probability',ylabel='Density',xlim=(0,1))
        ax.legend(fontsize=8)
        corr=float(np.corrcoef(a[j].ravel(),b[j].ravel())[0,1]) if a[j].std()>1e-12 and b[j].std()>1e-12 else None
        row={'primary':frame,'mean_delta':float((b[j]-a[j]).mean()),
             'mean_absolute_change':float(np.abs(b[j]-a[j]).mean()),'spatial_pearson':corr}
        for name,values in (('baseline',a[j]),('secondary',b[j])):
            row[name+'_mean']=float(values.mean())
            row[name+'_std']=float(values.std())
            for q in (5,50,95): row[f'{name}_p{q}']=float(np.percentile(values,q))
        stats.append(row)
        if frame not in selected: continue
        f,axs=plt.subplots(1,4,figsize=(17,4),constrained_layout=True)
        axs[0].imshow(rgb[j]);axs[0].set_title(f'Primary RGB {frame}')
        for k,values,title in ((1,a[j],'Baseline confidence'),(2,b[j],'Secondary confidence'),(3,b[j]-a[j],'Secondary - baseline')):
            im=axs[k].imshow(values,cmap='RdBu_r' if k==3 else 'viridis',vmin=-1 if k==3 else 0,vmax=1)
            axs[k].set_title(title);f.colorbar(im,ax=axs[k],shrink=.6)
        for ax2 in axs:ax2.axis('off')
        f.suptitle('Confidence: probability peak, not verified accuracy. Red difference = increased confidence.')
        f.savefig(args.out/f'confidence_primary{frame:06}.png',dpi=140);plt.close(f)
    fig.savefig(args.out/'confidence_histograms.png',dpi=150);plt.close(fig)
    write_csv(args.out/'confidence_statistics.csv',stats)

    groups=[]
    for key in data:
        if key.startswith('baseline_mv_level'):
            level=key[len('baseline_mv_level'):]
            maps,rows=feature_comparison(data[key],data['secondary_mv_level'+level])
            groups.append(('before_level'+level,maps,rows))
    for condition in ('secondary_features_only','secondary_mv_features_only'):
        prefix=condition+'_after_'
        if prefix+'cosine' in data:
            maps={k:data[prefix+k] for k in ('rms_baseline','rms_secondary','relative_difference','cosine')}
            rows=json.loads((args.run/f'{condition}_after_statistics.json').read_text())
            groups.append(('after_'+condition,maps,rows))
    feature_stats=[]
    for stage,maps,rows in groups:
        for j,row in enumerate(rows):feature_stats.append({'stage':stage,'primary':ids[j],**row})
        rms_max=max(float(np.percentile(np.r_[maps['rms_baseline'].ravel(),maps['rms_secondary'].ravel()],99)),1e-8)
        diff_max=max(float(np.percentile(maps['relative_difference'],99)),1e-8)
        for frame in selected:
            j=ids.index(frame)
            f,axs=plt.subplots(1,5,figsize=(20,4),constrained_layout=True)
            axs[0].imshow(rgb[j]);axs[0].set_title(f'Primary RGB {frame}')
            for k,key,title,cmap,low,high in (
                (1,'rms_baseline','Baseline magnitude (RMS)','viridis',0,rms_max),
                (2,'rms_secondary','Secondary magnitude (RMS)','viridis',0,rms_max),
                (3,'relative_difference','Relative RMS difference','magma',0,diff_max),
                (4,'cosine','Cosine similarity','coolwarm',-1,1)):
                palette=plt.get_cmap(cmap).copy();palette.set_bad('#b3b3b3')
                im=axs[k].imshow(np.ma.masked_invalid(maps[key][j]),cmap=palette,vmin=low,vmax=high)
                axs[k].set_title(title);f.colorbar(im,ax=axs[k],shrink=.6,extend='max' if k<4 else 'neither')
            for ax in axs:ax.axis('off')
            f.suptitle(f'{stage}: magnitude is not confidence; differences are not reconstruction errors.\n'
                       'Native resolution. RMS/difference limits: stage-wide percentile 99; grey cosine = undefined.')
            f.savefig(args.out/f'{stage}_primary{frame:06}.png',dpi=140);plt.close(f)
    write_csv(args.out/'feature_statistics.csv',feature_stats)
    print('Wrote:',args.out.resolve())


if __name__=='__main__':main()
