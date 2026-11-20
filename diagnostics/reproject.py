"""Offline baseline/full-secondary primary-depth reprojection on identical targets."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import pair_inputs, depth_array, write_csv, write_json
from .reprojection import project, combine, compare


def panel(a,b,gt,title,path,error_max):
    scores,common,ea,eb=compare(a,b,gt)
    fig,axes=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
    for ax,value,label in zip(axes[0],(gt,np.where(np.isfinite(a[1])[...,None],a[0],.7),
        np.where(np.isfinite(b[1])[...,None],b[0],.7)),('GT','Baseline reprojection','Secondary-depth reprojection')):
        ax.imshow(value);ax.set_title(label)
    for ax,value,label in zip(axes[1],(np.isfinite(a[1]),np.isfinite(b[1]),common),
                             ('Baseline coverage','Secondary coverage','Common coverage')):
        ax.imshow(value,cmap='gray',vmin=0,vmax=1);ax.set_title(label)
    for ax,value,label,is_delta in zip(axes[2],(ea,eb,eb-ea),
        ('Baseline MAE: common pixels','Secondary MAE: common pixels','Secondary - baseline MAE'),(False,False,True)):
        cmap=plt.get_cmap('RdBu_r' if is_delta else 'magma').copy();cmap.set_bad('#b3b3b3')
        im=ax.imshow(np.ma.masked_where(~common,value),cmap=cmap,vmin=-error_max if is_delta else 0,vmax=error_max)
        ax.set_title(label);fig.colorbar(im,ax=ax,shrink=.65)
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(title+'\nGrey = unevaluated; red difference = secondary worse. Photometric proxy, not depth accuracy.')
    fig.savefig(path,dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--method',choices=('fps','trajectory'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--targets',nargs='+',type=int,help='Plot these targets only; all targets always scored')
    p.add_argument('--primaries',nargs='+',type=int,help='Plot these primaries; default first primary')
    p.add_argument('--all-figures',action='store_true')
    p.add_argument('--error-max',type=float,default=.25)
    args=p.parse_args()
    if args.error_max<=0:raise ValueError('Positive error scale required.')
    a,b=pair_inputs(args.run,args.method);args.out.mkdir(parents=True,exist_ok=True)
    frames=a['primary_indices'].tolist();targets=a['target_index'][0].tolist()
    show_t=targets if args.all_figures else (args.targets or targets[:1])
    show_p=frames if args.all_figures else (args.primaries or frames[:1])
    if not set(show_t)<=set(targets) or not set(show_p)<=set(frames):raise ValueError('Unknown plot selection.')
    individual=[];combined_rows=[]
    # Identity projection checked once per primary/condition, not once per target.
    for data in (a,b):
        depths=depth_array(data['primary_depth'])
        for j,frame in enumerate(frames):
            row=int(data['primary_rows'][0,j]);d=depths[0,j]
            rgb=data['context_image'][0,row].transpose(1,2,0)
            k=data['context_intrinsics'][0,row].astype(np.float64)
            pose=data['context_extrinsics'][0,row].astype(np.float64)
            near,far=float(data['context_near'][0,row]),float(data['context_far'][0,row])
            identity=project(d,rgb,k,pose,k,pose,near,far)
            expected=np.isfinite(d)&(d>0)&(d>=near*(1-1e-6))&(d<=far*(1+1e-6))
            np.testing.assert_array_equal(np.isfinite(identity[1]),expected)
            np.testing.assert_allclose(identity[0][expected],rgb[expected],atol=1e-6)
    for t,target in enumerate(targets):
        gt=a['target_image'][0,t].transpose(1,2,0);warps=[]
        for data in (a,b):
            depths=depth_array(data['primary_depth']);current=[]
            for j,frame in enumerate(frames):
                row=int(data['primary_rows'][0,j])
                current.append(project(depths[0,j],data['context_image'][0,row].transpose(1,2,0),
                    data['context_intrinsics'][0,row].astype(np.float64),data['context_extrinsics'][0,row].astype(np.float64),
                    data['target_intrinsics'][0,t].astype(np.float64),data['target_extrinsics'][0,t].astype(np.float64),
                    float(data['target_near'][0,t]),float(data['target_far'][0,t])))
            warps.append(current)
        for j,frame in enumerate(frames):
            scores,*_=compare(warps[0][j],warps[1][j],gt)
            individual.append({'method':args.method,'target':target,'primary':frame,**scores})
            if target in show_t and frame in show_p:
                panel(warps[0][j],warps[1][j],gt,f'{args.method} | primary {frame} | target {target}',
                      args.out/f'target{target:06}_primary{frame:06}.png',args.error_max)
        ca,cb=[combine(v) for v in warps];scores,*_=compare(ca,cb,gt)
        combined_rows.append({'method':args.method,'target':target,**scores})
        if target in show_t:
            panel(ca,cb,gt,f'{args.method} | six primaries | target {target}',
                  args.out/f'target{target:06}_combined.png',args.error_max)
    write_csv(args.out/'per_primary.csv',individual);write_csv(args.out/'combined_per_target.csv',combined_rows)
    write_json(args.out/'checks.json',{'identity_projection':True,'input_alignment':True,'all_targets_scored':targets,
        'note':'Nearest-pixel z-buffer; no target-depth visibility test; per-target common masks differ.'})
    print('Wrote:',args.out.resolve())


if __name__=='__main__':main()
