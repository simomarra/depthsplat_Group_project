"""Inspect first-loop primary and secondary selection; no model inference."""
import argparse
import json
from pathlib import Path
import runpy
import numpy as np
import matplotlib.pyplot as plt
from compare_primary_selection import natural_key, plot_images
from selection_protocol import fps_numpy

module = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/dataset/view_sampler/first_loop.py'))


def plot_loop(positions, loop, selections, targets):
    fig = plt.figure(figsize=(15,7))
    end = loop['end_row']
    center = (positions.max(0)+positions.min(0))/2
    radius = np.ptp(positions,axis=0).max()*0.55
    for panel,(label,primary,secondary) in enumerate(selections,1):
        ax = fig.add_subplot(1,2,panel,projection='3d')
        ax.plot(*positions[end:].T,color='lightgray',alpha=.5,label='Excluded later trajectory')
        ax.plot(*positions[:end+1].T,color='black',alpha=.55,label='Detected first loop')
        ax.scatter(*positions[primary].T,color='tab:orange',s=70,label='Primary',depthshade=False)
        ax.scatter(*positions[secondary.ravel()].T,color='tab:blue',marker='^',s=35,label='Secondary',depthshade=False)
        ax.scatter(*positions[0],color='green',marker='*',s=180,label='Start',depthshade=False)
        ax.scatter(*positions[end],color='red',marker='X',s=100,label=f'Closing row {end} (excluded)',depthshade=False)
        if len(targets):
            ax.scatter(*positions[targets].T,color='purple',marker='s',s=25,label='Held-out targets')
        for p, supports in zip(primary,secondary):
            ax.text(*positions[p],f'P{p}',fontsize=8)
            for s in supports:
                ax.plot(*positions[[p,s]].T,color='tab:blue',linestyle='--',alpha=.5)
                ax.text(*positions[s],f'S{s}',fontsize=6)
        for setter,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):
            setter(c-radius,c+radius)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=25,azim=-60)
        ax.set_xlabel('World X'); ax.set_ylabel('World Y'); ax.set_zlabel('World Z')
        ax.set_title(label)
        ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('First spatial loop: inspect detection before using it for reconstruction')
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir',type=Path,required=True)
    parser.add_argument('--num-views',type=int,default=6)
    parser.add_argument('--return-fraction',type=float,default=.08)
    parser.add_argument('--departure-fraction',type=float,default=.30)
    parser.add_argument('--min-travel-fraction',type=float,default=1.5)
    parser.add_argument('--min-frames',type=int,default=12)
    parser.add_argument('--min-area-ratio',type=float,default=.01)
    parser.add_argument('--max-rotation-deg',type=float,default=None)
    parser.add_argument('--support-offset',type=float,default=.25)
    parser.add_argument('--heldout-targets',type=int,default=0)
    parser.add_argument('--no-images',action='store_true')
    parser.add_argument('--save-dir',type=Path,help='Optional NEW output folder for plan and figures')
    args = parser.parse_args()
    if args.num_views<2 or args.heldout_targets<0:
        parser.error('Require num-views >=2 and heldout-targets >=0.')
    data = json.loads((args.scene_dir/'transforms.json').read_text())
    frames = sorted(data['frames'],key=lambda f:natural_key(f['file_path']))
    names = [Path(f['file_path']).name for f in frames]
    if len(set(names))!=len(names):
        raise ValueError('Repeated image filenames.')
    transforms = np.asarray([f['transform_matrix'] for f in frames],dtype=float)
    if transforms.shape!=(len(frames),4,4):
        raise ValueError('Expected camera-to-world transforms [N,4,4].')
    positions = transforms[:,:3,3]
    try:
        loop = module['detect_first_loop'](positions, args.return_fraction,args.departure_fraction,
                args.min_travel_fraction,args.min_frames,args.min_area_ratio,
                rotations=transforms[:,:3,:3],max_rotation_deg=args.max_rotation_deg)
    except ValueError as error:
        print(str(error))
        print('Selection stopped. No frames have been selected and no HPC settings changed.')
        raise SystemExit(2)
    end = loop['end_row']
    if end < 3*args.num_views + args.heldout_targets:
        raise ValueError('Detected first loop is too short for all unique inputs and held-out targets. Reduce view count; a later loop was not substituted.')
    targets = np.linspace(0,end-1,args.heldout_targets+2)[1:-1].round().astype(np.int64)
    eligible = np.setdiff1d(np.arange(end),targets)
    primary_sets = [
        ('Original-rule FPS within loop',eligible[fps_numpy(positions[eligible],args.num_views)]),
        ('Uniform distance within loop',module['closed_trajectory_indices'](positions,end,args.num_views,targets))]
    selections=[]
    report={'loop':loop,'closing_filename':names[end],'total_frames':len(frames),
            'eligible_range':[0,end-1],'target_rows':targets.tolist(),
            'support_offset':args.support_offset,'methods':{}}
    for label,primary in primary_sets:
        secondary=module['closed_secondary_indices'](positions,end,primary,args.support_offset,targets)
        all_inputs=np.r_[primary,secondary.ravel()]
        assert np.unique(all_inputs).size==3*args.num_views
        assert not np.intersect1d(all_inputs,targets).size
        selections.append((label,primary,secondary))
        report['methods'][label]={'primary_rows':primary.tolist(),'secondary_rows':secondary.tolist(),
                                 'primary_files':[names[i] for i in primary],
                                 'secondary_files':[[names[i] for i in pair] for pair in secondary]}
    print(json.dumps(report,indent=2))
    figures=[('positions',plot_loop(positions,loop,selections,targets))]
    if not args.no_images:
        paths=[args.scene_dir/'images_8'/name for name in names]
        for number,(label,primary,secondary) in enumerate(selections):
            for i in np.r_[primary,secondary.ravel()]:
                if not paths[i].is_file(): raise FileNotFoundError(paths[i])
            plot_images(label,primary,secondary,paths,names)
            figures.append((f'images_{number}',plt.gcf()))
    if args.save_dir:
        args.save_dir.mkdir(parents=True,exist_ok=False)
        (args.save_dir/'first_loop_selection.json').write_text(json.dumps(report,indent=2))
        for name,figure in figures:
            figure.savefig(args.save_dir/f'{name}.png',dpi=160,bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    main()
