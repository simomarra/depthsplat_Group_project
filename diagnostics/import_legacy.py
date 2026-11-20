"""Import small measured tables from the original HPC outputs; never infer missing scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from .io import sha256,write_json,write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-outputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();summary=[];per_target=[];sources=[];missing=[]
    scene_ids=[p.stem for p in sorted((Path(__file__).parent/'plans').glob('*.json'))]
    mapping={'baseline':'baseline','hybrid_support_depth':'secondary_depth_only',
        'hybrid_support_probs':'secondary_depth_confidence','hybrid_support_features':'secondary_features_depth',
        'hybrid_support_all':'full_secondary','full_support_existing':'ordinary_secondary_control'}
    for run_name in ('support_diag_771900_0','support_diag_771900_1','ablation_uniform_scene1_772082','feature_ablation_uniform_scene1_772128'):
        run=args.source_outputs/run_name;scene=scene_ids[1] if run_name.endswith('_1') else scene_ids[0]
        path=run/'summary.json';sources.append({'path':str(path.relative_to(args.source_outputs)),'sha256':sha256(path)})
        for row in json.loads(path.read_text()):
            old=row['condition']
            if run_name.startswith('support_diag'):
                method='fps' if old.startswith('fps') else 'trajectory'
                name='baseline' if old.endswith('primary_only') else 'full_secondary'
            else:method='trajectory';name=mapping.get(old,old)
            base=dict(source_run=run_name,scene=scene,method=method,original_condition=old,condition=name)
            metrics={k:row[k] for k in ('psnr','ssim','lpips')};summary.append({**base,**metrics})
            path=run/old/'scores_per_target.json'
            if not path.exists():missing.append(str(path.relative_to(args.source_outputs)));continue
            sources.append({'path':str(path.relative_to(args.source_outputs)),'sha256':sha256(path)})
            targets=json.loads(path.read_text())
            for entry in targets:per_target.append({**base,'target':entry['frame_index'],**{k:entry[k] for k in metrics}})
            for metric in metrics:
                np.testing.assert_allclose(np.mean([x[metric] for x in targets]),metrics[metric],rtol=0,atol=1e-10)
    write_csv(args.out/'scene_averages.csv',summary);write_csv(args.out/'per_target.csv',per_target)
    write_json(args.out/'provenance.json',{'base_commit':'f5c372157edc494e85783bb0a7cd7b11c2be6b71',
        'source_files':sources,'missing_per_target_files':missing,
        'note':'Historical measurements only; means verified where per-target files exist. No new features-only measurements. Historical feature-group ablations used secondary depth.'})
    print(len(summary),'scene averages;',len(per_target),'per-target records;',len(missing),'missing control files')


if __name__=='__main__':main()
