"""Resolve the recorded DL3DV Base architecture without local absolute paths."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'config'),version_base=None):
        cfg=compose(config_name='main',overrides=['+experiment=dl3dv','mode=test','dataset/view_sampler=evaluation'])
    settings={
        'dataset.roots':[str(args.dataset_root.resolve())],
        'dataset.image_shape':[256,448], 'dataset.ori_image_shape':[540,960],
        'dataset.near':.5,'dataset.far':200.,'dataset.make_baseline_1':False,
        'dataset.baseline_scale_bounds':False,'dataset.overfit_to_scene':None,
        'model.encoder.monodepth_vit_type':'vitb','model.encoder.num_scales':2,
        'model.encoder.upsample_factor':4,'model.encoder.lowest_feature_resolution':8,
        'model.encoder.local_mv_match':2,'model.encoder.train_depth_only':False,
        'model.encoder.return_depth':True,
        'checkpointing.pretrained_model':str(args.checkpoint.resolve()),
        'checkpointing.no_strict_load':False,'wandb.mode':'disabled','seed':0,
        'data_loader.test.batch_size':1,'data_loader.test.num_workers':0,
        'data_loader.test.persistent_workers':False,'test.stablize_camera':False,
    }
    for key,value in settings.items():OmegaConf.update(cfg,key,value,merge=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg,args.out,resolve=True)
    print(args.out.resolve())


if __name__=='__main__':main()
