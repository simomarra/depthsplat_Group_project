"""Six input-source interventions and optional feature-group ablations.

Run as python -m diagnostics.run; requires the original CUDA environment.
"""
import argparse
import dataclasses
import json
import platform
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
from .conditions import CONDITIONS, FEATURE_KEYS, MAIN_CONDITIONS, condition_table, mix_results
from .io import VIEW_KEYS, depth_array, pair_inputs, sha256, write_csv, write_json


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='Four-condition run from inference exporter')
    p.add_argument('--method', choices=('fps', 'trajectory'), required=True)
    p.add_argument('--config', type=Path, help='Defaults to RUN/resolved_config.yaml')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='New directory; never overwritten')
    p.add_argument('--conditions', nargs='+', choices=tuple(CONDITIONS), default=list(MAIN_CONDITIONS))
    p.add_argument('--feature-ablations', action='store_true', help='Add CNN/mono/MV group substitutions with baseline and secondary depths')
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--atol', type=float, default=1e-6)
    return p.parse_args()


def main():
    args = arguments()
    if args.rtol < 0 or args.atol < 0:
        raise ValueError('Tolerances must be nonnegative.')
    a, b = pair_inputs(args.run, args.method)
    if len(a['primary_indices']) != 6:
        raise ValueError('This protocol requires six fixed primaries.')
    if args.out.exists():
        raise FileExistsError(args.out)
    config_path = args.config or args.run / 'resolved_config.yaml'
    checkpoint_hash = sha256(args.checkpoint)
    old_env_path = args.run / 'environment.json'
    if old_env_path.exists():
        old_env = json.loads(old_env_path.read_text())
        if checkpoint_hash != old_env['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from the run that supplied exact inputs.')
    else:
        raise ValueError('RUN/environment.json is required to verify checkpoint identity.')

    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.model.encoder.primary_support import primary_results
    from src.model.encoder.encoder_depthsplat_reference import EncoderDepthSplat as ReferenceEncoder
    from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips
    from src.misc.image_io import save_image

    if not torch.cuda.is_available():
        raise RuntimeError('Run through Slurm on an allocated CUDA node.')
    config = OmegaConf.load(config_path)
    original_config = OmegaConf.load(args.run / 'resolved_config.yaml')
    # Permit paths/output changes, never architecture/preprocessing changes.
    for key in ('model', 'dataset.image_shape', 'dataset.near', 'dataset.far',
                'dataset.make_baseline_1', 'dataset.baseline_scale_bounds', 'seed'):
        if OmegaConf.to_container(OmegaConf.create({'v': OmegaConf.select(config, key)})) != \
           OmegaConf.to_container(OmegaConf.create({'v': OmegaConf.select(original_config, key)})):
            raise ValueError(f'Config mismatch: {key}')
    config.mode = 'test'
    config.output_dir = str(args.out.resolve())
    config.wandb.mode = 'disabled'
    cfg = load_typed_root_config(config)
    set_cfg(config)
    if cfg.model.encoder.train_depth_only or cfg.model.encoder.local_mv_match != 2:
        raise ValueError('Require Gaussian model and two-source matching budget.')
    if cfg.test.stablize_camera:
        raise ValueError('Camera stabilization is outside this fixed-camera protocol.')
    if cfg.dataset.make_baseline_1 or cfg.dataset.baseline_scale_bounds:
        raise ValueError('Scene scaling must match the exported DL3DV experiment.')
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision('high')
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    enc_state = {k[8:]: v for k, v in state.items() if k.startswith('encoder.')}
    if not enc_state:
        raise ValueError('Expected full checkpoint with encoder.* keys.')
    encoder.load_state_dict(enc_state, strict=True)
    decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith('decoder.')}, strict=True)
    ignored = [k for k in state if not k.startswith(('encoder.', 'decoder.'))]
    del checkpoint, state, enc_state
    encoder.eval().cuda()
    decoder.eval().cuda()
    args.out.mkdir(parents=True)
    write_json(args.out / 'status.json', {'status': 'running'})
    OmegaConf.save(config, args.out / 'resolved_config.yaml', resolve=True)
    repo = Path(__file__).resolve().parents[1]
    def git(*cmd):
        proc = subprocess.run(['git', *cmd], cwd=repo, capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    metadata = {
        'scene': json.loads((args.run / 'selection.json').read_text())['scene'],
        'method': args.method, 'source_run': str(args.run.resolve()),
        'checkpoint_sha256': checkpoint_hash, 'checkpoint': str(args.checkpoint.resolve()),
        'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'conditions': condition_table(), 'rtol': args.rtol, 'atol': args.atol,
        'precision': 'float32, high matmul precision; no autocast',
        'timing': 'No benchmark: predictor outputs cached for controlled substitutions.',
        'ignored_checkpoint_keys': ignored,
        'inputs_sha256': {suffix: sha256(args.run / f'{args.method}_{suffix}/diagnostic_inputs.npz')
                          for suffix in ('primary_only', 'support')},
        'source_sha256': {str(p.relative_to(repo)): sha256(p) for folder in ('diagnostics','src/model/encoder','src/model/decoder')
                          for p in sorted((repo/folder).rglob('*.py'))},
        'primary_indices': a['primary_indices'].tolist(),
        'target_indices': a['target_index'][0].tolist(),
    }
    write_json(args.out / 'provenance.json', metadata)

    def tensors(data, group):
        return {k: torch.from_numpy(data[f'{group}_{k}']).cuda() for k in VIEW_KEYS}
    context, target = tensors(a, 'context'), tensors(a, 'target')
    support_context = tensors(b, 'context')
    for key in ('primary_rows', 'source_rows'):
        support_context[key] = torch.from_numpy(b[key]).cuda()
    if context['image'].shape[-2:] != tuple(a['primary_depth'].shape[2:4]):
        raise ValueError('Exported depth/preprocessed image shape mismatch.')
    pcount, h, w = context['image'].shape[1], *context['image'].shape[-2:]
    expected_gaussians = pcount * h * w * cfg.model.encoder.num_surfaces * cfg.model.encoder.gaussians_per_pixel
    if cfg.model.encoder.num_surfaces != 1 or cfg.model.encoder.gaussians_per_pixel != 1:
        raise ValueError('Validated diagnostic protocol currently requires one Gaussian per pixel.')

    def gs(result):
        return result['gaussians'] if isinstance(result, dict) else result
    def cpu_gaussians(result):
        return {f.name: getattr(gs(result), f.name).detach().cpu().clone()
                for f in dataclasses.fields(gs(result))}
    def check_gaussians(result, expected):
        for key, value in expected.items():
            torch.testing.assert_close(getattr(gs(result), key).cpu(), value,
                                       rtol=args.rtol, atol=args.atol)
    def ordinary(views, is_secondary):
        captured = {}
        def hook(module, inputs, output):
            selected = primary_results(output, views['primary_rows'], views['image'].shape[1]) \
                       if is_secondary else output
            captured.update(selected)
        handle = encoder.depth_predictor.register_forward_hook(hook)
        dump = {}
        try:
            result = encoder(views, 0, visualization_dump=dump)
        finally:
            handle.remove()
        return result, captured, dump
    def render_scores(result, name):
        folder = args.out / name
        rows = []
        for j, frame in enumerate(a['target_index'][0]):
            sl = slice(j, j+1)
            pred = decoder.forward(gs(result), target['extrinsics'][:, sl],
                target['intrinsics'][:, sl], target['near'][:, sl], target['far'][:, sl],
                (h, w), depth_mode=None).color[0]
            gt = target['image'][0, sl]
            rows.append({'scene': metadata['scene'], 'method': args.method,
                'condition': name, 'target': int(frame),
                'psnr': compute_psnr(gt, pred).item(), 'ssim': compute_ssim(gt, pred).item(),
                'lpips': compute_lpips(gt, pred).item()})
            save_image(pred[0], folder / 'rendered' / f'{int(frame):06d}.png')
            save_image(gt[0], args.out / 'ground_truth' / f'{int(frame):06d}.png')
        write_json(folder / 'scores_per_target.json', rows)
        return rows

    checks = {}
    all_scores, summary = [], []
    try:
        with torch.inference_mode():
            xyz = context['extrinsics'][:, :, :3, -1]
            baseline_neighbors = torch.argsort(torch.cdist(xyz, xyz, p=2), dim=-1)[:, :, :3]
            np.testing.assert_array_equal(baseline_neighbors.cpu().numpy(), a['matching_neighbors'])
            if not torch.equal(baseline_neighbors[0,:,0], torch.arange(pcount,device=xyz.device)):
                raise ValueError('Baseline nearest-neighbor rule did not place self first; inspect camera ties.')
            baseline_result, baseline_outputs, baseline_dump = ordinary(context, False)
            ordinary_cpu = cpu_gaussians(baseline_result)
            reference = ReferenceEncoder.forward(encoder, context, 0)
            check_gaussians(reference, ordinary_cpu)
            checks['ordinary_vs_reference_encoder'] = True
            del reference, baseline_result
            secondary_result, secondary_outputs, secondary_dump = ordinary(support_context, True)
            secondary_cpu = cpu_gaussians(secondary_result)
            np.testing.assert_array_equal(secondary_dump['matching_neighbors'].cpu().numpy(), b['matching_neighbors'])
            np.testing.assert_array_equal(b['matching_neighbors'][0,b['primary_rows'][0],1:], b['source_rows'][0])
            checks['baseline_and_secondary_matching_sources'] = True
            for outputs, data in ((baseline_outputs, a), (secondary_outputs, b)):
                torch.testing.assert_close(outputs['depth_preds'][-1].cpu(),
                    torch.from_numpy(depth_array(data['primary_depth'])), rtol=args.rtol, atol=args.atol)
            checks['recomputed_depths_match_exports'] = True
            del secondary_result, baseline_dump, secondary_dump, support_context
            inspection = {}
            for name, outputs in (('baseline', baseline_outputs), ('secondary', secondary_outputs)):
                probs = outputs['match_probs'][-1]
                if not torch.isfinite(probs).all() or (probs < 0).any():
                    raise ValueError('Invalid matching probabilities.')
                torch.testing.assert_close(probs.sum(1), torch.ones_like(probs[:, 0]), rtol=1e-4, atol=1e-5)
                inspection[f'{name}_confidence'] = probs.max(1).values.cpu().numpy()
                for level, feature in enumerate(outputs['features_mv']):
                    inspection[f'{name}_mv_level{level}'] = feature.cpu().numpy()
            inspection['primary_indices'] = a['primary_indices']
            inspection['rgb'] = a['context_image'][0]
            baseline_features = None
            conditions = list(dict.fromkeys(['baseline', *args.conditions]))
            if args.feature_ablations:
                conditions += [x for x in CONDITIONS if x not in conditions]
            for name in conditions:
                mixed = mix_results(baseline_outputs, secondary_outputs, name)
                expected_depth = mixed['depth_preds'][-1]
                conf = F.interpolate(mixed['match_probs'][-1].max(1, keepdim=True).values,
                                     size=(h,w), mode='nearest')
                counters = {'regressor': 0, 'head': 0, 'adapter': 0, 'upsampler': 0}
                def check_reg(module, inputs):
                    counters['regressor'] += 1
                    torch.testing.assert_close(inputs[0][:, 4:5], conf, rtol=0, atol=0)
                    torch.testing.assert_close(inputs[0][:, 3:4], expected_depth.reshape(-1,1,h,w), rtol=0, atol=0)
                    torch.testing.assert_close(inputs[0][:, :3], context['image'].reshape(-1,3,h,w), rtol=0, atol=0)
                    torch.testing.assert_close(inputs[0][:, 5:], captured_features['device'], rtol=0, atol=0)
                def check_head(module, inputs):
                    counters['head'] += 1
                    torch.testing.assert_close(inputs[0][:, -1:], conf, rtol=0, atol=0)
                    channels = captured_features['device'].shape[1]
                    torch.testing.assert_close(inputs[0][:, -channels-1:-1], captured_features['device'], rtol=0, atol=0)
                def check_up(module, inputs, kwargs):
                    counters['upsampler'] += 1
                    def lists_equal(x, y):
                        if len(x) != len(y): raise ValueError('Feature pyramid mismatch')
                        for u,vv in zip(x,y):
                            torch.testing.assert_close(u,vv,rtol=0,atol=0)
                    lists_equal(inputs[0], mixed['features_mono_intermediate'])
                    lists_equal(kwargs['cnn_features'], mixed['features_cnn_all_scales'][::-1])
                    expected_mv = mixed['features_mv'][0] if cfg.model.encoder.num_scales == 1 else mixed['features_mv'][::-1]
                    if isinstance(expected_mv, list): lists_equal(kwargs['mv_features'], expected_mv)
                    else: torch.testing.assert_close(kwargs['mv_features'], expected_mv, rtol=0, atol=0)
                captured_features = {}
                def capture_up(module, inputs, output):
                    captured_features['device'] = output
                    captured_features['value'] = output.detach().cpu()
                original_adapter = encoder.gaussian_adapter.forward
                def adapter(*values, **kwargs):
                    counters['adapter'] += 1
                    torch.testing.assert_close(values[3].reshape_as(expected_depth), expected_depth, rtol=0, atol=0)
                    torch.testing.assert_close(values[0].reshape(1,pcount,4,4), context['extrinsics'], rtol=0, atol=0)
                    torch.testing.assert_close(values[1].reshape(1,pcount,3,3), context['intrinsics'], rtol=0, atol=0)
                    if cfg.model.encoder.init_sh_input_img:
                        torch.testing.assert_close(kwargs['input_images'], context['image'], rtol=0, atol=0)
                    return original_adapter(*values, **kwargs)
                handles = [encoder.gaussian_regressor.register_forward_pre_hook(check_reg),
                    encoder.gaussian_head.register_forward_pre_hook(check_head),
                    encoder.feature_upsampler.register_forward_pre_hook(check_up, with_kwargs=True),
                    encoder.feature_upsampler.register_forward_hook(capture_up)]
                dump = {}
                try:
                    with patch.object(encoder.depth_predictor, 'forward', return_value=mixed), \
                         patch.object(encoder.gaussian_adapter, 'forward', side_effect=adapter):
                        result = encoder(context, 0, visualization_dump=dump)
                finally:
                    for handle in handles: handle.remove()
                if any(n != 1 for n in counters.values()):
                    raise ValueError(f'Unexpected module call counts: {counters}')
                checks[name + '_input_paths'] = counters
                if gs(result).means.shape[:2] != (1, expected_gaussians):
                    raise ValueError('Unexpected primary-only Gaussian count.')
                torch.testing.assert_close(dump['depth'][...,0,0], expected_depth, rtol=0, atol=0)
                if name == 'baseline':
                    check_gaussians(result, ordinary_cpu)
                    checks['diagnostic_baseline_vs_ordinary'] = True
                    baseline_features = captured_features['value'].numpy().copy()
                    # Exact identity substitution is checked above, not just render metrics.
                if name == 'full_secondary':
                    check_gaussians(result, secondary_cpu)
                    checks['mixed_full_secondary_vs_ordinary_secondary'] = True
                if name in ('baseline', 'secondary_features_only', 'secondary_mv_features_only'):
                    from .feature_maps import feature_comparison
                    current = captured_features['value'].numpy()
                    if name != 'baseline':
                        maps, stats = feature_comparison(baseline_features, current)
                        for key, value in maps.items(): inspection[f'{name}_after_{key}'] = value
                        write_json(args.out / f'{name}_after_statistics.json', stats)
                rows = render_scores(result, name)
                all_scores.extend(rows)
                summary.append({'scene': metadata['scene'], 'method': args.method, 'condition': name,
                    **{k: float(np.mean([r[k] for r in rows])) for k in ('psnr','ssim','lpips')},
                    'gaussian_count': expected_gaussians, 'targets': len(rows)})
                np.savez_compressed(args.out / name / 'primary_depth.npz', depth=expected_depth.cpu().numpy())
                write_json(args.out / 'summary.json', summary)
                write_csv(args.out / 'per_target.csv', all_scores)
                write_json(args.out / 'checks.json', checks)
                print(summary[-1], flush=True)
                del result, dump, captured_features
            np.savez_compressed(args.out / 'inspection.npz', **inspection)
        write_json(args.out / 'status.json', {'status':'complete'})
    except Exception as error:
        write_json(args.out / 'status.json', {'status':'failed', 'error':repr(error)})
        raise
    print('COMPLETE:', args.out.resolve())


if __name__ == '__main__':
    main()
