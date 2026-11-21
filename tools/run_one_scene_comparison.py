"""Compare uniform primaries with and without support on one frozen scene.

Use --plan-only first. Requires the original HPC environment and baseline's
resolved YAML. Does not submit jobs, download data or modify baseline outputs.
"""
import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selection_protocol import make_plan


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Baseline resolved_config.yaml')
    parser.add_argument(
        '--baseline-inputs', type=Path,
        help='DL3DV: JSON file listing the separate baseline input frame IDs'
    )
    parser.add_argument('--hypersim-plan', type=Path, help='Frozen Hypersim view plan JSON')
    parser.add_argument('--prepared-root', type=Path, help='Folder containing ai_XXX_YYY prepared scenes')
    parser.add_argument('--checkpoint', type=Path, help='Same full-model checkpoint used for baseline')
    parser.add_argument('--dataset-root', type=Path, help='Converted DL3DV dataset containing test/index.json')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', type=Path, required=True, help='NEW output directory; never overwritten')
    parser.add_argument('--expected-frames', type=int, help='Fail if conversion omitted frames; use 348 for this scene')
    parser.add_argument('--num-primary', type=int, default=6)
    parser.add_argument('--num-targets', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2026, help='Reproducible target selection')
    parser.add_argument('--support-offset', type=float, default=0.25)
    parser.add_argument('--render-chunk', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--save-gaussians', action='store_true', help='Save full tensors and primary-only PLY')
    return parser.parse_args()


def load_dl3dv_scene(args, cfg, DatasetDL3DV, torch, np):
    """Load one original converted scene and freeze shared target frames."""
    start = time.perf_counter()
    stage = args.dataset_root / 'test'
    index = json.loads((stage / 'index.json').read_text())
    if args.scene not in index:
        raise ValueError('Scene absent from converted test/index.json.')
    chunk_path = stage / index[args.scene]
    chunk = torch.load(chunk_path, map_location='cpu', weights_only=False)
    examples = [item for item in chunk if item['key'] == args.scene]
    if len(examples) != 1:
        raise ValueError('Expected exactly one scene entry in the indexed chunk.')
    example = examples[0]
    dataset = DatasetDL3DV(cfg.dataset, 'test', view_sampler=None)
    extrinsics, intrinsics = dataset.convert_poses(example['cameras'])
    n = len(extrinsics)
    if args.expected_frames is not None and n != args.expected_frames:
        raise ValueError(f'Converted scene has {n} frames; expected {args.expected_frames}.')
    if n != len(example['images']):
        raise ValueError('Image/camera count mismatch.')
    frame_ids = example['timestamps'].detach().cpu().numpy()
    if (frame_ids.ndim != 1 or len(frame_ids) != n
            or not np.issubdtype(frame_ids.dtype, np.integer)
            or len(np.unique(frame_ids)) != n):
        raise ValueError('Expected one unique integer frame ID per image.')
    chronological_order = np.argsort(frame_ids, kind='stable')
    chronological_ids = frame_ids[chronological_order]
    id_to_row = {str(int(frame_id)): row for row, frame_id in enumerate(chronological_ids)}
    baseline = json.loads(args.baseline_inputs.read_text(encoding='utf-8'))
    if baseline.get('scene') != args.scene or len(baseline.get('frame_ids', [])) != 6:
        raise ValueError('Baseline input file needs the same scene and six input frame IDs.')
    try:
        baseline_rows = np.asarray([id_to_row[str(i)] for i in baseline['frame_ids']],
                                   dtype=np.int64)
    except KeyError as error:
        raise ValueError(f'Baseline input frame absent from this scene: {error}') from error
    scene_seed = int.from_bytes(hashlib.sha256(
        f'{args.seed}:{args.scene}'.encode()).digest()[:8], 'big')
    plan = make_plan(extrinsics[chronological_order, :3, 3].numpy(),
                     baseline_indices=baseline_rows, count=args.num_primary,
                     target_count=args.num_targets, offset=args.support_offset,
                     seed=scene_seed)
    plan['target_indices'] = chronological_order[plan['target_indices']].tolist()
    plan['baseline_input_indices'] = chronological_order[plan['baseline_input_indices']].tolist()
    for method in plan['methods'].values():
        method['primary'] = chronological_order[method['primary']].tolist()
        method['secondary'] = chronological_order[method['secondary']].tolist()
    plan.update(scene=args.scene, dataset='dl3dv', dataset_chunk=str(chunk_path.resolve()),
                trajectory_mode='full_trajectory', selection_seconds=time.perf_counter() - start,
                chronological_to_converted_rows=chronological_order.tolist(),
                converted_source_frame_ids=frame_ids.tolist(),
                target_frame_ids=frame_ids[plan['target_indices']].tolist())
    return (plan, extrinsics, intrinsics, dataset, example['images'],
            [str(int(i)) for i in frame_ids])


def main():
    args = parse_args()
    if args.render_chunk < 1 or args.warmups < 1:
        raise ValueError('Require positive render chunk size and at least one warmup.')
    if args.num_primary != 6 or args.num_targets != 10:
        raise ValueError('Require six primaries and ten held-out targets.')
    is_hypersim = args.hypersim_plan is not None
    if is_hypersim:
        if args.prepared_root is None or args.dataset_root is not None or args.baseline_inputs is not None:
            raise ValueError('Hypersim requires --hypersim-plan and --prepared-root only.')
    elif args.dataset_root is None or args.baseline_inputs is None or args.prepared_root is not None:
        raise ValueError('DL3DV requires --dataset-root and --baseline-inputs.')
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.dataset.dataset_dl3dv import DatasetDL3DV
    from src.dataset.shims.crop_shim import apply_crop_shim    
    config = OmegaConf.load(args.config)
    # Saved resolved configuration supplies architecture and preprocessing.
    config.mode = 'test'
    config.dataset.roots = [str((args.prepared_root if is_hypersim else args.dataset_root).resolve())]
    config.dataset.overfit_to_scene = None
    config.output_dir = str(args.out.resolve())
    config.wandb.mode = 'disabled'
    cfg = load_typed_root_config(config)
    set_cfg(config)
    if cfg.dataset.name != 'dl3dv' or cfg.model.encoder.name != 'depthsplat':
        raise ValueError('Requires the frozen DL3DV DepthSplat model configuration.')
    if cfg.model.encoder.local_mv_match != 2:
        raise ValueError('This protocol needs local_mv_match=2; use the matching baseline configuration.')
    if cfg.model.encoder.train_depth_only:
        raise ValueError('A Gaussian reconstruction checkpoint/config is required.')
    # Keep the saved two-scale architecture and supervision flag unchanged.
    # MultiViewUniMatch emits intermediate predictions only while training;
    # eval() returns the final depth even when supervise_intermediate_depth=True.
    if cfg.model.encoder.num_surfaces != 1 or cfg.model.encoder.gaussians_per_pixel != 1:
        raise ValueError('First-run Gaussian count/export check expects one surface and Gaussian per pixel.')
    if cfg.dataset.make_baseline_1 or cfg.dataset.baseline_scale_bounds:
        raise ValueError('Use the DL3DV baseline configuration with both baseline scaling options false.')

    if is_hypersim:
        frozen = json.loads(args.hypersim_plan.read_text(encoding='utf-8'))
        if frozen.get('dataset') != 'hypersim' or args.scene not in frozen['scenes']:
            raise ValueError('Scene absent from the frozen Hypersim view plan.')
        full_plan = frozen['scenes'][args.scene]
        if args.expected_frames is not None and full_plan['frame_count'] != args.expected_frames:
            raise ValueError('The expected full scene frame count differs from the frozen plan.')
        prepared = args.prepared_root / args.scene
        prep = json.loads((prepared / 'preprocessing.json').read_text(encoding='utf-8'))
        if (prep.get('scene') != args.scene
                or prep.get('source_plan_sha256') != sha256(args.hypersim_plan)
                or prep.get('image_count') != len(full_plan['required_image_frame_keys'])):
            raise ValueError('Prepared images do not match the frozen view plan.')
        with np.load(prepared / 'selected_cameras.npz', allow_pickle=False) as cameras:
            scene_keys = cameras['frame_keys'].tolist()
            extrinsics = torch.from_numpy(cameras['extrinsics'].copy()).float()
            intrinsics = torch.from_numpy(cameras['intrinsics'].copy()).float()
        if scene_keys != full_plan['required_image_frame_keys']:
            raise ValueError('Prepared cameras are in a different order from the view plan.')
        image_paths = [prepared / 'images' / f'{key}.png' for key in scene_keys]
        if any(not path.is_file() for path in image_paths):
            raise FileNotFoundError('A prepared Hypersim image is missing.')
        key_to_local = {key: i for i, key in enumerate(scene_keys)}
        if len(key_to_local) != len(scene_keys):
            raise ValueError('Repeated prepared camera key.')
        def local_rows(rows):
            return [key_to_local[full_plan['frame_keys'][i]] for i in rows]
        plan = dict(full_plan)
        plan['target_indices'] = local_rows(full_plan['target_indices'])
        plan['baseline_input_indices'] = local_rows(full_plan['baseline_input_indices'])
        plan['methods'] = {
            name: {'primary': local_rows(method['primary']),
                   'secondary': [local_rows(pair) for pair in method['secondary']]}
            for name, method in full_plan['methods'].items()
        }
        plan.update(dataset='hypersim', prepared_root=str(prepared.resolve()),
                    frozen_plan_sha256=sha256(args.hypersim_plan),
                    preprocessing_sha256=sha256(prepared / 'preprocessing.json'),
                    selected_frame_keys=scene_keys,
                    full_scene_target_indices=full_plan['target_indices'],
                    target_frame_keys=full_plan['target_frame_keys'])
        n = len(scene_keys)
        dataset = None
    else:
        plan, extrinsics, intrinsics, dataset, image_paths, scene_keys = load_dl3dv_scene(
            args, cfg, DatasetDL3DV, torch, np)
        n = len(extrinsics)

    method = plan['methods']['trajectory']
    primary_rows = method['primary']
    support_rows = [row for pair in method['secondary'] for row in pair]
    target_rows = plan['target_indices']
    baseline_rows = plan['baseline_input_indices']
    inputs = primary_rows + support_rows + baseline_rows
    if (len(primary_rows), len(support_rows), len(target_rows), len(baseline_rows)) != (6, 12, 10, 6):
        raise ValueError('Frozen plan has unexpected condition or target counts.')
    if (len(set(primary_rows + support_rows)) != 18 or len(set(target_rows)) != 10
            or set(inputs) & set(target_rows)
            or any(row < 0 or row >= n for row in inputs + target_rows)):
        raise ValueError('Frozen selection repeats inputs, overlaps targets or exceeds the scene.')

    if not torch.isfinite(extrinsics).all() or not torch.isfinite(intrinsics).all():
        raise ValueError('Nonfinite scene cameras.')
    if not torch.allclose(torch.det(extrinsics[:, :3, :3]), torch.ones(n), atol=1e-4, rtol=1e-4):
        raise ValueError('Invalid camera rotation determinants.')
    from src.geometry.projection import get_fov
    if (get_fov(intrinsics).rad2deg() > cfg.dataset.max_fov).any():
        raise ValueError('Camera field of view exceeds original dataset limit.')
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'selection.json', plan)
    OmegaConf.save(config, args.out / 'resolved_config.yaml', resolve=True)
    np.savez(args.out / 'selected_cameras.npz', extrinsics=extrinsics.numpy(),
             intrinsics=intrinsics.numpy(), frame_keys=np.asarray(scene_keys))
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        print('Plan saved. No model constructed and no GPU inference performed.')
        return
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise ValueError('Supply the existing full-model --checkpoint path.')
    if not torch.cuda.is_available():
        raise RuntimeError('Run inference inside a GPU allocation, not on the login node.')

    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.model.encoder.primary_support import primary_context
    from src.model.encoder.encoder_depthsplat_reference import EncoderDepthSplat as ReferenceEncoder
    from src.misc.image_io import save_image
    from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips, get_lpips

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision('high')
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    encoder_state = {key[len('encoder.'):]: value for key, value in state.items() if key.startswith('encoder.')}
    decoder_state = {key[len('decoder.'):]: value for key, value in state.items() if key.startswith('decoder.')}
    if not encoder_state:
        raise ValueError('Expected a full-model checkpoint with encoder.* keys; no partial loading allowed.')
    encoder.load_state_dict(encoder_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    ignored = [key for key in state if not key.startswith(('encoder.', 'decoder.'))]
    del state, checkpoint, encoder_state, decoder_state
    encoder = encoder.eval().cuda()
    decoder = decoder.eval().cuda()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    code_paths = [ROOT / 'tools/run_one_scene_comparison.py', ROOT / 'tools/selection_protocol.py',
                  ROOT / 'src/model/encoder/encoder_depthsplat.py', ROOT / 'src/model/encoder/primary_support.py',
                  ROOT / 'src/dataset/view_sampler/trajectory_spacing.py', ROOT / 'src/dataset/view_sampler/secondary_selection.py']
    if is_hypersim:
        code_paths += [ROOT / 'tools/prepare_hypersim_images.py',
                       ROOT / 'tools/hypersim_camera_model.py']
    write(args.out / 'environment.json', {
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'python': platform.python_version(), 'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': sha256(args.checkpoint), 'git_commit': commit,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
        'ignored_checkpoint_keys': ignored, 'precision': 'float32; high matmul precision; no autocast',
        'warmups_per_condition': args.warmups, 'render_chunk': args.render_chunk,
        'memory_scope': 'absolute PyTorch peaks for encoder plus decoder including resident model/input tensors; excludes scoring and exports',
        'timing_scope': 'synchronized single measured pass after warmup; preprocessing and shared selection reported separately',
        'prototype': 'all-view depth prediction; primary-only Gaussian branch; global depth-regressor attention retained',
        'evaluated_dataset': 'hypersim' if is_hypersim else 'dl3dv',
    })

    def view_data(rows):
        rows = torch.as_tensor(rows, dtype=torch.long)
        if is_hypersim:
            def read_prepared_image(index):
                with Image.open(image_paths[index]) as source:
                    pixels = np.asarray(source.convert('RGB')).copy()
                return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)
            images = torch.stack([read_prepared_image(i) for i in rows.tolist()])
            near = torch.full((len(rows),), cfg.dataset.near if cfg.dataset.near > 0 else 0.1)
            far = torch.full((len(rows),), cfg.dataset.far if cfg.dataset.far > 0 else 1000.)
        else:
            images = dataset.convert_images([image_paths[i] for i in rows.tolist()])
            if cfg.dataset.skip_bad_shape and tuple(images.shape[1:]) != (3, *cfg.dataset.ori_image_shape):
                raise ValueError(f'Unexpected source image shape {images.shape[1:]}.')
            near, far = dataset.get_bound('near', len(rows)), dataset.get_bound('far', len(rows))
        return {'image': images, 'extrinsics': extrinsics[rows], 'intrinsics': intrinsics[rows],
                'near': near, 'far': far, 'index': rows}

    def frame_label(row):
        return (scene_keys[row] if is_hypersim else
                str(plan['converted_source_frame_ids'][row]))

    def frame_stem(row):
        return frame_label(row).replace('/', '_').replace('.', '_')

    def make_batch(primary, supports):
        loaded = list(primary) + [r for pair in supports for r in pair]
        batch = {'scene': args.scene, 'context': view_data(loaded), 'target': view_data(plan['target_indices'])}
        if list(batch['context']['image'].shape[-2:]) != list(cfg.dataset.image_shape):
            batch = apply_crop_shim(batch, tuple(cfg.dataset.image_shape))
        for group in ('context', 'target'):
            batch[group] = {key: value.unsqueeze(0) for key, value in batch[group].items()}
        batch['scene'] = [args.scene]
        batch = encoder.get_data_shim()(batch)
        if supports:
            p = len(primary)
            batch['context']['primary_rows'] = torch.arange(p)[None]
            batch['context']['source_rows'] = torch.arange(p, len(loaded)).reshape(1, p, 2)
        for group in ('context', 'target'):
            batch[group] = {key: value.cuda() for key, value in batch[group].items()}
        return batch

    def gaussians_of(result):
        return result['gaussians'] if isinstance(result, dict) else result

    def render(gaussians, target):
        colors = []
        for left in range(0, args.num_targets, args.render_chunk):
            sl = slice(left, left + args.render_chunk)
            output = decoder.forward(gaussians, target['extrinsics'][:, sl], target['intrinsics'][:, sl],
                                     target['near'][:, sl], target['far'][:, sl],
                                     tuple(target['image'].shape[-2:]), depth_mode=None)
            colors.append(output.color)
        return colors

    summary = []
    with torch.inference_mode():
        for method in ('trajectory',):
            primary = plan['methods'][method]['primary']
            for use_support in (False, True):
                label = 'uniform_support' if use_support else 'uniform_primary_only'
                folder = args.out / label
                folder.mkdir()
                write(folder / 'status.json', {'status': 'started'})
                try:
                    start = time.perf_counter()
                    batch = make_batch(primary, plan['methods'][method]['secondary'] if use_support else [])
                    torch.cuda.synchronize()
                    preprocess_seconds = time.perf_counter() - start
                    context, target = batch['context'], batch['target']
                    # Validate the untouched no-role path against the original
                    # forward method, using identical weights on the SAME object.
                    if not summary:
                        original = ReferenceEncoder.forward(encoder, context, 0)
                        patched = encoder(context, 0)
                        for field in dataclasses.fields(gaussians_of(original)):
                            torch.testing.assert_close(getattr(gaussians_of(original), field.name),
                                                       getattr(gaussians_of(patched), field.name), rtol=1e-5, atol=1e-6)
                        write(args.out / 'legacy_equivalence.json', {'passed': True, 'rtol': 1e-5, 'atol': 1e-6})
                        del original, patched
                    get_lpips.cache_clear()
                    gc.collect()
                    torch.cuda.empty_cache()
                    for _ in range(args.warmups):
                        warm = encoder(context, 0)
                        warm_colors = render(gaussians_of(warm), target)
                        del warm, warm_colors
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                    dump = {}
                    start = time.perf_counter()
                    result = encoder(context, 0, visualization_dump=dump)
                    torch.cuda.synchronize()
                    encoder_seconds = time.perf_counter() - start
                    gaussians = gaussians_of(result)
                    h, w = context['image'].shape[-2:]
                    expected = args.num_primary * h * w
                    if tuple(gaussians.means.shape[:2]) != (1, expected) or tuple(dump['depth'].shape[:2]) != (1, args.num_primary):
                        raise RuntimeError('Primary-only Gaussian/depth count check failed.')
                    if use_support:
                        actual = dump['matching_neighbors'][0, :args.num_primary, 1:]
                        torch.testing.assert_close(actual, context['source_rows'][0], rtol=0, atol=0)
                    start = time.perf_counter()
                    colors = render(gaussians, target)
                    torch.cuda.synchronize()
                    render_seconds = time.perf_counter() - start
                    allocated = torch.cuda.max_memory_allocated()
                    reserved = torch.cuda.max_memory_reserved()
                    prediction = torch.cat(colors, dim=1)[0]
                    del colors
                    per_target = []
                    for j, frame in enumerate(plan['target_indices']):
                        pred, gt = prediction[j:j+1], target['image'][0, j:j+1]
                        scores = {'frame_index': frame, 'frame_key': frame_label(frame),
                                  'psnr': compute_psnr(gt, pred).item(),
                                  'ssim': compute_ssim(gt, pred).item(), 'lpips': compute_lpips(gt, pred).item()}
                        per_target.append(scores)
                        save_image(pred[0], folder / 'rendered' / f'{frame_stem(frame)}.png')
                        save_image(gt[0], args.out / 'ground_truth' / f'{frame_stem(frame)}.png')
                    output_context = primary_context(context, context['primary_rows']) if use_support else context
                    for j, frame in enumerate(primary):
                        save_image(output_context['image'][0, j], folder / 'primary' / f'{frame_stem(frame)}.png')
                        np.save(folder / f'depth_{frame_stem(frame)}.npy', dump['depth'][0,j,:,:,0,0].cpu().numpy())
                    for j, frame in enumerate(context['index'][0].tolist()[args.num_primary:], args.num_primary):
                        save_image(context['image'][0,j], folder / 'secondary' / f'{frame_stem(frame)}.png')
                    if args.save_gaussians:
                        torch.save({field.name: getattr(gaussians, field.name).cpu() for field in dataclasses.fields(gaussians)}, folder / 'gaussians_full.pt')
                        from src.model.ply_export import save_gaussian_ply
                        save_gaussian_ply(gaussians, dump, {**batch, 'context': output_context}, folder / 'gaussians.ply')
                    metrics = {'condition': label, 'primary_count': args.num_primary,
                               'secondary_count': 2 * args.num_primary if use_support else 0,
                               **{key: float(np.mean([row[key] for row in per_target])) for key in ('psnr','ssim','lpips')},
                               'encoder_seconds': encoder_seconds, 'render_seconds': render_seconds,
                               'render_seconds_per_target': render_seconds / args.num_targets,
                               'preprocess_and_transfer_seconds': preprocess_seconds,
                               'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
                               'allocated_before_execution_bytes': before, 'gaussian_count': expected,
                               'target_indices': plan['target_indices'],
                               'target_frame_keys': [frame_label(i) for i in plan['target_indices']]}
                    write(folder / 'metrics.json', metrics)
                    write(folder / 'scores_per_target.json', per_target)
                    write(folder / 'status.json', {'status':'complete'})
                    summary.append(metrics)
                    write(args.out / 'summary.json', summary)
                    print(label, metrics, flush=True)
                    del result, gaussians, dump, prediction, output_context, context, target, batch, pred, gt
                    if use_support:
                        del actual
                except Exception as error:
                    write(folder / 'status.json', {'status':'failed', 'error':repr(error)})
                    raise
    print(
        f'All {len(summary)} conditions complete. One-scene exploratory result only.',
        flush=True,
    )


if __name__ == '__main__':
    main()
