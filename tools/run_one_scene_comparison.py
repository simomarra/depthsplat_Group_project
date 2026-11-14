"""One-scene, four-condition frozen-checkpoint experiment; run on allocated GPU.

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
    parser.add_argument('--checkpoint', type=Path, help='Same full-model checkpoint used for baseline')
    parser.add_argument('--dataset-root', type=Path, required=True, help='Converted dataset containing test/index.json')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', type=Path, required=True, help='NEW output directory; never overwritten')
    parser.add_argument('--expected-frames', type=int, help='Fail if conversion omitted frames; use 348 for this scene')
    parser.add_argument('--num-primary', type=int, default=6)
    parser.add_argument('--num-targets', type=int, default=8)
    parser.add_argument('--support-offset', type=float, default=0.25)
    parser.add_argument('--render-chunk', type=int, default=2)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--save-gaussians', action='store_true', help='Save full tensors and primary-only PLY')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.render_chunk < 1 or args.warmups < 1:
        raise ValueError('Require positive render chunk size and at least one warmup.')
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.dataset.dataset_dl3dv import DatasetDL3DV
    from src.dataset.shims.crop_shim import apply_crop_shim
    from src.dataset.view_sampler.view_sampler_bounded_v2 import farthest_point_sample

    config = OmegaConf.load(args.config)
    # Saved resolved configuration supplies architecture and preprocessing.
    config.mode = 'test'
    config.dataset.roots = [str(args.dataset_root.resolve())]
    config.dataset.overfit_to_scene = None
    config.output_dir = str(args.out.resolve())
    config.wandb.mode = 'disabled'
    cfg = load_typed_root_config(config)
    set_cfg(config)
    if cfg.dataset.name != 'dl3dv' or cfg.model.encoder.name != 'depthsplat':
        raise ValueError('This runner supports the DL3DV DepthSplat configuration only.')
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

    # Resolve exactly one converted scene, avoiding silent dataset iteration skips.
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
        raise ValueError(f'Converted scene has {n} frames; expected {args.expected_frames}. Reconvert the full scene before this experiment.')
    if n != len(example['images']):
        raise ValueError('Image/camera count mismatch.')
    if not torch.isfinite(extrinsics).all() or not torch.isfinite(intrinsics).all():
        raise ValueError('Nonfinite scene cameras.')
    if not torch.allclose(torch.det(extrinsics[:, :3, :3]), torch.ones(n), atol=1e-4, rtol=1e-4):
        raise ValueError('Invalid camera rotation determinants.')
    from src.geometry.projection import get_fov
    if (get_fov(intrinsics).rad2deg() > cfg.dataset.max_fov).any():
        raise ValueError('Camera field of view exceeds original dataset limit.')

    def original_fps(positions, count):
        xyz = torch.as_tensor(positions, dtype=torch.float32)[None]
        rows = farthest_point_sample(xyz, count)[0].cpu().numpy()
        if len(np.unique(rows)) != count:
            raise ValueError('Original FPS returned repeated indices; cannot enforce uniqueness.')
        return np.sort(rows)

    selection_start = time.perf_counter()
    plan = make_plan(extrinsics[:, :3, 3].numpy(), args.num_primary,
                     args.num_targets, args.support_offset, fps=original_fps)
    selection_seconds = time.perf_counter() - selection_start
    plan.update(scene=args.scene, dataset_chunk=str(chunk_path.resolve()),
                fps='original repository farthest_point_sample on float32 CPU poses',
                selection_seconds_both_methods=selection_seconds,
                frame_order='converted dataset row order; IDs are not raw image filenames')
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / 'selection.json', plan)
    OmegaConf.save(config, args.out / 'resolved_config.yaml', resolve=True)
    np.savez(args.out / 'full_scene_cameras.npz', extrinsics=extrinsics.numpy(), intrinsics=intrinsics.numpy())
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
    })

    def view_data(rows):
        rows = torch.as_tensor(rows, dtype=torch.long)
        images = dataset.convert_images([example['images'][i] for i in rows.tolist()])
        if cfg.dataset.skip_bad_shape and tuple(images.shape[1:]) != (3, *cfg.dataset.ori_image_shape):
            raise ValueError(f'Unexpected source image shape {images.shape[1:]}.')
        return {'image': images, 'extrinsics': extrinsics[rows], 'intrinsics': intrinsics[rows],
                'near': dataset.get_bound('near', len(rows)), 'far': dataset.get_bound('far', len(rows)), 'index': rows}

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
        for method in ('fps', 'trajectory'):
            primary = plan['methods'][method]['primary']
            for use_support in (False, True):
                if method == 'fps' and use_support:
                    continue

                label = method + ('_support' if use_support else '_primary_only')
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
                        scores = {'frame_index': frame, 'psnr': compute_psnr(gt, pred).item(),
                                  'ssim': compute_ssim(gt, pred).item(), 'lpips': compute_lpips(gt, pred).item()}
                        per_target.append(scores)
                        save_image(pred[0], folder / 'rendered' / f'{frame:06d}.png')
                        save_image(gt[0], args.out / 'ground_truth' / f'{frame:06d}.png')
                    output_context = primary_context(context, context['primary_rows']) if use_support else context
                    for j, frame in enumerate(primary):
                        save_image(output_context['image'][0, j], folder / 'primary' / f'{frame:06d}.png')
                        np.save(folder / f'depth_{frame:06d}.npy', dump['depth'][0,j,:,:,0,0].cpu().numpy())
                    for j, frame in enumerate(context['index'][0].tolist()[args.num_primary:], args.num_primary):
                        save_image(context['image'][0,j], folder / 'secondary' / f'{frame:06d}.png')
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
                               'target_indices': plan['target_indices']}
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
    print('All three conditions complete. One-scene exploratory result only.', flush=True)


if __name__ == '__main__':
    main()
