"""Frozen-input comparison of original, validity and ambiguity aggregation.

Run only inside a GPU allocation. Diagnostics and exports are outside timing.
"""
import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_encoder(reference_root, cfg):
    """Use original encoder, depth forward and warp, with shared unchanged dependencies."""
    package = "src.model.encoder.unimatch"
    matching = load_module(
        package + "._reference_matching",
        reference_root / "src/model/encoder/unimatch/matching.py",
    )
    mv = load_module(
        package + "._reference_mv",
        reference_root / "src/model/encoder/unimatch/mv_unimatch.py",
    )
    mv.warp_with_pose_depth_candidates = matching.warp_with_pose_depth_candidates
    encoder = load_module(
        "src.model.encoder._reference_encoder",
        reference_root / "src/model/encoder/encoder_depthsplat.py",
    )
    encoder.MultiViewUniMatch = mv.MultiViewUniMatch
    return encoder.EncoderDepthSplat(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--render-chunk", type=int, default=8)
    args = parser.parse_args()
    if args.warmups < 1 or args.render_chunk < 1:
        raise ValueError("Warmups and render chunk must be positive.")

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.dataset.dataset_dl3dv import DatasetDL3DV
    from src.dataset.shims.crop_shim import apply_crop_shim
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.model.ply_export import save_gaussian_ply
    from src.misc.image_io import save_image
    from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips, get_lpips

    if not torch.cuda.is_available():
        raise RuntimeError("Run this comparison inside a GPU allocation.")
    reference_commit = subprocess.check_output(
        ["git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if reference_commit != "2dad25a7b9ba7c5537aba08463ef88d8511a06be":
        raise ValueError("Reference must be the original pinned baseline commit.")
    subprocess.run(["git", "-C", str(args.reference_root), "diff", "--exit-code", "HEAD"], check=True)
    # Dependencies shared with the reference must still be original.
    allowed = {
        "src/model/encoder/encoder_depthsplat.py",
        "src/model/encoder/unimatch/matching.py",
        "src/model/encoder/unimatch/mv_unimatch.py",
        "src/model/encoder/unimatch/cost_volume_aggregation.py",
    }
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", reference_commit, "HEAD", "--", "src"],
        cwd=ROOT, text=True,
    ).splitlines()
    if set(changed) - allowed:
        raise ValueError(f"Unexpected changes in shared reference dependencies: {changed}")

    plan = json.loads(args.selection_plan.read_text())
    scene = plan["scene"]
    context_rows, target_rows = plan["context_indices"], plan["target_indices"]
    if len(context_rows) != 6 or len(set(context_rows)) != 6:
        raise ValueError("This initial experiment requires six distinct context views.")
    if not target_rows or len(set(target_rows)) != len(target_rows) or set(context_rows) & set(target_rows):
        raise ValueError("Targets must be unique and held out.")

    config = OmegaConf.load(args.config)
    config.mode = "test"
    config.dataset.roots = [str(args.dataset_root.resolve())]
    config.output_dir = str(args.out.resolve())
    config.wandb.mode = "disabled"
    config.model.encoder.cost_volume_aggregation = "mean"
    config.model.encoder.cost_volume_peak_exclusion_radius = 1
    config.model.encoder.cost_volume_reliability_floor = 0.05
    cfg = load_typed_root_config(config)
    set_cfg(config)
    if cfg.dataset.name != "dl3dv" or cfg.model.encoder.name != "depthsplat":
        raise ValueError("Expected DL3DV and DepthSplat.")
    if cfg.model.encoder.local_mv_match != 2 or cfg.model.encoder.train_depth_only:
        raise ValueError("Expected two matching sources and Gaussian reconstruction.")
    if cfg.dataset.make_baseline_1 or cfg.dataset.baseline_scale_bounds:
        raise ValueError("Unexpected scene rescaling settings.")
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "selection.json", plan)
    OmegaConf.save(config, args.out / "resolved_config.yaml", resolve=True)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    index = json.loads((args.dataset_root / "test/index.json").read_text())
    chunk_path = args.dataset_root / "test" / index[scene]
    chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
    examples = [item for item in chunk if item["key"] == scene]
    if len(examples) != 1:
        raise ValueError("Expected exactly one scene entry.")
    example = examples[0]
    dataset = DatasetDL3DV(cfg.dataset, "test", view_sampler=None)
    extrinsics, intrinsics = dataset.convert_poses(example["cameras"])
    if len(extrinsics) != plan["frame_count"] or len(example["images"]) != len(extrinsics):
        raise ValueError("Scene coverage differs from the frozen plan.")
    for rows in (context_rows, target_rows):
        if any(type(row) is not int or row < 0 or row >= len(extrinsics) for row in rows):
            raise ValueError("Invalid selected frame index.")

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    if not enc_state:
        raise ValueError("Expected a full-model checkpoint with encoder.* keys.")
    encoder.load_state_dict(enc_state, strict=True)
    decoder.load_state_dict(dec_state, strict=True)
    encoder = encoder.eval().cuda()
    decoder = decoder.eval().cuda()

    def view_data(rows):
        rows = torch.tensor(rows, dtype=torch.long)
        images = dataset.convert_images([example["images"][i] for i in rows.tolist()])
        if cfg.dataset.skip_bad_shape and tuple(images.shape[1:]) != (3, *cfg.dataset.ori_image_shape):
            raise ValueError(f"Unexpected input shape: {images.shape}")
        return {
            "image": images, "extrinsics": extrinsics[rows], "intrinsics": intrinsics[rows],
            "near": dataset.get_bound("near", len(rows)), "far": dataset.get_bound("far", len(rows)),
            "index": rows,
        }

    batch = {"scene": [scene], "context": view_data(context_rows), "target": view_data(target_rows)}
    if list(batch["context"]["image"].shape[-2:]) != list(cfg.dataset.image_shape):
        batch = apply_crop_shim(batch, tuple(cfg.dataset.image_shape))
    for group in ("context", "target"):
        batch[group] = {k: v.unsqueeze(0) for k, v in batch[group].items()}
    batch = encoder.get_data_shim()(batch)
    for group in ("context", "target"):
        batch[group] = {k: v.cuda() for k, v in batch[group].items()}
    context, target = batch["context"], batch["target"]
    del chunk, examples, example

    def gaussians_of(result):
        return result["gaussians"] if isinstance(result, dict) else result

    def fields_of(gaussians):
        return {field.name: getattr(gaussians, field.name) for field in dataclasses.fields(gaussians)}

    def render(gaussians):
        colors = []
        for left in range(0, len(target_rows), args.render_chunk):
            sl = slice(left, left + args.render_chunk)
            result = decoder.forward(
                gaussians, target["extrinsics"][:, sl], target["intrinsics"][:, sl],
                target["near"][:, sl], target["far"][:, sl],
                tuple(target["image"].shape[-2:]), depth_mode=None,
            )
            colors.append(result.color)
        return torch.cat(colors, dim=1)[0]

    environment = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "reference_commit": reference_commit,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
        "selection_sha256": digest(args.selection_plan),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(), "precision": "float32, high matmul precision, no autocast",
        "seed": cfg.seed, "warmups": args.warmups, "render_chunk": args.render_chunk,
        "timing_scope": "one synchronized pass after warmups; encoder and rendering separately",
        "memory_scope": "absolute PyTorch peaks including resident models and inputs; scoring, diagnostics and exports excluded",
        "ambiguity_parameters": {"exclusion_radius": 1, "reliability_floor": 0.05},
    }
    write(args.out / "environment.json", environment)
    summary = []
    with torch.inference_mode():
        reference = original_encoder(args.reference_root, cfg.model.encoder)
        reference.load_state_dict(enc_state, strict=True)
        reference = reference.eval().cuda()
        reference_dump = {}
        reference_result = reference(context, 0, visualization_dump=reference_dump)
        reference_fields = {k: v.cpu() for k, v in fields_of(gaussians_of(reference_result)).items()}
        reference_depth = reference_dump["depth"].cpu()
        del reference, reference_result, reference_dump, checkpoint, state, enc_state, dec_state
        gc.collect()
        torch.cuda.empty_cache()

        for mode in ("mean", "validity", "validity_ambiguity"):
            folder = args.out / mode
            folder.mkdir()
            write(folder / "status.json", {"status": "started"})
            try:
                encoder.depth_predictor.cost_volume_aggregation = mode
                encoder.depth_predictor.collect_aggregation_diagnostics = False
                mode_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
                mode_config.model.encoder.cost_volume_aggregation = mode
                OmegaConf.save(mode_config, folder / "resolved_config.yaml", resolve=True)
                get_lpips.cache_clear()
                gc.collect()
                torch.cuda.empty_cache()
                for _ in range(args.warmups):
                    warm = encoder(context, 0)
                    warm_render = render(gaussians_of(warm))
                    del warm, warm_render
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                dump = {}
                start = time.perf_counter()
                result = encoder(context, 0, visualization_dump=dump)
                torch.cuda.synchronize()
                encoder_seconds = time.perf_counter() - start
                gaussians = gaussians_of(result)
                start = time.perf_counter()
                prediction = render(gaussians)
                torch.cuda.synchronize()
                render_seconds = time.perf_counter() - start
                allocated = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
                for field in fields_of(gaussians).values():
                    if not torch.isfinite(field).all():
                        raise ValueError("Nonfinite Gaussian parameters.")
                if mode == "mean":
                    for key, value in fields_of(gaussians).items():
                        torch.testing.assert_close(value.cpu(), reference_fields[key], rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(dump["depth"].cpu(), reference_depth, rtol=1e-5, atol=1e-6)
                    write(args.out / "original_equivalence.json", {"passed": True, "rtol": 1e-5, "atol": 1e-6})
                    print("PASS: full encoder mean mode matches original", flush=True)
                    del reference_fields, reference_depth

                # Separate, untimed diagnostic pass; never included in performance peaks.
                encoder.depth_predictor.collect_aggregation_diagnostics = True
                try:
                    diagnostic_result = encoder(context, 0)
                    for key, value in fields_of(gaussians_of(diagnostic_result)).items():
                        torch.testing.assert_close(value, fields_of(gaussians)[key], rtol=1e-5, atol=1e-6)
                    diagnostics = list(encoder.depth_predictor.aggregation_diagnostics)
                    write(folder / "aggregation_diagnostics.json", diagnostics)
                    del diagnostic_result
                finally:
                    encoder.depth_predictor.collect_aggregation_diagnostics = False

                per_target = []
                for j, frame in enumerate(target_rows):
                    pred, gt = prediction[j:j+1], target["image"][0, j:j+1]
                    per_target.append({
                        "frame_index": frame, "psnr": compute_psnr(gt, pred).item(),
                        "ssim": compute_ssim(gt, pred).item(), "lpips": compute_lpips(gt, pred).item(),
                    })
                    save_image(pred[0], folder / "rendered" / f"{frame:06d}.png")
                    if mode == "mean":
                        save_image(gt[0], args.out / "ground_truth" / f"{frame:06d}.png")
                for j, frame in enumerate(context_rows):
                    depth = dump["depth"][0, j, :, :, 0, 0].cpu().numpy()
                    np.save(folder / f"depth_{frame:06d}.npy", depth)
                    if mode == "mean":
                        save_image(context["image"][0, j], args.out / "context" / f"{frame:06d}.png")
                save_gaussian_ply(gaussians, dump, batch, folder / "gaussians.ply")
                torch.save({k: v.cpu() for k, v in fields_of(gaussians).items()}, folder / "gaussians_full.pt")
                metrics = {
                    "condition": mode, "scene": scene, "context_indices": context_rows,
                    "target_indices": target_rows, "target_count": len(target_rows),
                    "gaussian_count": gaussians.means.shape[1],
                    "encoder_seconds": encoder_seconds, "render_seconds": render_seconds,
                    "render_seconds_per_target": render_seconds / len(target_rows),
                    "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                    **{k: float(np.mean([r[k] for r in per_target])) for k in ("psnr", "ssim", "lpips")},
                }
                write(folder / "scores_per_target.json", per_target)
                write(folder / "metrics.json", metrics)
                write(folder / "status.json", {"status": "complete"})
                summary.append(metrics)
                write(args.out / "summary.json", summary)
                print(mode, json.dumps(metrics), flush=True)
                del result, gaussians, dump, prediction, pred, gt
            except Exception as error:
                write(folder / "status.json", {"status": "failed", "error": repr(error)})
                raise
    print(f"COMPLETE: {scene}; all three aggregation modes", flush=True)


if __name__ == "__main__":
    main()
