#!/usr/bin/env python3
"""Collect traceable mechanism diagnostics for the MSO-Det revision.

The script is intentionally external to the training loop. It reconstructs the
eight controlled variants from their saved args.json files, runs targeted
checkpoint evaluation, and measures the two coupling paths discussed in the
manuscript:

1. ASWB -> MSIA: feature response and hypergraph topology changes.
2. UGDR -> ASWB/MSIA: localization-gradient magnitude and direction changes.

No values are synthesized. Every summary is derived from a raw CSV written by
the same invocation.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import csv
import inspect
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_MANIFEST = SCRIPT_PATH.with_name("synergy_manifest.json")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        fieldnames = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install the repository requirements first") from exc
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path: Optional[str], base: Path) -> Optional[Path]:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def nested_get(value: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    if not manifest.get("variants"):
        raise ValueError(f"No variants are defined in {path}")
    return manifest


def parse_variant_list(raw: Optional[str], manifest: Mapping[str, Any]) -> List[str]:
    if not raw:
        return list(manifest["variants"].keys())
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in manifest["variants"]]
    if unknown:
        raise ValueError(f"Unknown variants: {', '.join(unknown)}")
    return names


def infer_flags_from_files(entry_cfg: Mapping[str, Any], model_cfg_text: str) -> Dict[str, bool]:
    criterion_name = str(entry_cfg.get("criterion", "DEIMCriterion"))
    return {
        "ASWB": "WaveEncoderBlockV2" in model_cfg_text,
        "MSIA": "HyperGraphEnhance" in model_cfg_text,
        "UGDR": criterion_name == "CriterionWithUGDR" or bool(entry_cfg.get("CriterionWithUGDR")),
    }


def args_path_for_variant(logs_root: Optional[Path], spec: Mapping[str, Any]) -> Optional[Path]:
    if logs_root is None:
        return None
    path = logs_root / str(spec.get("log_dir", "")) / "args.json"
    return path if path.is_file() else None


def logged_output_dir(args_data: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not args_data:
        return None
    value = args_data.get("output_dir")
    return str(value) if value else None


def logged_model_config(args_data: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not args_data:
        return None
    value = nested_get(args_data, ["yaml_cfg", "DEIM_MG", "yaml_path"])
    return str(value) if value else None


def logged_stage2_start_epoch(
    args_data: Optional[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> int:
    fallback = int(manifest["paper_settings"].get("stage2_start_epoch", 32))
    value = nested_get(args_data or {}, ["yaml_cfg", "train_dataloader", "collate_fn", "stop_epoch"])
    try:
        return int(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def current_model_config(entry_cfg: Mapping[str, Any]) -> Optional[str]:
    value = nested_get(entry_cfg, ["DEIM_MG", "yaml_path"])
    return str(value) if value else None


def checkpoint_candidates(repo_root: Path, spec: Mapping[str, Any], args_data: Optional[Mapping[str, Any]]) -> List[Path]:
    explicit = spec.get("checkpoint")
    if explicit:
        path = resolve_path(str(explicit), repo_root)
        return [path] if path is not None else []

    output_dirs: List[str] = []
    logged = logged_output_dir(args_data)
    if logged:
        output_dirs.append(logged)
    output_dirs.extend(str(item) for item in spec.get("output_dir_candidates", []))

    names = [
        "best_stg2.pth",
        "checkpoint0071.pth",
        "last_stg2.pth",
        "last.pth",
        "best_stg1.pth",
    ]
    candidates: List[Path] = []
    seen = set()
    for output_dir in output_dirs:
        root = resolve_path(output_dir, repo_root)
        if root is None:
            continue
        for name in names:
            candidate = root / name
            key = str(candidate)
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)
        if root.is_dir():
            for candidate in sorted(root.glob("*.pth")):
                key = str(candidate.resolve())
                if key not in seen:
                    candidates.append(candidate.resolve())
                    seen.add(key)
    return candidates


def choose_checkpoint(repo_root: Path, spec: Mapping[str, Any], args_data: Optional[Mapping[str, Any]]) -> Optional[Path]:
    candidates = checkpoint_candidates(repo_root, spec, args_data)
    existing = [(index, path) for index, path in enumerate(candidates) if path.is_file()]
    if not existing:
        return None
    priority = {
        "best_stg2.pth": 0,
        "checkpoint0071.pth": 1,
        "last_stg2.pth": 2,
        "last.pth": 3,
        "best_stg1.pth": 4,
    }
    return min(existing, key=lambda item: (priority.get(item[1].name, 10), item[0]))[1]


def is_stage2_checkpoint_candidate(path: Path, stage2_start_epoch: int) -> bool:
    if path.name in {"best_stg2.pth", "last_stg2.pth"}:
        return True
    numbered = re.fullmatch(r"checkpoint(\d+)\.pth", path.name)
    return bool(numbered and int(numbered.group(1)) >= stage2_start_epoch)


def audit_variants(
    repo_root: Path,
    manifest: Mapping[str, Any],
    logs_root: Optional[Path],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {"repo_root": str(repo_root), "logs_root": str(logs_root) if logs_root else None, "variants": {}}

    for name, spec in manifest["variants"].items():
        entry_path = resolve_path(str(spec["config"]), repo_root)
        entry_exists = bool(entry_path and entry_path.is_file())
        entry_cfg = load_yaml(entry_path) if entry_exists and entry_path is not None else {}
        model_cfg_raw = current_model_config(entry_cfg)
        model_path = resolve_path(model_cfg_raw, repo_root)
        model_exists = bool(model_path and model_path.is_file())
        model_text = model_path.read_text(encoding="utf-8", errors="ignore") if model_exists and model_path else ""
        inferred = infer_flags_from_files(entry_cfg, model_text)
        expected = dict(spec["expected_modules"])

        args_path = args_path_for_variant(logs_root, spec)
        args_data = read_json(args_path) if args_path else None
        checkpoint = choose_checkpoint(repo_root, spec, args_data)
        epoch = args_data.get("epoches") if args_data else None
        seed = args_data.get("seed") if args_data else None
        stage2_start = logged_stage2_start_epoch(args_data, manifest)
        output_dir_logged = logged_output_dir(args_data)
        legacy_cfg = logged_model_config(args_data)
        flags_match = inferred == expected

        row = {
            "variant": name,
            "entry_config": str(entry_path) if entry_path else None,
            "entry_config_exists": entry_exists,
            "current_model_config": str(model_path) if model_path else model_cfg_raw,
            "current_model_config_exists": model_exists,
            "logged_model_config": legacy_cfg,
            "logged_output_dir": output_dir_logged,
            "logged_epochs": epoch,
            "logged_seed": seed,
            "stage2_start_epoch": stage2_start,
            "expected_aswb": expected["ASWB"],
            "expected_msia": expected["MSIA"],
            "expected_ugdr": expected["UGDR"],
            "inferred_aswb": inferred["ASWB"],
            "inferred_msia": inferred["MSIA"],
            "inferred_ugdr": inferred["UGDR"],
            "module_flags_match": flags_match,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_found": checkpoint is not None,
            "checkpoint_is_stage2_candidate": (
                is_stage2_checkpoint_candidate(checkpoint, stage2_start) if checkpoint else False
            ),
        }
        rows.append(row)
        details["variants"][name] = {
            **row,
            "args_json": str(args_path) if args_path else None,
            "legacy_model_configs": spec.get("legacy_model_configs", []),
            "checkpoint_candidates": [str(path) for path in checkpoint_candidates(repo_root, spec, args_data)],
        }

    write_csv(output_dir / "config_audit.csv", rows)
    write_json(output_dir / "config_audit.json", details)
    return rows


def import_runtime(repo_root: Path) -> Tuple[Any, Any]:
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    try:
        import torch
        from engine.core import YAMLConfig
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the DEIM runtime. Run from the repository environment "
            "after installing requirements.txt."
        ) from exc
    return torch, YAMLConfig


def update_dataset_config(
    cfg: Any,
    images: Path,
    annotations: Path,
    batch_size: int,
    num_workers: int,
    normalized_targets: bool,
) -> None:
    loader_cfg = cfg.yaml_cfg["val_dataloader"]
    dataset_cfg = loader_cfg["dataset"]
    dataset_cfg["img_folder"] = str(images)
    dataset_cfg["ann_file"] = str(annotations)
    if normalized_targets:
        transforms = dataset_cfg.setdefault("transforms", {"type": "Compose", "ops": []})
        operations = transforms.setdefault("ops", [])
        if operations is None:
            operations = []
            transforms["ops"] = operations
        elif not isinstance(operations, list):
            operations = list(operations)
            transforms["ops"] = operations
        if not any(isinstance(operation, Mapping) and operation.get("type") == "ConvertBoxes" for operation in operations):
            operations.append({"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True})
    loader_cfg.pop("batch_size", None)
    loader_cfg["total_batch_size"] = int(batch_size)
    loader_cfg["num_workers"] = int(num_workers)
    loader_cfg["shuffle"] = False
    loader_cfg["drop_last"] = False
    cfg._val_dataloader = None
    cfg._val_dataset = None
    cfg._evaluator = None


def torch_load_checkpoint(torch: Any, path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    ema = checkpoint.get("ema")
    if isinstance(ema, Mapping) and isinstance(ema.get("module"), Mapping):
        state = ema["module"]
    elif isinstance(checkpoint.get("model"), Mapping):
        state = checkpoint["model"]
    elif isinstance(checkpoint.get("_model"), Mapping):
        state = checkpoint["_model"]
    else:
        state = checkpoint
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def reshape_legacy_scalar_parameters(
    torch: Any,
    model: Any,
    state: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Bridge the scalar/singleton shape change already handled by BaseSolver."""
    current_state = model.state_dict()
    converted: Dict[str, Any] = {}
    reshaped: List[Dict[str, Any]] = []
    for key, value in state.items():
        target = current_state.get(key)
        source_shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        target_shape = tuple(target.shape) if isinstance(target, torch.Tensor) else None
        scalar_singleton_pair = {source_shape, target_shape} == {(), (1,)}
        if scalar_singleton_pair:
            converted[key] = value.reshape(target.shape)
            reshaped.append(
                {
                    "parameter": key,
                    "checkpoint_shape": list(source_shape),
                    "model_shape": list(target_shape),
                }
            )
        else:
            converted[key] = value
    return converted, reshaped


def load_checkpoint_into_model(
    torch: Any,
    model: Any,
    checkpoint_path: Path,
    maximum_epoch: int,
    allow_partial_load: bool,
) -> Dict[str, Any]:
    checkpoint = torch_load_checkpoint(torch, checkpoint_path)
    recorded_epoch = checkpoint.get("last_epoch", checkpoint.get("epoch"))
    if recorded_epoch is not None and not 0 <= int(recorded_epoch) <= int(maximum_epoch):
        raise RuntimeError(
            f"{checkpoint_path} records epoch {recorded_epoch}, outside the expected 0-{maximum_epoch} range"
        )
    state, scalar_reshapes = reshape_legacy_scalar_parameters(
        torch,
        model,
        extract_state_dict(checkpoint),
    )
    result = model.load_state_dict(state, strict=not allow_partial_load)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if (missing or unexpected) and not allow_partial_load:
        raise RuntimeError(f"Strict checkpoint load failed for {checkpoint_path}")
    return {
        "path": str(checkpoint_path),
        "recorded_epoch": int(recorded_epoch) if recorded_epoch is not None else None,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "legacy_scalar_reshapes": scalar_reshapes,
        "strict": not allow_partial_load,
    }


def build_runtime(
    repo_root: Path,
    manifest: Mapping[str, Any],
    variant: str,
    logs_root: Optional[Path],
    images: Path,
    annotations: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    allow_partial_load: bool,
    normalized_targets: bool = False,
) -> Dict[str, Any]:
    torch, YAMLConfig = import_runtime(repo_root)
    spec = manifest["variants"][variant]
    args_path = args_path_for_variant(logs_root, spec)
    args_data = read_json(args_path) if args_path else None
    checkpoint_path = choose_checkpoint(repo_root, spec, args_data)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for {variant}; set variants.{variant}.checkpoint in the manifest"
        )
    stage2_start_epoch = logged_stage2_start_epoch(args_data, manifest)
    if not spec.get("checkpoint") and not is_stage2_checkpoint_candidate(checkpoint_path, stage2_start_epoch):
        raise FileNotFoundError(
            f"Only a Stage-1 checkpoint was found for {variant}: {checkpoint_path}. "
            "Copy best_stg2.pth into the logged output directory or set an explicit checkpoint path."
        )

    config_path = resolve_path(str(spec["config"]), repo_root)
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(f"Missing config for {variant}: {config_path}")
    cfg = YAMLConfig(str(config_path))
    update_dataset_config(cfg, images, annotations, batch_size, num_workers, normalized_targets)
    model = cfg.model
    load_meta = load_checkpoint_into_model(
        torch,
        model,
        checkpoint_path,
        int(manifest["paper_settings"]["epochs"]) - 1,
        allow_partial_load,
    )
    torch_device = torch.device(device)
    model = model.to(torch_device)
    criterion = cfg.criterion.to(torch_device)
    postprocessor = cfg.postprocessor.to(torch_device)
    dataloader = cfg.val_dataloader
    return {
        "torch": torch,
        "cfg": cfg,
        "model": model,
        "criterion": criterion,
        "postprocessor": postprocessor,
        "dataloader": dataloader,
        "device": torch_device,
        "checkpoint": load_meta,
        "config": str(config_path),
    }


def move_targets_to_device(targets: Sequence[Mapping[str, Any]], device: Any) -> List[Dict[str, Any]]:
    return [
        {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in target.items()}
        for target in targets
    ]


def prediction_rows(results: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result, target in zip(results, targets):
        image_id = int(target["image_id"].item())
        boxes = result["boxes"].detach().cpu()
        labels = result["labels"].detach().cpu()
        scores = result["scores"].detach().cpu()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            rows.append(
                {
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score.item()),
                }
            )
    return rows


def normalize_single_class_prediction_ids(
    rows: Sequence[MutableMapping[str, Any]], annotation_data: Mapping[str, Any]
) -> int:
    category_ids = {int(category["id"]) for category in annotation_data.get("categories", [])}
    if len(category_ids) != 1:
        return 0
    category_id = next(iter(category_ids))
    changed = 0
    for row in rows:
        if int(row["category_id"]) != category_id:
            row["category_id"] = category_id
            changed += 1
    return changed


def run_predictions(
    repo_root: Path,
    manifest: Mapping[str, Any],
    logs_root: Optional[Path],
    images: Path,
    annotations: Path,
    output_dir: Path,
    variants: Sequence[str],
    device: str,
    batch_size: int,
    num_workers: int,
    allow_partial_load: bool,
) -> Dict[str, Path]:
    prediction_paths: Dict[str, Path] = {}
    metadata: Dict[str, Any] = {}
    annotation_data = read_json(annotations)
    for variant in variants:
        runtime = build_runtime(
            repo_root,
            manifest,
            variant,
            logs_root,
            images,
            annotations,
            batch_size,
            num_workers,
            device,
            allow_partial_load,
        )
        torch = runtime["torch"]
        model = runtime["model"]
        model.eval()
        runtime["postprocessor"].eval()
        rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for samples, targets in runtime["dataloader"]:
                samples = samples.to(runtime["device"], non_blocking=True)
                device_targets = move_targets_to_device(targets, runtime["device"])
                outputs = model(samples)
                original_sizes = torch.stack([target["orig_size"] for target in device_targets])
                results = runtime["postprocessor"](outputs, original_sizes, for_eval=True)
                rows.extend(prediction_rows(results, device_targets))
        remapped_categories = normalize_single_class_prediction_ids(rows, annotation_data)
        path = output_dir / "predictions" / f"{variant}.json"
        write_json(path, rows)
        prediction_paths[variant] = path
        metadata[variant] = {
            "config": runtime["config"],
            "checkpoint": runtime["checkpoint"],
            "predictions": len(rows),
            "single_class_category_ids_remapped": remapped_categories,
            "prediction_file": str(path),
        }
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output_dir / "prediction_metadata.json", metadata)
    return prediction_paths


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0 + 1.0
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def pearson(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(denom_x * denom_y)
    return numerator / denominator if denominator > 0 else None


def spearman(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return pearson(average_ranks(xs), average_ranks(ys))


def bootstrap_mean_ci(values: Sequence[float], seed: int = 42, iterations: int = 2000) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], clean[0]
    generator = random.Random(seed)
    means = []
    for _ in range(iterations):
        means.append(statistics.fmean(generator.choice(clean) for _ in clean))
    means.sort()
    lower = means[int(0.025 * (iterations - 1))]
    upper = means[int(0.975 * (iterations - 1))]
    return lower, upper


def summarize_values(values: Sequence[float], seed: int = 42) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "median": None, "ci95": [None, None]}
    lower, upper = bootstrap_mean_ci(clean, seed=seed)
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "median": statistics.median(clean),
        "ci95": [lower, upper],
    }


def density_group(count: int) -> str:
    bounds = [10, 20, 30, 40, 50, 60, 70, 80, 100]
    lower = 0
    for upper in bounds:
        if count <= upper:
            return f"{lower}-{upper}"
        lower = upper
    return ">100"


def box_intersection(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(value) for value in box_a]
    bx, by, bw, bh = [float(value) for value in box_b]
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    return max(0.0, right - left) * max(0.0, bottom - top)


def box_iou_xywh(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    intersection = box_intersection(box_a, box_b)
    area_a = max(0.0, float(box_a[2])) * max(0.0, float(box_a[3]))
    area_b = max(0.0, float(box_b[2])) * max(0.0, float(box_b[3]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def instance_occlusion_ratios(boxes: Sequence[Sequence[float]]) -> List[float]:
    ratios: List[float] = []
    for index, box in enumerate(boxes):
        area = max(0.0, float(box[2])) * max(0.0, float(box[3]))
        if area <= 0:
            ratios.append(0.0)
            continue
        overlap = sum(box_intersection(box, other) for other_index, other in enumerate(boxes) if other_index != index)
        ratios.append(overlap / area)
    return ratios


def occlusion_group(value: float) -> str:
    if value < 0.10:
        return "Low (<10%)"
    if value <= 0.30:
        return "Medium (10-30%)"
    return "High (>30%)"


def normalize_domain_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def infer_domain(image: Mapping[str, Any], known_domains: Sequence[str]) -> Optional[str]:
    candidates = [
        image.get("domain"),
        image.get("source"),
        image.get("subdataset"),
        image.get("dataset"),
        image.get("location"),
        image.get("file_name"),
    ]
    normalized_domains = sorted(
        ((domain, normalize_domain_text(domain)) for domain in known_domains),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for candidate in candidates:
        if not candidate:
            continue
        text = normalize_domain_text(candidate)
        for domain, normalized in normalized_domains:
            if normalized and normalized in text:
                return domain
    return None


SPLIT_DIRECTORIES = {
    "density": "test_density",
    "domain": "test_domain",
    "stage": "test_stage",
}

DENSITY_GROUP_ORDER = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-100", ">100"]
STAGE_GROUP_ORDER = ["Post-Flowering", "Filling", "Filling-Ripening", "Ripening"]
MINIMUM_STAGE_COVERAGE = 0.90


def infer_gwhd_root(annotation_path: Path) -> Optional[Path]:
    candidates = [annotation_path.parent, *annotation_path.parents]
    for candidate in candidates:
        if any((candidate / directory).is_dir() for directory in SPLIT_DIRECTORIES.values()):
            return candidate.resolve()
    return None


def split_group_label(path: Path, split_name: str, known_domains: Sequence[str]) -> str:
    raw = re.sub(r"(?i)(?:[_-]?annotations?)$", "", path.stem).strip("_- ")
    prefixes = {
        "density": r"(?i)^density[_-]+",
        "domain": r"(?i)^domain[_-]+",
        "stage": r"(?i)^stage[_-]+",
    }
    raw = re.sub(prefixes[split_name], "", raw).strip("_- ")
    normalized = normalize_domain_text(raw)
    if split_name == "density":
        if normalized in {"100", "100_plus", "over_100", "gt_100"} or "+" in raw:
            return ">100"
        return raw.replace("_", "-")
    if split_name == "domain":
        for domain in known_domains:
            if normalize_domain_text(domain) == normalized:
                return domain
        return raw
    stage_names = {
        "post_flowering": "Post-Flowering",
        "postflowering": "Post-Flowering",
        "filling": "Filling",
        "filling_ripening": "Filling-Ripening",
        "fillingripening": "Filling-Ripening",
        "ripening": "Ripening",
    }
    return stage_names.get(normalized, raw.replace("_", "-").title())


def full_image_lookups(
    annotation_data: Mapping[str, Any],
) -> Tuple[Dict[int, Mapping[str, Any]], Dict[str, set]]:
    images_by_id: Dict[int, Mapping[str, Any]] = {}
    ids_by_name: Dict[str, set] = defaultdict(set)
    for image in annotation_data.get("images", []):
        image_id = int(image["id"])
        images_by_id[image_id] = image
        file_name = str(image.get("file_name", "")).replace("\\", "/").lstrip("./").lower()
        if file_name:
            ids_by_name[file_name].add(image_id)
            ids_by_name[Path(file_name).name].add(image_id)
    return images_by_id, ids_by_name


def resolve_subset_image_id(
    image: Mapping[str, Any],
    full_images_by_id: Mapping[int, Mapping[str, Any]],
    full_ids_by_name: Mapping[str, set],
) -> Tuple[Optional[int], str]:
    raw_id = image.get("id")
    try:
        id_candidate = int(raw_id) if raw_id is not None and int(raw_id) in full_images_by_id else None
    except (TypeError, ValueError):
        id_candidate = None

    file_name = str(image.get("file_name", "")).replace("\\", "/").lstrip("./").lower()
    name_candidates: set = set()
    if file_name:
        name_candidates.update(full_ids_by_name.get(file_name, set()))
        name_candidates.update(full_ids_by_name.get(Path(file_name).name, set()))

    if len(name_candidates) == 1:
        name_candidate = next(iter(name_candidates))
        if id_candidate is None:
            return name_candidate, "file_name"
        if id_candidate != name_candidate:
            return name_candidate, "file_name_reindexed"
        return id_candidate, "id_and_file_name"
    if id_candidate is not None:
        return id_candidate, "image_id"
    return None, "ambiguous_file_name" if len(name_candidates) > 1 else "unresolved"


def expected_split_labels(split_name: str, known_domains: Sequence[str]) -> List[str]:
    if split_name == "density":
        return list(DENSITY_GROUP_ORDER)
    if split_name == "domain":
        return list(known_domains)
    return list(STAGE_GROUP_ORDER)


def load_authoritative_splits(
    dataset_root: Optional[Path],
    annotation_data: Mapping[str, Any],
    known_domains: Sequence[str],
) -> Tuple[Dict[str, Dict[int, str]], List[Dict[str, Any]], Dict[str, Any]]:
    full_images_by_id, full_ids_by_name = full_image_lookups(annotation_data)
    full_annotation_counts: Dict[int, int] = defaultdict(int)
    for annotation in annotation_data.get("annotations", []):
        full_annotation_counts[int(annotation["image_id"])] += 1

    assignments: Dict[str, Dict[int, str]] = {name: {} for name in SPLIT_DIRECTORIES}
    group_rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "dataset_root": str(dataset_root) if dataset_root else None,
        "full_test_images": len(full_images_by_id),
        "splits": {},
        "all_valid": True,
    }
    if dataset_root is None:
        for split_name in SPLIT_DIRECTORIES:
            report["splits"][split_name] = {"source": "fallback", "available": False}
        return assignments, group_rows, report

    for split_name, directory_name in SPLIT_DIRECTORIES.items():
        annotation_dir = dataset_root / directory_name / "annotations"
        files = sorted(annotation_dir.glob("*.json")) if annotation_dir.is_dir() else []
        if not files:
            report["splits"][split_name] = {
                "source": "fallback",
                "available": False,
                "annotation_dir": str(annotation_dir),
                "valid": False,
            }
            report["all_valid"] = False
            continue

        unresolved_examples: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []
        annotation_mismatches = 0
        mapped_methods: Dict[str, int] = defaultdict(int)
        group_image_ids: Dict[str, set] = defaultdict(set)
        labels: List[str] = []
        for path in files:
            subset = read_json(path)
            label = split_group_label(path, split_name, known_domains)
            labels.append(label)
            subset_to_full: Dict[int, int] = {}
            mapped_ids: set = set()
            methods: Dict[str, int] = defaultdict(int)
            for image in subset.get("images", []):
                full_id, method = resolve_subset_image_id(image, full_images_by_id, full_ids_by_name)
                methods[method] += 1
                mapped_methods[method] += 1
                if full_id is None:
                    if len(unresolved_examples) < 20:
                        unresolved_examples.append(
                            {
                                "annotation_file": str(path),
                                "image_id": image.get("id"),
                                "file_name": image.get("file_name"),
                                "reason": method,
                            }
                        )
                    continue
                subset_to_full[int(image["id"])] = full_id
                mapped_ids.add(full_id)
                previous = assignments[split_name].get(full_id)
                if previous is not None and previous != label:
                    conflicts.append({"image_id": full_id, "first_group": previous, "second_group": label})
                else:
                    assignments[split_name][full_id] = label

            subset_annotations = sum(
                1 for annotation in subset.get("annotations", []) if int(annotation["image_id"]) in subset_to_full
            )
            full_annotations = sum(full_annotation_counts[image_id] for image_id in mapped_ids)
            group_image_ids[label].update(mapped_ids)
            annotation_count_match = subset_annotations == full_annotations
            if not annotation_count_match:
                annotation_mismatches += 1
            group_rows.append(
                {
                    "split": split_name,
                    "group": label,
                    "annotation_file": str(path),
                    "subset_images": len(subset.get("images", [])),
                    "mapped_images": len(mapped_ids),
                    "subset_annotations": subset_annotations,
                    "full_annotations_for_images": full_annotations,
                    "annotation_count_match": annotation_count_match,
                    "mapping_methods": json.dumps(dict(sorted(methods.items())), sort_keys=True),
                }
            )

        expected_labels = expected_split_labels(split_name, known_domains)
        missing_labels = [label for label in expected_labels if label not in labels]
        unexpected_labels = [label for label in labels if label not in expected_labels]
        assigned_images = len(assignments[split_name])
        coverage = assigned_images / max(len(full_images_by_id), 1)
        coverage_complete = assigned_images == len(full_images_by_id)
        coverage_valid = coverage_complete or (split_name == "stage" and coverage >= MINIMUM_STAGE_COVERAGE)
        domain_membership_warnings_allowed = split_name == "domain"
        valid = (
            coverage_valid
            and not unresolved_examples
            and (not conflicts or domain_membership_warnings_allowed)
            and (annotation_mismatches == 0 or domain_membership_warnings_allowed)
            and not missing_labels
            and not unexpected_labels
        )
        report["splits"][split_name] = {
            "source": "authoritative_subset_annotations",
            "available": True,
            "annotation_dir": str(annotation_dir),
            "groups": labels,
            "expected_groups": expected_labels,
            "assigned_images": assigned_images,
            "coverage": coverage,
            "coverage_complete": coverage_complete,
            "minimum_accepted_coverage": MINIMUM_STAGE_COVERAGE if split_name == "stage" else 1.0,
            "unassigned_full_test_images": len(full_images_by_id) - assigned_images,
            "coverage_policy": (
                "partial_allowed_for_images_without_growth-stage_metadata"
                if split_name == "stage"
                else "full_test_coverage_required"
            ),
            "unresolved_images": sum(
                row["subset_images"] - row["mapped_images"]
                for row in group_rows
                if row["split"] == split_name
            ),
            "unresolved_examples": unresolved_examples,
            "cross_group_conflicts": conflicts[:20],
            "annotation_count_mismatches": annotation_mismatches,
            "missing_groups": missing_labels,
            "unexpected_groups": unexpected_labels,
            "mapping_methods": dict(sorted(mapped_methods.items())),
            "group_image_ids": {
                label: sorted(group_image_ids.get(label, set()))
                for label in expected_labels
            },
            "group_memberships": sum(len(image_ids) for image_ids in group_image_ids.values()),
            "overlapping_memberships": sum(len(image_ids) for image_ids in group_image_ids.values()) - assigned_images,
            "valid": valid,
        }
        report["all_valid"] = bool(report["all_valid"] and valid)
    return assignments, group_rows, report


def build_image_groups(
    annotation_data: Mapping[str, Any],
    known_domains: Sequence[str],
    split_assignments: Optional[Mapping[str, Mapping[int, str]]] = None,
) -> List[Dict[str, Any]]:
    split_assignments = split_assignments or {}
    annotations_by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotation_data.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    rows: List[Dict[str, Any]] = []
    for image in annotation_data.get("images", []):
        image_id = int(image["id"])
        annotations = annotations_by_image.get(image_id, [])
        boxes = [annotation["bbox"] for annotation in annotations]
        ratios = instance_occlusion_ratios(boxes)
        mean_occlusion = statistics.fmean(ratios) if ratios else 0.0
        rows.append(
            {
                "image_id": image_id,
                "file_name": image.get("file_name", ""),
                "instances": len(annotations),
                "density_group": split_assignments.get("density", {}).get(image_id, density_group(len(annotations))),
                "density_source": (
                    "authoritative_subset_annotations"
                    if image_id in split_assignments.get("density", {})
                    else "derived_from_instance_count"
                ),
                "domain": split_assignments.get("domain", {}).get(
                    image_id, infer_domain(image, known_domains) or "UNKNOWN"
                ),
                "domain_source": (
                    "authoritative_subset_annotations"
                    if image_id in split_assignments.get("domain", {})
                    else "inferred_from_full_annotation_metadata"
                ),
                "growth_stage": split_assignments.get("stage", {}).get(image_id, "UNKNOWN"),
                "stage_source": (
                    "authoritative_subset_annotations"
                    if image_id in split_assignments.get("stage", {})
                    else "unavailable"
                ),
                "mean_instance_occlusion": mean_occlusion,
                "occlusion_group": occlusion_group(mean_occlusion),
            }
        )
    return rows


def build_instance_occlusion_groups(annotation_data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    annotations_by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotation_data.get("annotations", []):
        if not annotation.get("iscrowd", 0):
            annotations_by_image[int(annotation["image_id"])].append(annotation)
    rows: List[Dict[str, Any]] = []
    for image_id, annotations in annotations_by_image.items():
        ratios = instance_occlusion_ratios([annotation["bbox"] for annotation in annotations])
        for annotation, ratio in zip(annotations, ratios):
            rows.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "image_id": image_id,
                    "occlusion_ratio": ratio,
                    "occlusion_group": occlusion_group(ratio),
                }
            )
    return rows


def coco_from_dataset(dataset: Mapping[str, Any]) -> Any:
    from pycocotools.coco import COCO

    coco = COCO()
    coco.dataset = copy.deepcopy(dict(dataset))
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def occlusion_group_ground_truth(
    annotation_data: Mapping[str, Any],
    instance_rows: Sequence[Mapping[str, Any]],
    group_name: str,
) -> Tuple[Any, List[int], int, set]:
    active_ids = {
        int(row["annotation_id"])
        for row in instance_rows
        if row["occlusion_group"] == group_name
    }
    grouped_data = copy.deepcopy(dict(annotation_data))
    active_images = set()
    for annotation in grouped_data.get("annotations", []):
        annotation_id = int(annotation["id"])
        if annotation_id in active_ids:
            annotation["iscrowd"] = 0
            annotation["ignore"] = 0
            active_images.add(int(annotation["image_id"]))
        else:
            annotation["iscrowd"] = 1
            annotation["ignore"] = 1
    coco = coco_from_dataset(grouped_data)
    return coco, sorted(active_images), len(active_ids), active_ids


def evaluate_coco_subset(coco_gt: Any, coco_dt: Any, image_ids: Sequence[int]) -> Dict[str, Optional[float]]:
    if not image_ids:
        return {"AP": None, "AP50": None, "AP75": None}
    from pycocotools.cocoeval import COCOeval

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = list(image_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "AP": 100.0 * float(stats[0]) if stats[0] >= 0 else None,
        "AP50": 100.0 * float(stats[1]) if stats[1] >= 0 else None,
        "AP75": 100.0 * float(stats[2]) if stats[2] >= 0 else None,
    }


def log_metric_records(logs_root: Optional[Path], spec: Mapping[str, Any]) -> List[Dict[str, float]]:
    if logs_root is None:
        return []
    path = logs_root / str(spec.get("log_dir", "")) / "log.txt"
    if not path.is_file():
        return []
    records: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            values = record.get("test_coco_eval_bbox")
            if not isinstance(values, list) or len(values) < 3:
                continue
            try:
                epoch = int(record["epoch"])
                records[epoch] = {
                    "epoch": epoch,
                    "AP": 100.0 * float(values[0]),
                    "AP50": 100.0 * float(values[1]),
                    "AP75": 100.0 * float(values[2]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return [records[epoch] for epoch in sorted(records)]


def metric_differences(
    observed: Mapping[str, Optional[float]],
    logged: Mapping[str, float],
) -> Dict[str, Optional[float]]:
    return {
        metric: (
            float(observed[metric]) - float(logged[metric])
            if observed.get(metric) is not None
            else None
        )
        for metric in ("AP", "AP50", "AP75")
    }


def closest_log_record(
    records: Sequence[Mapping[str, float]],
    observed: Mapping[str, Optional[float]],
) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for record in records:
        differences = metric_differences(observed, record)
        if any(value is None for value in differences.values()):
            continue
        absolute = [abs(float(value)) for value in differences.values() if value is not None]
        candidates.append(
            {
                "record": dict(record),
                "max_abs_difference_pp": max(absolute),
                "rmse_pp": math.sqrt(statistics.fmean(value * value for value in absolute)),
            }
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            float(item["max_abs_difference_pp"]),
            float(item["rmse_pp"]),
            int(item["record"]["epoch"]),
        ),
    )


def checkpoint_log_target(
    checkpoint_meta: Mapping[str, Any],
    records: Sequence[Mapping[str, float]],
    stage2_start_epoch: int,
    closest: Optional[Mapping[str, Any]],
) -> Tuple[Optional[Dict[str, float]], str]:
    checkpoint_path = str(checkpoint_meta.get("path", ""))
    checkpoint_name = Path(checkpoint_path).name
    stage1_records = [dict(record) for record in records if int(record["epoch"]) < stage2_start_epoch]
    stage2_records = [dict(record) for record in records if int(record["epoch"]) >= stage2_start_epoch]
    records_by_epoch = {int(record["epoch"]): dict(record) for record in records}

    if checkpoint_name == "best_stg2.pth" and stage2_records:
        return max(stage2_records, key=lambda record: float(record["AP"])), "best_stg2_selection_rule"
    if checkpoint_name == "best_stg1.pth" and stage1_records:
        return max(stage1_records, key=lambda record: float(record["AP"])), "best_stg1_selection_rule"

    numbered = re.fullmatch(r"checkpoint(\d+)\.pth", checkpoint_name)
    if numbered:
        epoch = int(numbered.group(1))
        if epoch in records_by_epoch:
            return records_by_epoch[epoch], "checkpoint_filename"

    recorded_epoch = checkpoint_meta.get("recorded_epoch")
    try:
        epoch = int(recorded_epoch) if recorded_epoch is not None else None
    except (TypeError, ValueError):
        epoch = None
    if epoch is not None and epoch in records_by_epoch:
        return records_by_epoch[epoch], "checkpoint_metadata"

    if closest is not None:
        return dict(closest["record"]), "closest_stage2_metrics"
    return None, "unresolved"


def validate_prediction_checkpoints(
    manifest: Mapping[str, Any],
    logs_root: Optional[Path],
    coco_gt: Any,
    prediction_paths: Mapping[str, Path],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    image_ids = sorted(int(image_id) for image_id in coco_gt.getImgIds())
    metadata_path = output_dir / "prediction_metadata.json"
    prediction_metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    rows: List[Dict[str, Any]] = []
    for variant, path in prediction_paths.items():
        predictions = read_json(path)
        normalize_single_class_prediction_ids(predictions, coco_gt.dataset)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(predictions)
        observed = evaluate_coco_subset(coco_gt, coco_dt, image_ids)
        spec = manifest["variants"][variant]
        args_path = args_path_for_variant(logs_root, spec)
        args_data = read_json(args_path) if args_path else None
        stage2_start_epoch = logged_stage2_start_epoch(args_data, manifest)
        records = log_metric_records(logs_root, spec)
        stage2_records = [record for record in records if int(record["epoch"]) >= stage2_start_epoch]
        closest = closest_log_record(stage2_records, observed)
        variant_metadata = prediction_metadata.get(variant, {})
        checkpoint_meta = variant_metadata.get("checkpoint", {}) if isinstance(variant_metadata, Mapping) else {}
        expected, epoch_source = checkpoint_log_target(
            checkpoint_meta,
            records,
            stage2_start_epoch,
            closest,
        )
        recorded_epoch_raw = checkpoint_meta.get("recorded_epoch") if isinstance(checkpoint_meta, Mapping) else None
        try:
            recorded_epoch = int(recorded_epoch_raw) if recorded_epoch_raw is not None else None
        except (TypeError, ValueError):
            recorded_epoch = None
        checkpoint_path = checkpoint_meta.get("path") if isinstance(checkpoint_meta, Mapping) else None
        matched_epoch = int(expected["epoch"]) if expected else None
        closest_epoch = int(closest["record"]["epoch"]) if closest else None
        row: Dict[str, Any] = {
            "variant": variant,
            "images": len(image_ids),
            "checkpoint": checkpoint_path,
            "checkpoint_name": Path(str(checkpoint_path)).name if checkpoint_path else None,
            "checkpoint_recorded_epoch": recorded_epoch,
            "stage2_start_epoch": stage2_start_epoch,
            "matched_log_epoch": matched_epoch,
            "epoch_source": epoch_source,
            "closest_stage2_log_epoch": closest_epoch,
            "metadata_to_log_epoch_offset": (
                matched_epoch - int(recorded_epoch)
                if matched_epoch is not None and recorded_epoch is not None
                else None
            ),
            "observed_AP": observed["AP"],
            "observed_AP50": observed["AP50"],
            "observed_AP75": observed["AP75"],
            "logged_AP": expected["AP"] if expected else None,
            "logged_AP50": expected["AP50"] if expected else None,
            "logged_AP75": expected["AP75"] if expected else None,
        }
        differences = metric_differences(observed, expected) if expected else {}
        for metric in ("AP", "AP50", "AP75"):
            row[f"difference_{metric}"] = (
                differences.get(metric) if expected is not None else None
            )
        row["metric_match_0.05pp"] = (
            expected is not None
            and all(
                row[f"difference_{metric}"] is not None
                and abs(float(row[f"difference_{metric}"])) <= 0.05
                for metric in ("AP", "AP50", "AP75")
            )
        )
        row["selection_epoch_matches_metric_nearest"] = (
            matched_epoch is not None and closest_epoch is not None and matched_epoch == closest_epoch
        )
        row["closest_stage2_max_abs_difference_pp"] = (
            closest["max_abs_difference_pp"] if closest else None
        )
        rows.append(row)
    write_csv(output_dir / "checkpoint_log_validation.csv", rows)
    return rows


def image_accuracy_rows(
    annotation_data: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    confidence: float,
    iou_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    gt_by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    pred_by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotation_data.get("annotations", []):
        if not annotation.get("iscrowd", 0):
            gt_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= confidence:
            pred_by_image[int(prediction["image_id"])].append(prediction)

    rows: Dict[int, Dict[str, Any]] = {}
    for image in annotation_data.get("images", []):
        image_id = int(image["id"])
        ground_truth = gt_by_image.get(image_id, [])
        detections = sorted(pred_by_image.get(image_id, []), key=lambda row: float(row["score"]), reverse=True)
        matched = set()
        true_positive = 0
        false_positive = 0
        for detection in detections:
            best_index = None
            best_iou = iou_threshold
            for index, target in enumerate(ground_truth):
                if index in matched or int(target["category_id"]) != int(detection["category_id"]):
                    continue
                iou = box_iou_xywh(detection["bbox"], target["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_index = index
            if best_index is None:
                false_positive += 1
            else:
                matched.add(best_index)
                true_positive += 1
        false_negative = len(ground_truth) - len(matched)
        denominator = true_positive + false_positive + false_negative
        rows[image_id] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "ai": true_positive / denominator if denominator else None,
        }
    return rows


def aggregate_group_accuracy(image_ids: Sequence[int], accuracy: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    selected = [accuracy[image_id] for image_id in image_ids if image_id in accuracy]
    ai_values = [float(row["ai"]) for row in selected if row.get("ai") is not None]
    return {
        "TP": sum(int(row["tp"]) for row in selected),
        "FP": sum(int(row["fp"]) for row in selected),
        "FN": sum(int(row["fn"]) for row in selected),
        "AI": statistics.fmean(ai_values) if ai_values else None,
    }


def instance_group_accuracy(
    annotation_data: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    active_annotation_ids: set,
    image_ids: Sequence[int],
    confidence: float,
    iou_threshold: float,
) -> Dict[str, Any]:
    active_by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    ignored_by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    predictions_by_image: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotation_data.get("annotations", []):
        image_id = int(annotation["image_id"])
        destination = active_by_image if int(annotation["id"]) in active_annotation_ids else ignored_by_image
        destination[image_id].append(annotation)
    for prediction in predictions:
        if float(prediction.get("score", 0.0)) >= confidence:
            predictions_by_image[int(prediction["image_id"])].append(prediction)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    image_ai: List[float] = []
    for image_id in image_ids:
        active = active_by_image.get(image_id, [])
        ignored = ignored_by_image.get(image_id, [])
        detections = sorted(
            predictions_by_image.get(image_id, []),
            key=lambda row: float(row["score"]),
            reverse=True,
        )
        matched = set()
        tp = 0
        fp = 0
        for detection in detections:
            best_index = None
            best_iou = iou_threshold
            for index, target in enumerate(active):
                if index in matched or int(target["category_id"]) != int(detection["category_id"]):
                    continue
                iou = box_iou_xywh(detection["bbox"], target["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_index = index
            if best_index is not None:
                matched.add(best_index)
                tp += 1
                continue
            matches_ignored = any(
                int(target["category_id"]) == int(detection["category_id"])
                and box_iou_xywh(detection["bbox"], target["bbox"]) >= iou_threshold
                for target in ignored
            )
            if not matches_ignored:
                fp += 1
        fn = len(active) - len(matched)
        denominator = tp + fp + fn
        if denominator:
            image_ai.append(tp / denominator)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "AI": statistics.fmean(image_ai) if image_ai else None,
    }


def group_sort_key(analysis_name: str, group_name: str, known_domains: Sequence[str]) -> Tuple[int, str]:
    orders = {
        "density": DENSITY_GROUP_ORDER,
        "domain": list(known_domains),
        "stage": STAGE_GROUP_ORDER,
        "occlusion": ["Low (<10%)", "Medium (10-30%)", "High (>30%)"],
    }
    order = orders.get(analysis_name, [])
    return (order.index(group_name) if group_name in order else len(order), group_name)


def prepare_dataset_groups(
    manifest: Mapping[str, Any],
    annotations: Path,
    output_dir: Path,
    dataset_root: Optional[Path],
) -> Tuple[Mapping[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    annotation_data = read_json(annotations)
    resolved_dataset_root = dataset_root or infer_gwhd_root(annotations)
    if resolved_dataset_root is None:
        raise FileNotFoundError(
            "Unable to locate authoritative GWHD test partitions from the annotation path; "
            "pass --gwhd-root pointing to the directory that contains test_density, "
            "test_domain, and test_stage"
        )
    split_assignments, split_group_rows, split_report = load_authoritative_splits(
        resolved_dataset_root,
        annotation_data,
        manifest["known_gwhd_test_domains"],
    )
    write_csv(output_dir / "dataset_split_groups.csv", split_group_rows)
    write_json(output_dir / "dataset_split_audit.json", split_report)
    if not split_report["all_valid"]:
        failed = [
            name
            for name, report in split_report["splits"].items()
            if not report.get("valid", True)
        ]
        raise RuntimeError(
            "Authoritative GWHD subset annotations failed coverage/consistency checks: "
            + ", ".join(failed)
            + f". Inspect {output_dir / 'dataset_split_audit.json'}"
        )
    image_rows = build_image_groups(
        annotation_data,
        manifest["known_gwhd_test_domains"],
        split_assignments,
    )
    write_csv(output_dir / "image_groups.csv", image_rows)
    return annotation_data, image_rows, split_report


def run_targeted_evaluation(
    manifest: Mapping[str, Any],
    logs_root: Optional[Path],
    annotations: Path,
    output_dir: Path,
    prediction_paths: Mapping[str, Path],
    confidence: float,
    iou_threshold: float,
    dataset_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    try:
        from pycocotools.coco import COCO
    except ImportError as exc:
        raise RuntimeError("pycocotools is required for targeted evaluation") from exc

    annotation_data, image_rows, split_report = prepare_dataset_groups(
        manifest,
        annotations,
        output_dir,
        dataset_root,
    )
    instance_rows = build_instance_occlusion_groups(annotation_data)
    write_csv(output_dir / "instance_occlusion_groups.csv", instance_rows)

    group_columns = {
        "density": "density_group",
        "domain": "domain",
        "stage": "growth_stage",
        "occlusion": "occlusion_group",
    }
    annotations_by_image: Dict[int, int] = defaultdict(int)
    for annotation in annotation_data.get("annotations", []):
        if not annotation.get("iscrowd", 0):
            annotations_by_image[int(annotation["image_id"])] += 1

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotations))
    checkpoint_rows = validate_prediction_checkpoints(manifest, logs_root, coco_gt, prediction_paths, output_dir)

    result_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "dataset_splits": split_report,
        "checkpoint_validation": checkpoint_rows,
        "analyses": {},
    }
    for analysis_name, variants in manifest["targeted_comparisons"].items():
        column = group_columns[analysis_name]
        groups: Dict[str, List[int]] = defaultdict(list)
        if analysis_name == "occlusion":
            for row in instance_rows:
                groups[str(row["occlusion_group"])].append(int(row["image_id"]))
            groups = {name: sorted(set(image_ids)) for name, image_ids in groups.items()}
        else:
            authoritative_groups = split_report["splits"].get(analysis_name, {}).get("group_image_ids", {})
            if authoritative_groups:
                groups = {
                    str(name): [int(image_id) for image_id in image_ids]
                    for name, image_ids in authoritative_groups.items()
                }
            else:
                for row in image_rows:
                    if row[column] != "UNKNOWN":
                        groups[str(row[column])].append(int(row["image_id"]))
        ordered_groups = sorted(
            groups.items(),
            key=lambda item: group_sort_key(
                analysis_name,
                item[0],
                manifest["known_gwhd_test_domains"],
            ),
        )
        summary["analyses"][analysis_name] = {
            "variants": variants,
            "groups": [name for name, _image_ids in ordered_groups],
        }

        for variant in variants:
            if variant not in prediction_paths:
                raise FileNotFoundError(f"Missing predictions for targeted comparison: {variant}")
            predictions = read_json(prediction_paths[variant])
            normalize_single_class_prediction_ids(predictions, annotation_data)
            accuracy = image_accuracy_rows(annotation_data, predictions, confidence, iou_threshold)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_dt = coco_gt.loadRes(predictions)
            group_ai_values: List[float] = []
            for group_name, image_ids in ordered_groups:
                if analysis_name == "occlusion":
                    grouped_gt, image_ids, active_instances, active_ids = occlusion_group_ground_truth(
                        annotation_data, instance_rows, group_name
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        grouped_dt = grouped_gt.loadRes(predictions)
                    coco_metrics = evaluate_coco_subset(grouped_gt, grouped_dt, image_ids)
                    accuracy_metrics = instance_group_accuracy(
                        annotation_data,
                        predictions,
                        active_ids,
                        image_ids,
                        confidence,
                        iou_threshold,
                    )
                    instance_count = active_instances
                else:
                    coco_metrics = evaluate_coco_subset(coco_gt, coco_dt, image_ids)
                    accuracy_metrics = aggregate_group_accuracy(image_ids, accuracy)
                    instance_count = sum(annotations_by_image[image_id] for image_id in image_ids)
                if accuracy_metrics["AI"] is not None:
                    group_ai_values.append(float(accuracy_metrics["AI"]))
                result_rows.append(
                    {
                        "analysis": analysis_name,
                        "group": group_name,
                        "variant": variant,
                        "images": len(image_ids),
                        "instances": instance_count,
                        **coco_metrics,
                        **accuracy_metrics,
                    }
                )
            summary["analyses"][analysis_name].setdefault("WDA", {})[variant] = (
                statistics.fmean(group_ai_values) if group_ai_values else None
            )

    write_csv(output_dir / "module_problem_correspondence.csv", result_rows)
    write_json(output_dir / "module_problem_correspondence_summary.json", summary)
    return result_rows


def find_modules_by_class_name(model: Any, class_name: str) -> List[Tuple[str, Any]]:
    return [(name, module) for name, module in model.named_modules() if module.__class__.__name__ == class_name]


def require_unique_module(model: Any, class_name: str) -> Tuple[str, Any]:
    matches = find_modules_by_class_name(model, class_name)
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise RuntimeError(f"Expected exactly one {class_name}, found {len(matches)}: {names}")
    return matches[0]


def detach_tree(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach()
    if isinstance(value, (list, tuple)):
        return type(value)(detach_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    return value


def relative_l2(torch: Any, first: Any, second: Any, eps: float = 1e-12) -> float:
    if isinstance(first, (list, tuple)):
        numerator = sum(torch.sum((left.float() - right.float()) ** 2) for left, right in zip(first, second))
        denominator = sum(torch.sum(left.float() ** 2) for left in first)
    else:
        numerator = torch.sum((first.float() - second.float()) ** 2)
        denominator = torch.sum(first.float() ** 2)
    return float(torch.sqrt(numerator / denominator.clamp_min(eps)).item())


def spectral_high_frequency_ratio(torch: Any, tensor: Any, fraction: float = 0.5) -> float:
    values = tensor.float()
    height, width = values.shape[-2:]
    spectrum = torch.fft.rfft2(values, norm="ortho")
    energy = spectrum.abs().pow(2).mean(dim=tuple(range(values.ndim - 2)))
    freq_y = torch.fft.fftfreq(height, device=values.device).abs().view(height, 1)
    freq_x = torch.fft.rfftfreq(width, device=values.device).view(1, -1)
    radius = torch.sqrt(freq_y.pow(2) + freq_x.pow(2))
    threshold = float(fraction) * math.sqrt(0.5)
    mask = radius >= threshold
    total = energy.sum().clamp_min(1e-12)
    return float((energy[mask].sum() / total).item())


def infer_alpha_mapping(wave_module: Any) -> str:
    try:
        source = inspect.getsource(wave_module.forward)
    except (OSError, TypeError):
        return "direct"
    if re.search(r"\(\s*1(?:\.0)?\s*-\s*alpha_scale\s*\)", source):
        return "inverse"
    return "direct"


def wave_parameters(torch: Any, wave_module: Any, wave_input: Any) -> Dict[str, Any]:
    if not hasattr(wave_module, "param_generator"):
        raise RuntimeError("WaveEncoderBlockV2 does not expose param_generator")
    scales = wave_module.param_generator(wave_input)
    alpha_scale = scales[:, 0].reshape(scales.shape[0], -1).mean(dim=1)
    speed_scale = scales[:, 1].reshape(scales.shape[0], -1).mean(dim=1)
    alpha_min = float(getattr(wave_module, "alpha_min"))
    alpha_max = float(getattr(wave_module, "alpha_max"))
    speed_min = float(getattr(wave_module, "speed_min"))
    speed_max = float(getattr(wave_module, "speed_max"))
    mapping = infer_alpha_mapping(wave_module)
    alpha_position = 1.0 - alpha_scale if mapping == "inverse" else alpha_scale
    dynamic_alpha = alpha_min + (alpha_max - alpha_min) * alpha_position
    dynamic_speed = speed_min + (speed_max - speed_min) * speed_scale
    return {
        "alpha_scale": float(alpha_scale[0].item()),
        "speed_scale": float(speed_scale[0].item()),
        "dynamic_alpha": float(dynamic_alpha[0].item()),
        "dynamic_speed": float(dynamic_speed[0].item()),
        "alpha_mapping": mapping,
        "alpha_range": [alpha_min, alpha_max],
        "speed_range": [speed_min, speed_max],
    }


def topology_comparison(
    torch: Any,
    normal_feature: Any,
    bypass_feature: Any,
    threshold: float,
    chunk_size: int,
) -> Dict[str, float]:
    normal_nodes = normal_feature[0].float().flatten(1).transpose(0, 1).contiguous()
    bypass_nodes = bypass_feature[0].float().flatten(1).transpose(0, 1).contiguous()
    if normal_nodes.shape != bypass_nodes.shape:
        raise RuntimeError(f"Topology inputs differ in shape: {normal_nodes.shape} vs {bypass_nodes.shape}")
    count = int(normal_nodes.shape[0])
    degree_normal: List[float] = []
    degree_bypass: List[float] = []
    intersection = 0
    union = 0
    changed = 0
    total = 0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        normal_edges = torch.cdist(normal_nodes[start:stop], normal_nodes) < threshold
        bypass_edges = torch.cdist(bypass_nodes[start:stop], bypass_nodes) < threshold
        degree_normal.extend(float(value) for value in normal_edges.sum(dim=1).detach().cpu().tolist())
        degree_bypass.extend(float(value) for value in bypass_edges.sum(dim=1).detach().cpu().tolist())
        intersection += int(torch.logical_and(normal_edges, bypass_edges).sum().item())
        union += int(torch.logical_or(normal_edges, bypass_edges).sum().item())
        changed += int(torch.logical_xor(normal_edges, bypass_edges).sum().item())
        total += int(normal_edges.numel())
    return {
        "normal_mean_nodes_per_hyperedge": statistics.fmean(degree_normal),
        "bypass_mean_nodes_per_hyperedge": statistics.fmean(degree_bypass),
        "normal_degree_std": statistics.pstdev(degree_normal),
        "bypass_degree_std": statistics.pstdev(degree_bypass),
        "topology_jaccard": intersection / union if union else 1.0,
        "topology_change_fraction": changed / total if total else 0.0,
    }


def as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def module_setting_report(wave_module: Any, hyper_module: Any, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    settings = manifest["paper_settings"]
    alpha_range = [float(wave_module.alpha_min), float(wave_module.alpha_max)]
    speed_range = [float(wave_module.speed_min), float(wave_module.speed_max)]
    target_size = int(getattr(hyper_module, "target_size"))
    threshold_value = getattr(
        hyper_module,
        "threshold",
        getattr(getattr(hyper_module, "hyper_compute", None), "threshold", None),
    )
    if threshold_value is None:
        raise RuntimeError("Unable to read the MSIA hypergraph threshold")
    threshold = float(threshold_value)
    residual = as_float(getattr(hyper_module, "residual_weight"))

    reduction = None
    generator = getattr(wave_module, "param_generator", None)
    if generator is not None:
        convolutions = [module for module in generator.modules() if module.__class__.__name__ == "Conv2d"]
        if convolutions and convolutions[0].out_channels:
            reduction = int(convolutions[0].in_channels // convolutions[0].out_channels)

    observed = {
        "aswb_alpha_range": alpha_range,
        "aswb_speed_range": speed_range,
        "aswb_reduction": reduction,
        "msia_target_size": target_size,
        "msia_threshold": threshold,
        "msia_residual_weight": residual,
    }
    expected = {key: settings[key] for key in observed}
    checks = {
        key: (
            all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-8) for a, b in zip(observed[key], expected[key]))
            if isinstance(observed[key], list)
            else observed[key] == expected[key]
            if isinstance(observed[key], int) or observed[key] is None
            else math.isclose(float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-8)
        )
        for key in observed
    }
    return {"expected": expected, "observed": observed, "checks": checks, "all_match": all(checks.values())}


def stratified_forward_sample(
    image_rows: Sequence[Mapping[str, Any]],
    max_images: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if max_images <= 0:
        raise ValueError("Forward diagnostic sample size must be positive")
    limit = min(max_images, len(image_rows))
    generator = random.Random(seed)
    candidates = [dict(row) for row in sorted(image_rows, key=lambda row: int(row["image_id"]))]
    generator.shuffle(candidates)

    grouping_columns = ("domain", "density_group", "growth_stage")

    def labels(row: Mapping[str, Any]) -> set:
        return {
            (column, str(row.get(column, "UNKNOWN")))
            for column in grouping_columns
            if str(row.get(column, "UNKNOWN")) != "UNKNOWN"
        }

    uncovered = set().union(*(labels(row) for row in candidates)) if candidates else set()
    selected: List[Dict[str, Any]] = []
    selected_ids: set = set()

    # Cover every available marginal group first, then balance the remaining
    # sample over joint domain-density-stage strata.
    while uncovered and len(selected) < limit:
        best_row = max(candidates, key=lambda row: len(labels(row) & uncovered))
        newly_covered = labels(best_row) & uncovered
        if not newly_covered:
            break
        selected.append({**best_row, "selection_phase": "marginal_group_coverage"})
        selected_ids.add(int(best_row["image_id"]))
        uncovered.difference_update(newly_covered)

    buckets: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if int(row["image_id"]) in selected_ids:
            continue
        stratum = tuple(str(row.get(column, "UNKNOWN")) for column in grouping_columns)
        buckets[stratum].append(row)
    strata = sorted(buckets)
    generator.shuffle(strata)
    for stratum in strata:
        generator.shuffle(buckets[stratum])

    while len(selected) < limit:
        added = False
        for stratum in strata:
            if not buckets[stratum]:
                continue
            row = buckets[stratum].pop()
            selected.append({**row, "selection_phase": "joint_stratum_round_robin"})
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break

    for index, row in enumerate(selected, start=1):
        row["selection_rank"] = index
        row["selection_stratum"] = " | ".join(
            str(row.get(column, "UNKNOWN")) for column in grouping_columns
        )
    return selected


def summarize_forward_rows(rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    metrics = [
        "aswb_relative_change",
        "high_frequency_ratio_change",
        "msia_input_relative_change",
        "msia_output_relative_change",
        "decoder_logits_relative_change",
        "decoder_boxes_relative_change",
        "normal_mean_nodes_per_hyperedge",
        "bypass_mean_nodes_per_hyperedge",
        "topology_jaccard",
        "topology_change_fraction",
    ]
    summary: Dict[str, Any] = {
        "images": len(rows),
        "metrics": {metric: summarize_values([row[metric] for row in rows], seed=seed) for metric in metrics},
        "density_correlations": {},
    }
    object_counts = [float(row["instances"]) for row in rows]
    for metric in [
        "alpha_scale",
        "speed_scale",
        "dynamic_alpha",
        "dynamic_speed",
        "aswb_relative_change",
        "high_frequency_ratio_change",
        "topology_change_fraction",
        "msia_output_relative_change",
    ]:
        values = [float(row[metric]) for row in rows]
        summary["density_correlations"][metric] = {
            "spearman_rho": spearman(object_counts, values),
            "pearson_r": pearson(object_counts, values),
        }
    grouped_metrics = [
        "dynamic_alpha",
        "dynamic_speed",
        "high_frequency_ratio_change",
        "topology_change_fraction",
        "msia_output_relative_change",
        "decoder_boxes_relative_change",
    ]
    summary["grouped_metrics"] = {}
    for column in ("density_group", "domain", "growth_stage"):
        groups = sorted({str(row[column]) for row in rows if str(row.get(column, "UNKNOWN")) != "UNKNOWN"})
        summary["grouped_metrics"][column] = {}
        for group in groups:
            selected = [row for row in rows if str(row.get(column)) == group]
            summary["grouped_metrics"][column][group] = {
                "images": len(selected),
                **{
                    metric: summarize_values([float(row[metric]) for row in selected], seed=seed)
                    for metric in grouped_metrics
                },
            }
    return summary


def run_forward_diagnostics(
    repo_root: Path,
    manifest: Mapping[str, Any],
    logs_root: Optional[Path],
    images: Path,
    annotations: Path,
    output_dir: Path,
    device: str,
    num_workers: int,
    max_images: int,
    topology_chunk_size: int,
    high_frequency_fraction: float,
    allow_partial_load: bool,
    require_paper_settings: bool,
    dataset_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    seed = int(manifest["paper_settings"]["seed"])
    _annotation_data, image_rows, split_report = prepare_dataset_groups(
        manifest,
        annotations,
        output_dir,
        dataset_root,
    )
    selected_image_rows = stratified_forward_sample(image_rows, max_images, seed)
    if not selected_image_rows:
        raise RuntimeError("No GWHD test images are available for forward diagnostics")
    write_csv(output_dir / "forward_sample_manifest.csv", selected_image_rows)
    selected_by_id = {int(row["image_id"]): row for row in selected_image_rows}
    selected_image_ids = set(selected_by_id)

    runtime = build_runtime(
        repo_root,
        manifest,
        "Ours",
        logs_root,
        images,
        annotations,
        batch_size=1,
        num_workers=num_workers,
        device=device,
        allow_partial_load=allow_partial_load,
    )
    torch = runtime["torch"]
    model = runtime["model"]
    model.eval()
    wave_name, wave_module = require_unique_module(model, "WaveEncoderBlockV2")
    hyper_name, hyper_module = require_unique_module(model, "HyperGraphEnhance")
    core_name, core_module = require_unique_module(model, "HyperComputeCore")

    setting_report = module_setting_report(wave_module, hyper_module, manifest)
    setting_report.update(
        {
            "wave_module": wave_name,
            "hypergraph_module": hyper_name,
            "hypergraph_core": core_name,
            "checkpoint": runtime["checkpoint"],
        }
    )
    write_json(output_dir / "runtime_setting_conformance.json", setting_report)
    if require_paper_settings and not setting_report["all_match"]:
        failed = [key for key, matches in setting_report["checks"].items() if not matches]
        raise RuntimeError(f"Runtime settings do not match the diagnostic manifest: {', '.join(failed)}")

    captures: Dict[str, Any] = {}

    def capture_wave_input(_module: Any, inputs: Tuple[Any, ...]) -> None:
        captures["wave_input"] = detach_tree(inputs[0])

    def capture_wave_output(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
        captures["wave_output"] = detach_tree(output)

    def capture_core_input(_module: Any, inputs: Tuple[Any, ...]) -> None:
        captures["core_input"] = detach_tree(inputs[0])

    def capture_hyper_output(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
        captures["hyper_output"] = detach_tree(output)

    handles = [
        wave_module.register_forward_pre_hook(capture_wave_input),
        wave_module.register_forward_hook(capture_wave_output),
        core_module.register_forward_pre_hook(capture_core_input),
        hyper_module.register_forward_hook(capture_hyper_output),
    ]

    rows: List[Dict[str, Any]] = []
    threshold_value = getattr(core_module, "threshold", None)
    if threshold_value is None:
        threshold_value = getattr(hyper_module, "threshold", None)
    if threshold_value is None:
        raise RuntimeError("Unable to read the MSIA hypergraph threshold")
    threshold = float(threshold_value)
    try:
        with torch.no_grad():
            for samples, targets in runtime["dataloader"]:
                if len(rows) >= len(selected_image_ids):
                    break
                if samples.shape[0] != 1:
                    raise RuntimeError("Forward intervention diagnostics require batch size 1")
                image_id = int(targets[0]["image_id"].item())
                group_row = selected_by_id.get(image_id)
                if group_row is None:
                    continue
                samples = samples.to(runtime["device"], non_blocking=True)

                captures.clear()
                normal_model_output = model(samples)
                normal = dict(captures)
                required = {"wave_input", "wave_output", "core_input", "hyper_output"}
                if not required.issubset(normal):
                    raise RuntimeError(f"Missing normal forward captures: {sorted(required - set(normal))}")

                def bypass_wave(_module: Any, inputs: Tuple[Any, ...], _output: Any) -> Any:
                    return inputs[0]

                captures.clear()
                bypass_handle = wave_module.register_forward_hook(bypass_wave)
                try:
                    bypass_model_output = model(samples)
                finally:
                    bypass_handle.remove()
                bypass = dict(captures)
                if not {"core_input", "hyper_output"}.issubset(bypass):
                    raise RuntimeError("Missing bypass forward captures")

                parameters = wave_parameters(torch, wave_module, normal["wave_input"])
                spectral_input = spectral_high_frequency_ratio(torch, normal["wave_input"], high_frequency_fraction)
                spectral_output = spectral_high_frequency_ratio(torch, normal["wave_output"], high_frequency_fraction)
                topology = topology_comparison(
                    torch,
                    normal["core_input"],
                    bypass["core_input"],
                    threshold,
                    topology_chunk_size,
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "file_name": group_row.get("file_name", ""),
                        "instances": int(group_row["instances"]),
                        "density_group": group_row.get("density_group", "UNKNOWN"),
                        "density_source": group_row.get("density_source", "unavailable"),
                        "domain": group_row.get("domain", "UNKNOWN"),
                        "domain_source": group_row.get("domain_source", "unavailable"),
                        "growth_stage": group_row.get("growth_stage", "UNKNOWN"),
                        "stage_source": group_row.get("stage_source", "unavailable"),
                        "mean_instance_occlusion": group_row.get("mean_instance_occlusion"),
                        "occlusion_group": group_row.get("occlusion_group", "UNKNOWN"),
                        "selection_rank": int(group_row["selection_rank"]),
                        "selection_phase": group_row["selection_phase"],
                        "selection_stratum": group_row["selection_stratum"],
                        "alpha_scale": parameters["alpha_scale"],
                        "speed_scale": parameters["speed_scale"],
                        "dynamic_alpha": parameters["dynamic_alpha"],
                        "dynamic_speed": parameters["dynamic_speed"],
                        "alpha_mapping": parameters["alpha_mapping"],
                        "input_high_frequency_ratio": spectral_input,
                        "output_high_frequency_ratio": spectral_output,
                        "high_frequency_ratio_change": spectral_output - spectral_input,
                        "aswb_relative_change": relative_l2(torch, normal["wave_input"], normal["wave_output"]),
                        "msia_input_relative_change": relative_l2(torch, normal["core_input"], bypass["core_input"]),
                        "msia_output_relative_change": relative_l2(torch, normal["hyper_output"], bypass["hyper_output"]),
                        "decoder_logits_relative_change": relative_l2(
                            torch, normal_model_output["pred_logits"], bypass_model_output["pred_logits"]
                        ),
                        "decoder_boxes_relative_change": relative_l2(
                            torch, normal_model_output["pred_boxes"], bypass_model_output["pred_boxes"]
                        ),
                        **topology,
                    }
                )
    finally:
        for handle in handles:
            handle.remove()

    observed_ids = {int(row["image_id"]) for row in rows}
    missing_ids = sorted(selected_image_ids - observed_ids)
    if missing_ids:
        raise RuntimeError(
            f"The evaluation dataloader did not yield {len(missing_ids)} selected images; "
            f"first missing IDs: {missing_ids[:20]}"
        )
    rows.sort(key=lambda row: int(row["selection_rank"]))
    summary = summarize_forward_rows(rows, seed)
    summary.update(
        {
            "checkpoint": runtime["checkpoint"],
            "dataset_splits": split_report,
            "sampling": {
                "method": "deterministic marginal-group coverage followed by joint-stratum round robin",
                "seed": seed,
                "requested_images": max_images,
                "selected_images": len(selected_image_rows),
                "covered_density_groups": sorted(
                    {str(row["density_group"]) for row in selected_image_rows},
                    key=lambda group: group_sort_key("density", group, manifest["known_gwhd_test_domains"]),
                ),
                "covered_domains": sorted(
                    {str(row["domain"]) for row in selected_image_rows if row["domain"] != "UNKNOWN"},
                    key=lambda group: group_sort_key("domain", group, manifest["known_gwhd_test_domains"]),
                ),
                "covered_growth_stages": sorted(
                    {
                        str(row["growth_stage"])
                        for row in selected_image_rows
                        if row["growth_stage"] != "UNKNOWN"
                    },
                    key=lambda group: group_sort_key("stage", group, manifest["known_gwhd_test_domains"]),
                ),
            },
            "topology_threshold": threshold,
            "topology_chunk_size": topology_chunk_size,
            "high_frequency_fraction": high_frequency_fraction,
            "intervention": "ASWB output is replaced by its input while all other weights remain fixed",
        }
    )
    write_csv(output_dir / "forward_coupling_per_image.csv", rows)
    write_json(output_dir / "forward_coupling_summary.json", summary)
    return rows


def normalized_box_occlusion(torch: Any, boxes_cxcywh: Any) -> Any:
    if boxes_cxcywh.numel() == 0:
        return boxes_cxcywh.new_zeros((0,))
    centers = boxes_cxcywh
    x1 = centers[:, 0] - centers[:, 2] / 2
    y1 = centers[:, 1] - centers[:, 3] / 2
    x2 = centers[:, 0] + centers[:, 2] / 2
    y2 = centers[:, 1] + centers[:, 3] / 2
    boxes = torch.stack((x1, y1, x2, y2), dim=1)
    left_top = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    right_bottom = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    intersections = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    intersections.fill_diagonal_(0)
    areas = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return intersections.sum(dim=1) / areas.clamp_min(1e-12)


def scheduled_beta(epoch: int, schedule_epochs: int, beta_start: float, beta_end: float) -> float:
    progress = min(max(float(epoch), 0.0), float(schedule_epochs)) / max(float(schedule_epochs), 1.0)
    return beta_start + (beta_end - beta_start) * progress


def ugdr_objectives(
    torch: Any,
    outputs: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    base_criterion: Any,
    epoch: int,
    schedule_epochs: int,
    beta_start: float,
    beta_end: float,
) -> Optional[Dict[str, Any]]:
    import torch.nn.functional as functional
    from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou
    from engine.deim.dfine_utils import bbox2distance

    outputs_without_aux = {key: value for key, value in outputs.items() if "aux" not in key}
    matcher_result = base_criterion.matcher(
        outputs_without_aux,
        targets,
        epoch=epoch,
        num_queries_list=outputs.get("num_queries_list"),
    )
    indices = matcher_result["indices"] if isinstance(matcher_result, Mapping) else matcher_result
    index = base_criterion._get_src_permutation_idx(indices)
    if index[0].numel() == 0:
        return None

    target_boxes = torch.cat([target["boxes"][target_ids] for target, (_, target_ids) in zip(targets, indices)], dim=0)
    predicted_boxes = outputs["pred_boxes"][index]
    predicted_corner_logits = outputs["pred_corners"][index]
    number_boxes = int(predicted_corner_logits.shape[0])
    if predicted_corner_logits.ndim == 3 and predicted_corner_logits.shape[1] == 4:
        number_bins = int(predicted_corner_logits.shape[2])
    elif predicted_corner_logits.ndim == 2 and predicted_corner_logits.shape[1] % 4 == 0:
        number_bins = int(predicted_corner_logits.shape[1] // 4)
    else:
        raise RuntimeError(f"Unexpected pred_corners shape: {tuple(predicted_corner_logits.shape)}")
    reg_max = number_bins - 1
    predicted_corners = predicted_corner_logits.reshape(number_boxes, 4, number_bins)

    reference_points = outputs["ref_points"][index].detach()
    with torch.no_grad():
        target_corners, weight_right, weight_left = bbox2distance(
            reference_points,
            box_cxcywh_to_xyxy(target_boxes),
            reg_max,
            outputs["reg_scale"],
            outputs["up"],
        )
        target_corners = target_corners.reshape(number_boxes, 4).clamp(0, reg_max - 1e-4)
        weight_right = weight_right.reshape(number_boxes, 4)
        weight_left = weight_left.reshape(number_boxes, 4)
        ious = torch.diag(
            box_iou(box_cxcywh_to_xyxy(predicted_boxes), box_cxcywh_to_xyxy(target_boxes))[0]
        ).detach()

    logits = predicted_corners.reshape(-1, number_bins)
    flattened_targets = target_corners.reshape(-1)
    left = flattened_targets.long().clamp(max=reg_max - 1)
    right = left + 1
    per_corner_loss = (
        functional.cross_entropy(logits, left, reduction="none") * weight_left.reshape(-1)
        + functional.cross_entropy(logits, right, reduction="none") * weight_right.reshape(-1)
    ).reshape(number_boxes, 4)
    per_corner_loss = per_corner_loss * ious[:, None]

    probabilities = functional.softmax(predicted_corners, dim=-1)
    entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=-1)
    bins = torch.arange(number_bins, device=probabilities.device, dtype=probabilities.dtype)
    mean = (probabilities * bins).sum(dim=-1)
    variance = (probabilities * (bins - mean.unsqueeze(-1)).pow(2)).sum(dim=-1)
    normalized_entropy = entropy / math.log(number_bins)
    normalized_variance = variance / max((reg_max ** 2) / 12.0, 1e-8)
    corner_uncertainty = (normalized_entropy + normalized_variance).mul(0.5).clamp(0.0, 1.0)
    box_uncertainty = corner_uncertainty.mean(dim=1)

    beta = scheduled_beta(epoch, schedule_epochs, beta_start, beta_end)
    reliability = beta + (1.0 - beta) * (1.0 - box_uncertainty)
    fgl_weight = float(getattr(base_criterion, "weight_dict", {}).get("loss_fgl", 1.0))
    denominator = max(float(number_boxes), 1.0)
    standard_loss = fgl_weight * per_corner_loss.sum() / denominator
    reweighted_loss = fgl_weight * (per_corner_loss * reliability.detach()[:, None]).sum() / denominator
    ugdr_loss = fgl_weight * (per_corner_loss * reliability[:, None]).sum() / denominator

    box_rows: List[Dict[str, Any]] = []
    offset = 0
    for batch_index, (_source_ids, target_ids) in enumerate(indices):
        image_id = int(targets[batch_index]["image_id"].item())
        target_occlusion = normalized_box_occlusion(torch, targets[batch_index]["boxes"])
        for target_id in target_ids.tolist():
            box_rows.append(
                {
                    "image_id": image_id,
                    "target_index": int(target_id),
                    "uncertainty": float(box_uncertainty[offset].detach().item()),
                    "reliability": float(reliability[offset].detach().item()),
                    "iou": float(ious[offset].item()),
                    "localization_error": float(1.0 - ious[offset].item()),
                    "occlusion_ratio": float(target_occlusion[target_id].detach().item()),
                }
            )
            offset += 1

    return {
        "standard_loss": standard_loss,
        "reweighted_loss": reweighted_loss,
        "ugdr_loss": ugdr_loss,
        "box_rows": box_rows,
        "reg_max": reg_max,
        "beta": beta,
        "fgl_weight": fgl_weight,
        "matched_boxes": number_boxes,
    }


def unique_trainable_parameters(items: Sequence[Any], recurse: bool = True) -> List[Any]:
    parameters: List[Any] = []
    seen: set = set()
    for item in items:
        if item is None:
            continue
        if hasattr(item, "parameters"):
            candidates = item.parameters(recurse=recurse)
        elif hasattr(item, "requires_grad"):
            candidates = [item]
        else:
            continue
        for parameter in candidates:
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            parameters.append(parameter)
            seen.add(id(parameter))
    return parameters


def collect_parameter_group_layout(model: Any) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Build non-overlapping leaf groups plus exact module-level aggregates."""
    _wave_name, wave_module = require_unique_module(model, "WaveEncoderBlockV2")
    _hyper_name, hyper_module = require_unique_module(model, "HyperGraphEnhance")
    restore_convs = getattr(hyper_module, "restore_convs", None)
    scattering_items = list(restore_convs.parameters()) if hasattr(restore_convs, "parameters") else []
    scattering_items.extend(list(hyper_module.parameters(recurse=False)))

    leaf_specs: List[Tuple[str, str, List[Any]]] = [
        (
            "ASWB",
            "Parameter generator",
            unique_trainable_parameters([getattr(wave_module, "param_generator", None)]),
        ),
        (
            "ASWB",
            "Wave propagation",
            unique_trainable_parameters([getattr(wave_module, "wave_op", None)]),
        ),
        (
            "ASWB",
            "Self-modulation",
            unique_trainable_parameters([wave_module], recurse=False),
        ),
        (
            "ASWB",
            "Feed-forward",
            unique_trainable_parameters(
                [
                    getattr(wave_module, "linear1", None),
                    getattr(wave_module, "linear2", None),
                    getattr(wave_module, "norm1", None),
                    getattr(wave_module, "norm2", None),
                ]
            ),
        ),
        (
            "MSIA",
            "Semantic collection",
            unique_trainable_parameters([getattr(hyper_module, "fusion_conv", None)]),
        ),
        (
            "MSIA",
            "Hypergraph interaction",
            unique_trainable_parameters([getattr(hyper_module, "hyper_compute", None)]),
        ),
        (
            "MSIA",
            "Semantic scattering",
            unique_trainable_parameters(scattering_items),
        ),
        (
            "Decoder",
            "Detection decoder",
            unique_trainable_parameters([model.decoder]),
        ),
    ]

    owner_parameters = {
        "ASWB": unique_trainable_parameters([wave_module]),
        "MSIA": unique_trainable_parameters([hyper_module]),
        "Decoder": unique_trainable_parameters([model.decoder]),
    }
    ordered_specs: List[Tuple[str, str, List[Any]]] = []
    for module_name in ("ASWB", "MSIA", "Decoder"):
        selected_specs = [spec for spec in leaf_specs if spec[0] == module_name]
        grouped_ids = {id(parameter) for _module, _component, group in selected_specs for parameter in group}
        missing = [parameter for parameter in owner_parameters[module_name] if id(parameter) not in grouped_ids]
        ordered_specs.extend(selected_specs)
        if missing:
            ordered_specs.append((module_name, "Other trainable parameters", missing))
    leaf_specs = ordered_specs

    flat_parameters: List[Any] = []
    leaf_layout: List[Dict[str, Any]] = []
    globally_seen: set = set()
    for module_name, component, parameters in leaf_specs:
        if not parameters:
            continue
        overlap = [parameter for parameter in parameters if id(parameter) in globally_seen]
        if overlap:
            raise RuntimeError(f"Overlapping gradient parameter groups detected for {module_name}/{component}")
        start = len(flat_parameters)
        flat_parameters.extend(parameters)
        stop = len(flat_parameters)
        globally_seen.update(id(parameter) for parameter in parameters)
        leaf_layout.append(
            {
                "parameter_group": f"{module_name}/{component}",
                "module": module_name,
                "component": component,
                "group_level": "component",
                "slice": slice(start, stop),
                "parameter_tensors": len(parameters),
                "parameters": sum(int(parameter.numel()) for parameter in parameters),
            }
        )

    expected_ids = {id(parameter) for parameters in owner_parameters.values() for parameter in parameters}
    if globally_seen != expected_ids:
        raise RuntimeError("Gradient parameter layout does not cover each ASWB/MSIA/decoder parameter exactly once")

    module_layout: List[Dict[str, Any]] = []
    for module_name in ("ASWB", "MSIA", "Decoder"):
        selected = [row for row in leaf_layout if row["module"] == module_name]
        if not selected:
            raise RuntimeError(f"Empty gradient parameter module: {module_name}")
        start = min(row["slice"].start for row in selected)
        stop = max(row["slice"].stop for row in selected)
        module_layout.append(
            {
                "parameter_group": module_name,
                "module": module_name,
                "component": "All",
                "group_level": "module",
                "slice": slice(start, stop),
                "parameter_tensors": sum(int(row["parameter_tensors"]) for row in selected),
                "parameters": sum(int(row["parameters"]) for row in selected),
            }
        )
    return flat_parameters, module_layout + leaf_layout


def gradients_for_loss(torch: Any, loss: Any, parameters: Sequence[Any], retain_graph: bool) -> Tuple[Any, ...]:
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )


def gradient_vector_stats(torch: Any, first: Sequence[Any], second: Sequence[Any]) -> Dict[str, float]:
    first_sq = None
    second_sq = None
    dot = None
    difference_sq = None
    used = 0
    for left, right in zip(first, second):
        if left is None and right is None:
            continue
        if left is None:
            left = torch.zeros_like(right)
        if right is None:
            right = torch.zeros_like(left)
        left = left.float()
        right = right.float()
        left_value = torch.sum(left * left)
        right_value = torch.sum(right * right)
        dot_value = torch.sum(left * right)
        difference = right - left
        difference_value = torch.sum(difference * difference)
        first_sq = left_value if first_sq is None else first_sq + left_value
        second_sq = right_value if second_sq is None else second_sq + right_value
        dot = dot_value if dot is None else dot + dot_value
        difference_sq = difference_value if difference_sq is None else difference_sq + difference_value
        used += 1
    if used == 0:
        return {
            "first_norm": 0.0,
            "second_norm": 0.0,
            "norm_ratio": 0.0,
            "cosine": 0.0,
            "relative_difference": 0.0,
            "used_tensors": 0,
        }
    first_norm = torch.sqrt(first_sq.clamp_min(0))
    second_norm = torch.sqrt(second_sq.clamp_min(0))
    difference_norm = torch.sqrt(difference_sq.clamp_min(0))
    denominator = (first_norm * second_norm).clamp_min(1e-20)
    cosine = float((dot / denominator).item())
    return {
        "first_norm": float(first_norm.item()),
        "second_norm": float(second_norm.item()),
        "norm_ratio": float((second_norm / first_norm.clamp_min(1e-20)).item()),
        "cosine": max(-1.0, min(1.0, cosine)),
        "relative_difference": float((difference_norm / first_norm.clamp_min(1e-20)).item()),
        "used_tensors": used,
    }


def gradient_decomposition_stats(
    torch: Any,
    standard: Sequence[Any],
    reweighted: Sequence[Any],
    full: Sequence[Any],
) -> Dict[str, float]:
    standard_full = gradient_vector_stats(torch, standard, full)
    reweighted_full = gradient_vector_stats(torch, reweighted, full)
    uncertainty_term: List[Any] = []
    zero_reference: List[Any] = []
    for weighted_gradient, full_gradient in zip(reweighted, full):
        if weighted_gradient is None and full_gradient is None:
            uncertainty_term.append(None)
            zero_reference.append(None)
            continue
        if weighted_gradient is None:
            weighted_gradient = torch.zeros_like(full_gradient)
        if full_gradient is None:
            full_gradient = torch.zeros_like(weighted_gradient)
        term = full_gradient - weighted_gradient
        uncertainty_term.append(term)
        zero_reference.append(torch.zeros_like(term))
    term_stats = gradient_vector_stats(torch, zero_reference, uncertainty_term)
    full_norm = standard_full["second_norm"]
    return {
        "standard_norm": standard_full["first_norm"],
        "ugdr_norm": full_norm,
        "ugdr_to_standard_norm_ratio": standard_full["norm_ratio"],
        "standard_ugdr_cosine": standard_full["cosine"],
        "standard_ugdr_relative_difference": standard_full["relative_difference"],
        "reweighted_norm": reweighted_full["first_norm"],
        "reweighted_ugdr_cosine": reweighted_full["cosine"],
        "uncertainty_term_norm": term_stats["second_norm"],
        "uncertainty_term_fraction_of_ugdr": term_stats["second_norm"] / max(full_norm, 1e-20),
        "used_parameter_tensors": standard_full["used_tensors"],
    }


def uncertainty_quartiles(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row["uncertainty"]))
    if not ordered:
        return []
    quartiles: List[Dict[str, Any]] = []
    for quartile in range(4):
        start = len(ordered) * quartile // 4
        stop = len(ordered) * (quartile + 1) // 4
        selected = ordered[start:stop]
        if not selected:
            continue
        quartiles.append(
            {
                "quartile": quartile + 1,
                "n": len(selected),
                "uncertainty": statistics.fmean(float(row["uncertainty"]) for row in selected),
                "reliability": statistics.fmean(float(row["reliability"]) for row in selected),
                "iou": statistics.fmean(float(row["iou"]) for row in selected),
                "localization_error": statistics.fmean(float(row["localization_error"]) for row in selected),
                "occlusion_ratio": statistics.fmean(float(row["occlusion_ratio"]) for row in selected),
            }
        )
    return quartiles


def clustered_spearman_ci(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    cluster_key: str,
    seed: int,
    iterations: int = 1000,
) -> List[Optional[float]]:
    clusters: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row[cluster_key])].append(row)
    cluster_names = sorted(clusters)
    if len(cluster_names) < 3:
        return [None, None]
    generator = random.Random(seed)
    estimates: List[float] = []
    for _ in range(iterations):
        sampled_rows: List[Mapping[str, Any]] = []
        for _cluster in cluster_names:
            sampled_rows.extend(clusters[generator.choice(cluster_names)])
        estimate = spearman(
            [float(row[x_key]) for row in sampled_rows],
            [float(row[y_key]) for row in sampled_rows],
        )
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return [None, None]
    estimates.sort()
    return [
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    ]


def summarize_backward_rows(
    gradient_rows: Sequence[Mapping[str, Any]],
    box_rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    gradient_summary: Dict[str, Any] = {}
    groups = sorted({str(row["parameter_group"]) for row in gradient_rows})
    gradient_metrics = [
        "standard_norm",
        "reweighted_norm",
        "uncertainty_term_norm",
        "ugdr_norm",
        "ugdr_to_standard_norm_ratio",
        "standard_ugdr_cosine",
        "uncertainty_term_fraction_of_ugdr",
    ]
    for group in groups:
        selected = [row for row in gradient_rows if row["parameter_group"] == group]
        group_summary = {
            metric: summarize_values([float(row[metric]) for row in selected], seed=seed)
            for metric in gradient_metrics
        }
        nonzero = sum(float(row["uncertainty_term_norm"]) > 1e-12 for row in selected)
        first = selected[0]
        group_summary.update(
            {
                "module": first.get("module", group.split("/", 1)[0]),
                "component": first.get("component", "All"),
                "group_level": first.get("group_level", "module" if "/" not in group else "component"),
                "parameter_tensors": int(first.get("parameter_tensors", 0)),
                "parameters": int(first.get("parameters", 0)),
                "nonzero_uncertainty_term_batches": nonzero,
                "nonzero_uncertainty_term_fraction": nonzero / len(selected) if selected else None,
            }
        )
        gradient_summary[group] = group_summary

    uncertainties = [float(row["uncertainty"]) for row in box_rows]
    errors = [float(row["localization_error"]) for row in box_rows]
    occlusions = [float(row["occlusion_ratio"]) for row in box_rows]
    return {
        "gradient_batches": len({int(row["batch"]) for row in gradient_rows}),
        "matched_boxes": len(box_rows),
        "gradient_groups": gradient_summary,
        "uncertainty_correlations": {
            "uncertainty_vs_localization_error_spearman": spearman(uncertainties, errors),
            "uncertainty_vs_localization_error_pearson": pearson(uncertainties, errors),
            "uncertainty_vs_occlusion_spearman": spearman(uncertainties, occlusions),
            "uncertainty_vs_occlusion_pearson": pearson(uncertainties, occlusions),
        },
        "uncertainty_correlation_clustered_ci95": {
            "uncertainty_vs_localization_error_spearman": clustered_spearman_ci(
                box_rows,
                "uncertainty",
                "localization_error",
                "image_id",
                seed,
            ),
            "uncertainty_vs_occlusion_spearman": clustered_spearman_ci(
                box_rows,
                "uncertainty",
                "occlusion_ratio",
                "image_id",
                seed + 1,
            ),
        },
        "uncertainty_quartiles": uncertainty_quartiles(box_rows),
    }


def run_backward_diagnostics(
    repo_root: Path,
    manifest: Mapping[str, Any],
    logs_root: Optional[Path],
    images: Path,
    annotations: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    max_images: int,
    allow_partial_load: bool,
    require_paper_settings: bool,
    dataset_root: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if batch_size != 1:
        raise ValueError("Stratified backward diagnostics require --batch-size 1")
    if max_images <= 0:
        raise ValueError("Backward diagnostic sample size must be positive")

    _annotation_data, image_rows, split_report = prepare_dataset_groups(
        manifest,
        annotations,
        output_dir,
        dataset_root,
    )
    seed = int(manifest["paper_settings"]["seed"])
    replacement_margin = max(16, max_images // 2)
    candidate_limit = min(len(image_rows), max_images + replacement_margin)
    candidate_rows = stratified_forward_sample(image_rows, candidate_limit, seed)
    candidate_by_id = {int(row["image_id"]): row for row in candidate_rows}
    candidate_ids = set(candidate_by_id)

    runtime = build_runtime(
        repo_root,
        manifest,
        "Ours",
        logs_root,
        images,
        annotations,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        allow_partial_load=allow_partial_load,
        normalized_targets=True,
    )
    torch = runtime["torch"]
    model = runtime["model"]
    criterion = runtime["criterion"]
    model.train()
    for module in model.modules():
        if isinstance(
            module,
            (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd),
        ):
            module.eval()
    base_criterion = getattr(criterion, "base_criterion", criterion)
    base_criterion.train()
    flat_parameters, parameter_layout = collect_parameter_group_layout(model)

    settings = manifest["paper_settings"]
    spec = manifest["variants"]["Ours"]
    args_path = args_path_for_variant(logs_root, spec)
    args_data = read_json(args_path) if args_path else None
    stage2_start_epoch = logged_stage2_start_epoch(args_data, manifest)
    checkpoint_record, checkpoint_epoch_source = checkpoint_log_target(
        runtime["checkpoint"],
        log_metric_records(logs_root, spec),
        stage2_start_epoch,
        None,
    )
    recorded_epoch = runtime["checkpoint"].get("recorded_epoch")
    checkpoint_log_epoch = (
        int(checkpoint_record["epoch"])
        if checkpoint_record is not None
        else int(recorded_epoch)
        if recorded_epoch is not None
        else int(settings["endpoint_epoch"])
    )
    if checkpoint_record is None:
        checkpoint_epoch_source = "checkpoint_metadata" if recorded_epoch is not None else "manifest_fallback"

    ugdr_module = getattr(criterion, "ugdr", None)
    schedule_epochs = int(getattr(criterion, "total_epochs", settings["ugdr_schedule_epochs"]))
    beta_start = float(getattr(ugdr_module, "beta_start", settings["ugdr_beta_start"]))
    beta_end = float(getattr(ugdr_module, "beta_end", settings["ugdr_beta_end"]))
    candidate_gradient_rows: List[Dict[str, Any]] = []
    candidate_box_rows: List[Dict[str, Any]] = []
    candidate_status: Dict[int, Dict[str, Any]] = {
        image_id: {
            "image_id": image_id,
            "candidate_rank": int(row["selection_rank"]),
            "status": "not_yielded",
            "matched_boxes": 0,
            "selected_for_analysis": False,
        }
        for image_id, row in candidate_by_id.items()
    }
    successful_ids: set = set()
    processed_ids: set = set()
    observed_reg_max = None
    observed_beta = None
    fgl_weight = None

    for loader_batch, (samples, targets) in enumerate(runtime["dataloader"]):
        if samples.shape[0] != 1:
            raise RuntimeError("Stratified backward diagnostics require dataloader batch size 1")
        image_id = int(targets[0]["image_id"].item())
        group_row = candidate_by_id.get(image_id)
        if group_row is None:
            continue
        processed_ids.add(image_id)
        candidate_rank = int(group_row["selection_rank"])
        torch.manual_seed(seed + candidate_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + candidate_rank)
        samples = samples.to(runtime["device"], non_blocking=True)
        device_targets = move_targets_to_device(targets, runtime["device"])
        outputs = model(samples, targets=device_targets)
        objective = ugdr_objectives(
            torch,
            outputs,
            device_targets,
            base_criterion,
            checkpoint_log_epoch,
            schedule_epochs,
            beta_start,
            beta_end,
        )
        if objective is None:
            candidate_status[image_id]["status"] = "no_matched_positive"
            model.zero_grad(set_to_none=True)
            if processed_ids == candidate_ids:
                break
            continue
        successful_ids.add(image_id)
        candidate_status[image_id]["status"] = "success"
        candidate_status[image_id]["matched_boxes"] = int(objective["matched_boxes"])
        observed_reg_max = objective["reg_max"]
        observed_beta = objective["beta"]
        fgl_weight = objective["fgl_weight"]
        standard_gradients = gradients_for_loss(
            torch, objective["standard_loss"], flat_parameters, retain_graph=True
        )
        reweighted_gradients = gradients_for_loss(
            torch, objective["reweighted_loss"], flat_parameters, retain_graph=True
        )
        ugdr_gradients = gradients_for_loss(
            torch, objective["ugdr_loss"], flat_parameters, retain_graph=False
        )
        for group in parameter_layout:
            group_slice = group["slice"]
            stats = gradient_decomposition_stats(
                torch,
                standard_gradients[group_slice],
                reweighted_gradients[group_slice],
                ugdr_gradients[group_slice],
            )
            candidate_gradient_rows.append(
                {
                    "batch": loader_batch,
                    "image_id": image_id,
                    "candidate_rank": candidate_rank,
                    "density_group": group_row.get("density_group", "UNKNOWN"),
                    "domain": group_row.get("domain", "UNKNOWN"),
                    "growth_stage": group_row.get("growth_stage", "UNKNOWN"),
                    "parameter_group": group["parameter_group"],
                    "module": group["module"],
                    "component": group["component"],
                    "group_level": group["group_level"],
                    "parameter_tensors": group["parameter_tensors"],
                    "parameters": group["parameters"],
                    "matched_boxes": objective["matched_boxes"],
                    "standard_loss": float(objective["standard_loss"].detach().item()),
                    "reweighted_loss": float(objective["reweighted_loss"].detach().item()),
                    "ugdr_loss": float(objective["ugdr_loss"].detach().item()),
                    "beta": objective["beta"],
                    **stats,
                }
            )
        for row in objective["box_rows"]:
            candidate_box_rows.append(
                {
                    "batch": loader_batch,
                    "candidate_rank": candidate_rank,
                    "density_group": group_row.get("density_group", "UNKNOWN"),
                    "domain": group_row.get("domain", "UNKNOWN"),
                    "growth_stage": group_row.get("growth_stage", "UNKNOWN"),
                    **row,
                }
            )
        model.zero_grad(set_to_none=True)
        if processed_ids == candidate_ids:
            break

    missing_candidate_ids = sorted(candidate_ids - processed_ids)
    if missing_candidate_ids:
        raise RuntimeError(
            f"The evaluation dataloader did not yield {len(missing_candidate_ids)} backward candidates; "
            f"first missing IDs: {missing_candidate_ids[:20]}"
        )
    write_csv(
        output_dir / "backward_candidate_status.csv",
        sorted(candidate_status.values(), key=lambda row: int(row["candidate_rank"])),
    )
    successful_rows = [row for row in candidate_rows if int(row["image_id"]) in successful_ids]
    selected_rows = stratified_forward_sample(successful_rows, max_images, seed)
    if len(selected_rows) < min(max_images, len(image_rows)):
        raise RuntimeError(
            f"Only {len(selected_rows)} images produced matched positives; requested {max_images}. "
            "Increase the candidate margin or inspect backward_candidate_status.csv."
        )
    selected_by_id = {int(row["image_id"]): row for row in selected_rows}
    selected_ids = set(selected_by_id)
    for image_id in selected_ids:
        candidate_status[image_id]["selected_for_analysis"] = True

    gradient_rows = [row for row in candidate_gradient_rows if int(row["image_id"]) in selected_ids]
    box_rows = [row for row in candidate_box_rows if int(row["image_id"]) in selected_ids]
    for row in gradient_rows:
        row["selection_rank"] = int(selected_by_id[int(row["image_id"])]["selection_rank"])
    for row in box_rows:
        row["selection_rank"] = int(selected_by_id[int(row["image_id"])]["selection_rank"])
    gradient_rows.sort(key=lambda row: (int(row["selection_rank"]), str(row["parameter_group"])))
    box_rows.sort(key=lambda row: (int(row["selection_rank"]), int(row["target_index"])))

    sample_manifest = [
        {
            **row,
            "matched_boxes": int(candidate_status[int(row["image_id"])]["matched_boxes"]),
        }
        for row in selected_rows
    ]
    write_csv(output_dir / "backward_sample_manifest.csv", sample_manifest)
    write_csv(
        output_dir / "backward_candidate_status.csv",
        sorted(candidate_status.values(), key=lambda row: int(row["candidate_rank"])),
    )

    conformance_path = output_dir / "runtime_setting_conformance.json"
    conformance = read_json(conformance_path) if conformance_path.is_file() else {"expected": {}, "observed": {}, "checks": {}}
    expected_schedule_epochs = int(settings["ugdr_schedule_epochs"])
    expected_beta_start = float(settings["ugdr_beta_start"])
    expected_beta_end = float(settings["ugdr_beta_end"])
    expected_checkpoint_beta = scheduled_beta(
        checkpoint_log_epoch,
        expected_schedule_epochs,
        expected_beta_start,
        expected_beta_end,
    )
    conformance.setdefault("expected", {}).update(
        {
            "ugdr_reg_max": int(settings["ugdr_reg_max"]),
            "ugdr_schedule_epochs": expected_schedule_epochs,
            "ugdr_beta_range": [expected_beta_start, expected_beta_end],
            "ugdr_beta_at_checkpoint": expected_checkpoint_beta,
        }
    )
    conformance.setdefault("observed", {}).update(
        {
            "ugdr_reg_max": observed_reg_max,
            "ugdr_schedule_epochs": schedule_epochs,
            "ugdr_beta_range": [beta_start, beta_end],
            "ugdr_beta_at_checkpoint": observed_beta,
        }
    )
    conformance.setdefault("checks", {}).update(
        {
            "ugdr_reg_max": observed_reg_max == int(settings["ugdr_reg_max"]),
            "ugdr_schedule_epochs": schedule_epochs == expected_schedule_epochs,
            "ugdr_beta_range": math.isclose(beta_start, expected_beta_start, abs_tol=1e-8)
            and math.isclose(beta_end, expected_beta_end, abs_tol=1e-8),
            "ugdr_beta_at_checkpoint": observed_beta is not None
            and math.isclose(observed_beta, expected_checkpoint_beta, abs_tol=1e-8),
        }
    )
    conformance["all_match"] = all(conformance["checks"].values())
    write_json(conformance_path, conformance)
    if require_paper_settings and not conformance["all_match"]:
        failed = [key for key, matches in conformance["checks"].items() if not matches]
        raise RuntimeError(f"Runtime settings do not match the diagnostic manifest: {', '.join(failed)}")

    summary = summarize_backward_rows(gradient_rows, box_rows, seed)
    summary.update(
        {
            "checkpoint": runtime["checkpoint"],
            "checkpoint_log_epoch": checkpoint_log_epoch,
            "checkpoint_epoch_source": checkpoint_epoch_source,
            "schedule_epochs": schedule_epochs,
            "reg_max": observed_reg_max,
            "beta": observed_beta,
            "fgl_weight": fgl_weight,
            "dataset_splits": split_report,
            "sampling": {
                "method": "deterministic marginal-group coverage followed by joint-stratum round robin",
                "seed": seed,
                "requested_images": max_images,
                "candidate_images": len(candidate_rows),
                "successful_candidate_images": len(successful_rows),
                "selected_images": len(selected_rows),
                "covered_density_groups": sorted(
                    {str(row["density_group"]) for row in selected_rows},
                    key=lambda group: group_sort_key("density", group, manifest["known_gwhd_test_domains"]),
                ),
                "covered_domains": sorted(
                    {str(row["domain"]) for row in selected_rows if row["domain"] != "UNKNOWN"},
                    key=lambda group: group_sort_key("domain", group, manifest["known_gwhd_test_domains"]),
                ),
                "covered_growth_stages": sorted(
                    {
                        str(row["growth_stage"])
                        for row in selected_rows
                        if row["growth_stage"] != "UNKNOWN"
                    },
                    key=lambda group: group_sort_key("stage", group, manifest["known_gwhd_test_domains"]),
                ),
            },
            "parameter_group_layout": [
                {key: value for key, value in group.items() if key != "slice"}
                for group in parameter_layout
            ],
            "objective_definition": {
                "standard": "IoU-weighted FGL on matched positive boxes",
                "reweighted": "FGL multiplied by detached UGDR reliability",
                "ugdr": "FGL multiplied by differentiable UGDR reliability",
                "uncertainty_term": "gradient(ugdr) - gradient(reweighted)",
            },
        }
    )
    write_csv(output_dir / "backward_gradient_per_batch.csv", gradient_rows)
    write_csv(output_dir / "uncertainty_localization_per_box.csv", box_rows)
    write_json(output_dir / "backward_coupling_summary.json", summary)
    return gradient_rows, box_rows


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-images", required=True, help="GWHD test image directory")
    parser.add_argument("--data-annotations", required=True, help="GWHD test COCO annotation JSON")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys. Do not use such outputs as paper evidence.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--logs-root", default=None, help="Root containing the eight logs/<variant>/args.json files")
    parser.add_argument("--output-dir", default="outputs/synergy_diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit", help="Map logs, configs, module flags, and saved checkpoints")

    split_parser = subparsers.add_parser("audit-splits", help="Validate authoritative GWHD test partitions")
    split_parser.add_argument("--data-annotations", required=True)
    split_parser.add_argument(
        "--gwhd-root",
        default=None,
        help="GWHD root containing test_density, test_domain, and test_stage; inferred when omitted",
    )

    predict_parser = subparsers.add_parser("predict", help="Export COCO predictions for selected variants")
    add_data_arguments(predict_parser)
    add_runtime_arguments(predict_parser)
    predict_parser.add_argument("--batch-size", type=int, default=8)
    predict_parser.add_argument("--variants", default="Baseline,ASWB,MSIA,UGDR,Ours")

    evaluate_parser = subparsers.add_parser("evaluate", help="Compute module-to-problem grouped metrics")
    evaluate_parser.add_argument("--data-annotations", required=True)
    evaluate_parser.add_argument(
        "--gwhd-root",
        default=None,
        help="GWHD root containing test_density, test_domain, and test_stage; inferred when omitted",
    )
    evaluate_parser.add_argument("--predictions-dir", default=None)
    evaluate_parser.add_argument("--confidence", type=float, default=0.5)
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)

    forward_parser = subparsers.add_parser("forward", help="Measure the ASWB-to-MSIA forward coupling")
    add_data_arguments(forward_parser)
    add_runtime_arguments(forward_parser)
    forward_parser.add_argument(
        "--gwhd-root",
        default=None,
        help="GWHD root containing test_density, test_domain, and test_stage; inferred when omitted",
    )
    forward_parser.add_argument("--max-images", type=int, default=64)
    forward_parser.add_argument("--topology-chunk-size", type=int, default=256)
    forward_parser.add_argument("--high-frequency-fraction", type=float, default=0.5)
    forward_parser.add_argument("--require-paper-settings", action="store_true")

    backward_parser = subparsers.add_parser("backward", help="Measure UGDR gradient propagation")
    add_data_arguments(backward_parser)
    add_runtime_arguments(backward_parser)
    backward_parser.add_argument(
        "--gwhd-root",
        default=None,
        help="GWHD root containing test_density, test_domain, and test_stage; inferred when omitted",
    )
    backward_parser.add_argument("--batch-size", type=int, default=1)
    backward_parser.add_argument(
        "--max-images",
        "--max-batches",
        dest="max_images",
        type=int,
        default=64,
        help="Number of stratified images; --max-batches is retained as a compatibility alias",
    )
    backward_parser.add_argument("--require-paper-settings", action="store_true")

    all_parser = subparsers.add_parser("all", help="Run audit, targeted evaluation, and both coupling diagnostics")
    add_data_arguments(all_parser)
    add_runtime_arguments(all_parser)
    all_parser.add_argument("--prediction-batch-size", type=int, default=8)
    all_parser.add_argument(
        "--gwhd-root",
        default=None,
        help="GWHD root containing test_density, test_domain, and test_stage; inferred when omitted",
    )
    all_parser.add_argument("--variants", default="Baseline,ASWB,MSIA,UGDR,Ours")
    all_parser.add_argument("--confidence", type=float, default=0.5)
    all_parser.add_argument("--iou-threshold", type=float, default=0.5)
    all_parser.add_argument("--forward-max-images", type=int, default=64)
    all_parser.add_argument("--topology-chunk-size", type=int, default=256)
    all_parser.add_argument("--high-frequency-fraction", type=float, default=0.5)
    all_parser.add_argument("--gradient-batch-size", type=int, default=1)
    all_parser.add_argument(
        "--gradient-max-images",
        "--gradient-max-batches",
        dest="gradient_max_images",
        type=int,
        default=64,
        help="Number of stratified images for gradient diagnostics",
    )
    all_parser.add_argument("--require-paper-settings", action="store_true")

    subparsers.add_parser("self-test", help="Run dependency-free checks for grouping and statistics")
    return parser


def checked_data_paths(args: argparse.Namespace, repo_root: Path) -> Tuple[Path, Path]:
    images = resolve_path(args.data_images, repo_root)
    annotations = resolve_path(args.data_annotations, repo_root)
    if images is None or not images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images}")
    if annotations is None or not annotations.is_file():
        raise FileNotFoundError(f"Annotation JSON not found: {annotations}")
    return images, annotations


def predictions_from_directory(manifest: Mapping[str, Any], directory: Path) -> Dict[str, Path]:
    required = sorted(
        {
            variant
            for variants in manifest["targeted_comparisons"].values()
            for variant in variants
        }
    )
    paths = {variant: directory / f"{variant}.json" for variant in required}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files: {missing}")
    return paths


def run_self_test() -> None:
    assert density_group(0) == "0-10"
    assert density_group(10) == "0-10"
    assert density_group(11) == "10-20"
    assert density_group(100) == "80-100"
    assert density_group(101) == ">100"
    ratios = instance_occlusion_ratios([[0, 0, 10, 10], [5, 0, 10, 10]])
    assert all(math.isclose(value, 0.5, abs_tol=1e-12) for value in ratios)
    assert math.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]) or 0.0, 1.0, abs_tol=1e-12)
    assert math.isclose(spearman([1, 2, 3, 4], [40, 30, 20, 10]) or 0.0, -1.0, abs_tol=1e-12)
    assert occlusion_group(0.09).startswith("Low")
    assert occlusion_group(0.20).startswith("Medium")
    assert occlusion_group(0.31).startswith("High")
    assert split_group_label(Path("100+_annotations.json"), "density", []) == ">100"
    assert split_group_label(Path("Density_100+_annotations.json"), "density", []) == ">100"
    assert split_group_label(Path("post_flowering_annotations.json"), "stage", []) == "Post-Flowering"
    assert split_group_label(Path("Stage_Post_Flowering_annotations.json"), "stage", []) == "Post-Flowering"
    assert split_group_label(Path("ARC_1_annotations.json"), "domain", ["ARC_1"]) == "ARC_1"
    assert split_group_label(Path("Domain_ARC_1_annotations.json"), "domain", ["ARC_1"]) == "ARC_1"
    annotations = {
        "images": [{"id": 1, "file_name": "CIMMYT_1/example.jpg"}],
        "categories": [{"id": 1, "name": "wheat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [5, 0, 10, 10]},
        ],
    }
    instance_rows = build_instance_occlusion_groups(annotations)
    assert len(instance_rows) == 2
    full_images_by_id, full_ids_by_name = full_image_lookups(annotations)
    mapped_id, mapping_method = resolve_subset_image_id(
        {"id": 999, "file_name": "example.jpg"},
        full_images_by_id,
        full_ids_by_name,
    )
    assert mapped_id == 1 and mapping_method == "file_name"
    grouped_images = build_image_groups(
        annotations,
        ["CIMMYT_1"],
        {
            "density": {1: "40-50"},
            "domain": {1: "CIMMYT_1"},
            "stage": {1: "Filling"},
        },
    )
    assert grouped_images[0]["density_group"] == "40-50"
    assert grouped_images[0]["domain"] == "CIMMYT_1"
    assert grouped_images[0]["growth_stage"] == "Filling"
    sample_rows = [
        {"image_id": 1, "domain": "A", "density_group": "0-10", "growth_stage": "Filling"},
        {"image_id": 2, "domain": "B", "density_group": "10-20", "growth_stage": "Ripening"},
        {"image_id": 3, "domain": "A", "density_group": "20-30", "growth_stage": "Ripening"},
    ]
    sample = stratified_forward_sample(sample_rows, 3, 42)
    assert {row["image_id"] for row in sample} == {1, 2, 3}
    assert sample == stratified_forward_sample(sample_rows, 3, 42)
    with tempfile.TemporaryDirectory() as temporary_directory:
        dataset_root = Path(temporary_directory)
        split_labels = {
            "density": DENSITY_GROUP_ORDER,
            "domain": ["CIMMYT_1"],
            "stage": STAGE_GROUP_ORDER,
        }
        for split_name, labels in split_labels.items():
            annotation_dir = dataset_root / SPLIT_DIRECTORIES[split_name] / "annotations"
            for index, label in enumerate(labels):
                file_label = "100+" if label == ">100" else label.lower().replace("-", "_")
                subset = {
                    "images": annotations["images"] if index == 0 else [],
                    "annotations": annotations["annotations"] if index == 0 else [],
                    "categories": annotations["categories"],
                }
                write_json(annotation_dir / f"{file_label}_annotations.json", subset)
        assignments, split_rows, split_report = load_authoritative_splits(
            dataset_root,
            annotations,
            ["CIMMYT_1"],
        )
        assert split_report["all_valid"] and len(split_rows) == 15
        assert assignments["density"][1] == "0-10"
        assert assignments["domain"][1] == "CIMMYT_1"
        assert assignments["stage"][1] == "Post-Flowering"
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
    accuracy = instance_group_accuracy(annotations, predictions, {1}, [1], 0.5, 0.5)
    assert accuracy["TP"] == 1 and accuracy["FP"] == 0 and accuracy["FN"] == 0
    metric_records = [
        {"epoch": 31, "AP": 20.0, "AP50": 40.0, "AP75": 10.0},
        {"epoch": 32, "AP": 22.0, "AP50": 42.0, "AP75": 12.0},
        {"epoch": 40, "AP": 25.0, "AP50": 45.0, "AP75": 15.0},
    ]
    observed = {"AP": 25.01, "AP50": 45.01, "AP75": 15.01}
    closest = closest_log_record(metric_records[1:], observed)
    assert closest is not None and int(closest["record"]["epoch"]) == 40
    target, source = checkpoint_log_target(
        {"path": "/tmp/best_stg2.pth", "recorded_epoch": 39},
        metric_records,
        32,
        closest,
    )
    assert target is not None and int(target["epoch"]) == 40
    assert source == "best_stg2_selection_rule"
    assert is_stage2_checkpoint_candidate(Path("best_stg2.pth"), 32)
    assert not is_stage2_checkpoint_candidate(Path("last.pth"), 32)
    assert math.isclose(scheduled_beta(70, 160, 1.0, 0.1), 0.60625, abs_tol=1e-12)
    assert math.isclose(scheduled_beta(71, 160, 1.0, 0.1), 0.600625, abs_tol=1e-12)

    class FakeTensor:
        def __init__(self, shape: Sequence[int]):
            self.shape = tuple(shape)

        def reshape(self, shape: Sequence[int]) -> "FakeTensor":
            return FakeTensor(shape)

    class FakeTorch:
        Tensor = FakeTensor

    class FakeModel:
        @staticmethod
        def state_dict() -> Dict[str, FakeTensor]:
            return {
                "wave_speed": FakeTensor((1,)),
                "damping": FakeTensor((1,)),
                "other": FakeTensor((2,)),
            }

    legacy_state = {
        "wave_speed": FakeTensor(()),
        "damping": FakeTensor(()),
        "other": FakeTensor((3,)),
    }
    converted, reshaped = reshape_legacy_scalar_parameters(FakeTorch, FakeModel(), legacy_state)
    assert converted["wave_speed"].shape == (1,) and converted["damping"].shape == (1,)
    assert converted["other"] is legacy_state["other"]
    assert {row["parameter"] for row in reshaped} == {"wave_speed", "damping"}
    print("Self-test passed")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = resolve_path(args.repo_root, Path.cwd())
    manifest_path = resolve_path(args.manifest, Path.cwd())
    if repo_root is None or not repo_root.is_dir():
        raise FileNotFoundError(f"Repository root not found: {repo_root}")
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    logs_root = resolve_path(args.logs_root, repo_root) if args.logs_root else None
    output_dir = resolve_path(args.output_dir, repo_root)
    if output_dir is None:
        raise ValueError("Output directory cannot be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "self-test":
        run_self_test()
        return 0

    write_json(
        output_dir / "invocation.json",
        {
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "command": args.command,
            "repo_root": str(repo_root),
            "manifest": str(manifest_path),
            "logs_root": str(logs_root) if logs_root else None,
        },
    )

    if args.command == "audit":
        rows = audit_variants(repo_root, manifest, logs_root, output_dir)
        print(f"Wrote configuration audit for {len(rows)} variants to {output_dir}")
        return 0

    if args.command == "audit-splits":
        annotations = resolve_path(args.data_annotations, repo_root)
        if annotations is None or not annotations.is_file():
            raise FileNotFoundError(f"Annotation JSON not found: {annotations}")
        dataset_root = resolve_path(args.gwhd_root, repo_root) if args.gwhd_root else infer_gwhd_root(annotations)
        if dataset_root is None or not dataset_root.is_dir():
            raise FileNotFoundError(
                "Unable to locate the GWHD root containing test_density, test_domain, and test_stage; "
                "pass --gwhd-root explicitly"
            )
        _annotation_data, _image_rows, split_report = prepare_dataset_groups(
            manifest,
            annotations,
            output_dir,
            dataset_root,
        )
        coverage = ", ".join(
            f"{name}={report['assigned_images']}/{split_report['full_test_images']} ({report['coverage']:.1%})"
            for name, report in split_report["splits"].items()
        )
        print(f"Authoritative GWHD partitions validated: {coverage}")
        return 0

    if args.command == "predict":
        images, annotations = checked_data_paths(args, repo_root)
        variants = parse_variant_list(args.variants, manifest)
        paths = run_predictions(
            repo_root,
            manifest,
            logs_root,
            images,
            annotations,
            output_dir,
            variants,
            args.device,
            args.batch_size,
            args.num_workers,
            args.allow_partial_load,
        )
        print(f"Wrote {len(paths)} prediction files to {output_dir / 'predictions'}")
        return 0

    if args.command == "evaluate":
        annotations = resolve_path(args.data_annotations, repo_root)
        if annotations is None or not annotations.is_file():
            raise FileNotFoundError(f"Annotation JSON not found: {annotations}")
        predictions_dir = resolve_path(args.predictions_dir, repo_root) if args.predictions_dir else output_dir / "predictions"
        dataset_root = resolve_path(args.gwhd_root, repo_root) if args.gwhd_root else None
        if dataset_root is not None and not dataset_root.is_dir():
            raise FileNotFoundError(f"GWHD root not found: {dataset_root}")
        paths = predictions_from_directory(manifest, predictions_dir)
        rows = run_targeted_evaluation(
            manifest,
            logs_root,
            annotations,
            output_dir,
            paths,
            args.confidence,
            args.iou_threshold,
            dataset_root,
        )
        print(f"Wrote {len(rows)} grouped metric rows to {output_dir}")
        return 0

    if args.command == "forward":
        images, annotations = checked_data_paths(args, repo_root)
        dataset_root = resolve_path(args.gwhd_root, repo_root) if args.gwhd_root else None
        if dataset_root is not None and not dataset_root.is_dir():
            raise FileNotFoundError(f"GWHD root not found: {dataset_root}")
        rows = run_forward_diagnostics(
            repo_root,
            manifest,
            logs_root,
            images,
            annotations,
            output_dir,
            args.device,
            args.num_workers,
            args.max_images,
            args.topology_chunk_size,
            args.high_frequency_fraction,
            args.allow_partial_load,
            args.require_paper_settings,
            dataset_root,
        )
        print(f"Wrote forward diagnostics for {len(rows)} images to {output_dir}")
        return 0

    if args.command == "backward":
        images, annotations = checked_data_paths(args, repo_root)
        dataset_root = resolve_path(args.gwhd_root, repo_root) if args.gwhd_root else None
        if dataset_root is not None and not dataset_root.is_dir():
            raise FileNotFoundError(f"GWHD root not found: {dataset_root}")
        gradients, boxes = run_backward_diagnostics(
            repo_root,
            manifest,
            logs_root,
            images,
            annotations,
            output_dir,
            args.device,
            args.batch_size,
            args.num_workers,
            args.max_images,
            args.allow_partial_load,
            args.require_paper_settings,
            dataset_root,
        )
        print(f"Wrote {len(gradients)} gradient rows and {len(boxes)} matched-box rows to {output_dir}")
        return 0

    if args.command == "all":
        images, annotations = checked_data_paths(args, repo_root)
        dataset_root = resolve_path(args.gwhd_root, repo_root) if args.gwhd_root else None
        if dataset_root is not None and not dataset_root.is_dir():
            raise FileNotFoundError(f"GWHD root not found: {dataset_root}")
        audit_variants(repo_root, manifest, logs_root, output_dir)
        variants = parse_variant_list(args.variants, manifest)
        required = {
            variant
            for comparison in manifest["targeted_comparisons"].values()
            for variant in comparison
        }
        missing = sorted(required - set(variants))
        if missing:
            raise ValueError(f"The all command requires targeted variants: {missing}")
        paths = run_predictions(
            repo_root,
            manifest,
            logs_root,
            images,
            annotations,
            output_dir,
            variants,
            args.device,
            args.prediction_batch_size,
            args.num_workers,
            args.allow_partial_load,
        )
        run_targeted_evaluation(
            manifest,
            logs_root,
            annotations,
            output_dir,
            paths,
            args.confidence,
            args.iou_threshold,
            dataset_root,
        )
        run_forward_diagnostics(
            repo_root,
            manifest,
            logs_root,
            images,
            annotations,
            output_dir,
            args.device,
            args.num_workers,
            args.forward_max_images,
            args.topology_chunk_size,
            args.high_frequency_fraction,
            args.allow_partial_load,
            args.require_paper_settings,
            dataset_root,
        )
        run_backward_diagnostics(
            repo_root,
            manifest,
            logs_root,
            images,
            annotations,
            output_dir,
            args.device,
            args.gradient_batch_size,
            args.num_workers,
            args.gradient_max_images,
            args.allow_partial_load,
            args.require_paper_settings,
            dataset_root,
        )
        print(f"All diagnostics completed: {output_dir}")
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
