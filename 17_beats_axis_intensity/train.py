#!/usr/bin/env python3
"""Train one or all BEATs per-axis intensity models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    AXIS_NAMES,
    CLASS_SCHEMA,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BACKBONE_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIRS,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_HEAD_LEARNING_RATE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PATIENCE,
    DEFAULT_PRETRAINED_CHECKPOINT_PATH,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SEED,
    DEFAULT_SEGMENT_SECONDS,
    DEFAULT_VAL_SIZE,
    DEFAULT_WEIGHT_DECAY,
    artifact_path_for_axis,
    output_dir_for_axis,
    split_path_for_axis,
)
from dataset import create_train_val_split, save_split_manifest, scan_dataset  # noqa: E402
from shared.audio import ensure_directory  # noqa: E402
from shared.torch_training import choose_device, seed_everything  # noqa: E402
from utils import (  # noqa: E402
    build_model,
    class_weights_from_labels,
    create_dataloaders,
    create_optimizer,
    evaluate_axis_records,
    fit_axis_model,
)


def train_one_axis(args, axis_name: str, records, device: torch.device) -> dict:
    artifact_path = Path(args.artifact_dir) / f"{axis_name}_best_model.pt"
    split_path = Path(args.artifact_dir) / f"{axis_name}_split.json"
    latest_path = Path(args.artifact_dir) / f"{axis_name}_latest_training.pt"
    output_dir = Path(args.output_dir) / axis_name
    data_dirs = [str(Path(data_dir).expanduser().resolve()) for data_dir in args.data_dir]

    resume_state = None
    if args.resume and latest_path.exists():
        resume_state = torch.load(latest_path, map_location=device)
        if resume_state.get("class_schema") == CLASS_SCHEMA:
            print(f"[INFO] Loaded {axis_name} resume checkpoint: {latest_path.resolve()}", flush=True)
        else:
            resume_state = None
            print(
                f"[WARN] {axis_name} resume checkpoint uses an older class schema; starting fresh.",
                flush=True,
            )
    elif args.resume:
        print(f"[WARN] {axis_name} resume requested but checkpoint was not found: {latest_path.resolve()}", flush=True)

    if resume_state and split_path.exists():
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        if split_payload.get("data_dirs") == data_dirs:
            record_by_path = {record.path: record for record in records}
            split = {
                "axis_name": axis_name,
                "seed": split_payload["seed"],
                "val_size": split_payload["val_size"],
                "strategy": split_payload["strategy"],
                "notes": split_payload["notes"],
                "data_dirs": split_payload.get("data_dirs", []),
                "class_schema": split_payload.get("class_schema", CLASS_SCHEMA),
                "train_records": [record_by_path[item["path"]] for item in split_payload["train_records"]],
                "val_records": [record_by_path[item["path"]] for item in split_payload["val_records"]],
            }
            print(f"[INFO] Reusing {axis_name} split manifest: {split_path.resolve()}", flush=True)
        else:
            resume_state = None
            split = create_train_val_split(records, axis_name=axis_name, val_size=args.val_size, seed=args.seed)
            split["data_dirs"] = data_dirs
            split["class_schema"] = CLASS_SCHEMA
            save_split_manifest(split, split_path)
            print(
                f"[WARN] {axis_name} dataset roots changed; starting a fresh split instead of resuming old data.",
                flush=True,
            )
    else:
        split = create_train_val_split(records, axis_name=axis_name, val_size=args.val_size, seed=args.seed)
        split["data_dirs"] = data_dirs
        split["class_schema"] = CLASS_SCHEMA
        save_split_manifest(split, split_path)
    print(
        f"[INFO] {axis_name} records: total={len(records)} train={len(split['train_records'])} "
        f"val={len(split['val_records'])}",
        flush=True,
    )

    train_loader, val_loader = create_dataloaders(
        split["train_records"],
        split["val_records"],
        axis_name=axis_name,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pretrained_checkpoint_path = Path(args.pretrained_checkpoint).expanduser().resolve()
    pretrained_exists = pretrained_checkpoint_path.exists()
    model = build_model(axis_name, pretrained_checkpoint_path if pretrained_exists else None).to(device)
    print(f"[INFO] {axis_name} backbone pretrained loaded: {model.pretrained_loaded}", flush=True)

    train_labels = [record.target_for_axis(axis_name) for record in split["train_records"]]
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights_from_labels(train_labels).to(device),
        label_smoothing=0.02,
    )
    optimizer = create_optimizer(
        model,
        backbone_learning_rate=args.backbone_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    if resume_state:
        state_dict = resume_state.get("model_state_dict")
        if state_dict is None:
            raise KeyError(f"Resume checkpoint does not contain model weights: {latest_path}")
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler_state = resume_state.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

    history, best_bundle = fit_axis_model(
        axis_name=axis_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=artifact_path,
        scheduler=scheduler,
        log_interval=args.log_interval,
        gradient_clip_norm=args.grad_clip_norm,
        resume_state=resume_state,
    )

    checkpoint = torch.load(artifact_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    file_metrics = evaluate_axis_records(
        model=model,
        records=split["val_records"],
        axis_name=axis_name,
        device=device,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        prefix="val_file_level",
    )

    final_bundle = {
        "axis_name": axis_name,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "axis_name": axis_name,
            "pretrained_loaded": model.pretrained_loaded,
            "beats_backbone_config": model.backbone_config,
        },
        "training_config": {
            "sample_rate": args.sample_rate,
            "segment_seconds": args.segment_seconds,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "backbone_learning_rate": args.backbone_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "val_size": args.val_size,
            "seed": args.seed,
            "grad_clip_norm": args.grad_clip_norm,
            "pretrained_checkpoint": str(pretrained_checkpoint_path),
            "data_dirs": data_dirs,
            "class_schema": CLASS_SCHEMA,
        },
        "split_manifest_path": str(split_path.resolve()),
        "metrics_path": file_metrics["metrics_path"],
        "history": history,
        "best_epoch": best_bundle["epoch"],
        "best_composite_score": best_bundle["composite_score"],
    }
    torch.save(final_bundle, artifact_path)
    artifact_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "axis_name": axis_name,
                "artifact_path": str(artifact_path.resolve()),
                "split_manifest_path": str(split_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "pretrained_checkpoint": str(pretrained_checkpoint_path),
                "pretrained_loaded": model.pretrained_loaded,
                "metrics_path": file_metrics["metrics_path"],
                "class_schema": CLASS_SCHEMA,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] Saved final {axis_name} artifact: {artifact_path.resolve()}", flush=True)
    return {"axis_name": axis_name, "artifact_path": str(artifact_path), "metrics": file_metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BEATs per-axis intensity models.")
    parser.add_argument("--axis", choices=[*AXIS_NAMES, "all"], default="all")
    parser.add_argument(
        "--data-dir",
        nargs="+",
        default=[str(path) for path in DEFAULT_DATA_DIRS],
        help="One or more dataset roots. Defaults to single_effect and multi_effect.",
    )
    parser.add_argument("--pretrained-checkpoint", default=str(DEFAULT_PRETRAINED_CHECKPOINT_PATH))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--segment-seconds", type=float, default=DEFAULT_SEGMENT_SECONDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--backbone-learning-rate", type=float, default=DEFAULT_BACKBONE_LEARNING_RATE)
    parser.add_argument("--head-learning-rate", type=float, default=DEFAULT_HEAD_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--grad-clip-norm", type=float, default=DEFAULT_GRAD_CLIP_NORM)
    parser.add_argument("--resume", action="store_true", help="Resume from per-axis latest training checkpoints when present")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device()
    ensure_directory(args.artifact_dir)
    ensure_directory(args.output_dir)
    print(f"[INFO] Device: {device}", flush=True)
    dataset_roots = [Path(data_dir).expanduser().resolve() for data_dir in args.data_dir]
    print("[INFO] Dataset roots:", flush=True)
    for dataset_root in dataset_roots:
        print(f"[INFO]   - {dataset_root}", flush=True)
    print(f"[INFO] Artifact dir: {Path(args.artifact_dir).resolve()}", flush=True)
    print(f"[INFO] Output dir: {Path(args.output_dir).resolve()}", flush=True)

    pretrained_checkpoint_path = Path(args.pretrained_checkpoint).expanduser().resolve()
    if pretrained_checkpoint_path.exists():
        print(f"[INFO] Using pretrained BEATs checkpoint: {pretrained_checkpoint_path}", flush=True)
    else:
        print(f"[WARN] Pretrained checkpoint not found: {pretrained_checkpoint_path}", flush=True)
        print("[WARN] Training will proceed with random backbone initialization.", flush=True)

    records = scan_dataset(args.data_dir)
    axes_to_train = AXIS_NAMES if args.axis == "all" else [args.axis]
    summaries = [train_one_axis(args, axis_name, records, device) for axis_name in axes_to_train]
    summary_path = Path(args.output_dir) / "axis_training_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[INFO] Saved training summary: {summary_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
