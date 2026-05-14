#!/usr/bin/env python3
"""Train the BEATs multi-axis intensity model."""

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
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_BACKBONE_LEARNING_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
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
    DEFAULT_SPLIT_PATH,
    DEFAULT_VAL_SIZE,
    DEFAULT_WEIGHT_DECAY,
)
from dataset import create_train_val_split, save_split_manifest, scan_dataset  # noqa: E402
from shared.audio import ensure_directory  # noqa: E402
from shared.torch_training import choose_device, seed_everything  # noqa: E402
from utils import build_model, class_weights_from_labels, create_dataloaders, create_optimizer, evaluate_records, fit_multiaxis_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the BEATs multi-axis intensity model.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--pretrained-checkpoint", default=str(DEFAULT_PRETRAINED_CHECKPOINT_PATH))
    parser.add_argument("--artifact-path", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--split-manifest-path", default=str(DEFAULT_SPLIT_PATH))
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
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device()
    output_dir = ensure_directory(args.output_dir)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Dataset root: {Path(args.data_dir).resolve()}", flush=True)

    pretrained_checkpoint_path = Path(args.pretrained_checkpoint).expanduser().resolve()
    pretrained_exists = pretrained_checkpoint_path.exists()
    if pretrained_exists:
        print(f"[INFO] Using pretrained BEATs checkpoint: {pretrained_checkpoint_path}", flush=True)
    else:
        print(f"[WARN] Pretrained checkpoint not found: {pretrained_checkpoint_path}", flush=True)
        print("[WARN] Training will proceed with random backbone initialization.", flush=True)

    records = scan_dataset(args.data_dir)
    split = create_train_val_split(records, val_size=args.val_size, seed=args.seed)
    save_split_manifest(split, args.split_manifest_path)
    print(f"[INFO] Records: total={len(records)} train={len(split['train_records'])} val={len(split['val_records'])}", flush=True)

    train_loader, val_loader = create_dataloaders(
        split["train_records"], split["val_records"],
        sample_rate=args.sample_rate, segment_seconds=args.segment_seconds, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    model = build_model(pretrained_checkpoint_path if pretrained_exists else None).to(device)
    print(f"[INFO] Backbone pretrained loaded: {model.pretrained_loaded}", flush=True)

    axis_train_levels = {
        axis_name: [getattr(record, f"{axis_name}_level") for record in split["train_records"]]
        for axis_name in AXIS_NAMES
    }
    drive_criterion = torch.nn.CrossEntropyLoss(weight=class_weights_from_labels(axis_train_levels["drive"]).to(device), label_smoothing=0.02)
    phase_criterion = torch.nn.CrossEntropyLoss(weight=class_weights_from_labels(axis_train_levels["phase"]).to(device), label_smoothing=0.02)
    space_criterion = torch.nn.CrossEntropyLoss(weight=class_weights_from_labels(axis_train_levels["space"]).to(device), label_smoothing=0.02)

    optimizer = create_optimizer(model, backbone_learning_rate=args.backbone_learning_rate, head_learning_rate=args.head_learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    history, _ = fit_multiaxis_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        drive_criterion=drive_criterion,
        phase_criterion=phase_criterion,
        space_criterion=space_criterion,
        device=device,
        num_epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.artifact_path,
        scheduler=scheduler,
        log_interval=args.log_interval,
        gradient_clip_norm=args.grad_clip_norm,
    )

    checkpoint = torch.load(args.artifact_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    file_metrics = evaluate_records(
        model=model,
        records=split["val_records"],
        device=device,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        prefix="val_file_level",
    )

    final_bundle = {
        "axis_names": AXIS_NAMES,
        "model_state_dict": model.state_dict(),
        "model_config": {
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
        },
        "metrics_path": file_metrics["metrics_path"],
        "history": history,
    }
    torch.save(final_bundle, args.artifact_path)
    Path(args.artifact_path).with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "artifact_path": str(Path(args.artifact_path).resolve()),
                "split_manifest_path": str(Path(args.split_manifest_path).resolve()),
                "output_dir": str(Path(args.output_dir).resolve()),
                "pretrained_checkpoint": str(pretrained_checkpoint_path),
                "pretrained_loaded": model.pretrained_loaded,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] Saved artifact: {Path(args.artifact_path).resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

