#!/usr/bin/env python3
"""
Unified entry point for Dexter runtime and TRM pipelines.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _enable_max_training() -> None:
    os.environ["DEXTER_MAX_TRAINING"] = "1"


def run_trm_pipeline(
    include_success: bool = True,
    train_model: bool = True,
    seq_len: int = 128,
    augmentations: int = 5,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> None:
    from core.bootstrap import run_trm_tool_pipeline

    run_trm_tool_pipeline(
        repo_root=_repo_root(),
        include_success=include_success,
        train_model=train_model,
        seq_len=seq_len,
        augmentations=augmentations,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )


def run_agent(intent: Optional[str], run_pipeline_first: bool, include_success: bool, max_training: bool, pipeline_args: Dict[str, Any]) -> None:
    if max_training:
        _enable_max_training()
        include_success = True
    if run_pipeline_first:
        run_trm_pipeline(include_success=include_success, **pipeline_args)

    from dexter import Dexter

    dexter = Dexter()
    asyncio.run(dexter.startup(initial_intent=intent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dexter entry point")
    subparsers = parser.add_subparsers(dest="command")

    agent_parser = subparsers.add_parser("agent", help="Run Dexter agent (default)")
    agent_parser.add_argument("--intent", help="Optional initial intent")
    agent_parser.add_argument("--pipeline-first", action="store_true", help="Run TRM tool pipeline before starting Dexter.")
    agent_parser.add_argument("--no-success", action="store_true", help="Exclude successful tool calls from TRM dataset.")
    agent_parser.add_argument("--max-training", action="store_true", help="Force max training data capture.")

    pipeline_parser = subparsers.add_parser("trm-pipeline", help="Run TRM tool training pipeline only")
    pipeline_parser.add_argument("--no-success", action="store_true", help="Exclude successful tool calls from TRM dataset.")
    pipeline_parser.add_argument("--no-train", action="store_true", help="Skip training step.")
    pipeline_parser.add_argument("--seq-len", type=int, default=128)
    pipeline_parser.add_argument("--augmentations", type=int, default=5)
    pipeline_parser.add_argument("--epochs", type=int, default=50)
    pipeline_parser.add_argument("--batch-size", type=int, default=16)
    pipeline_parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    command = args.command or "agent"

    pipeline_args = {
        "seq_len": getattr(args, "seq_len", 128),
        "augmentations": getattr(args, "augmentations", 5),
        "epochs": getattr(args, "epochs", 50),
        "batch_size": getattr(args, "batch_size", 16),
        "lr": getattr(args, "lr", 1e-4),
        "train_model": not getattr(args, "no_train", False),
    }

    include_success = not bool(getattr(args, "no_success", False))
    if command == "trm-pipeline":
        run_trm_pipeline(include_success=include_success, **pipeline_args)
        return

    run_agent(
        intent=getattr(args, "intent", None),
        run_pipeline_first=bool(getattr(args, "pipeline_first", False)),
        include_success=include_success,
        max_training=bool(getattr(args, "max_training", False)),
        pipeline_args=pipeline_args,
    )


if __name__ == "__main__":
    main()
