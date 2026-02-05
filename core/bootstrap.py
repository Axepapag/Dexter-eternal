from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from core.trained_trm_loader import get_trm_root


def _load_logs(log_path: Path) -> List[Dict[str, Any]]:
    from scripts.build_trm_tool_dataset import load_logs

    return load_logs(log_path)


def _build_raw_pairs(logs: List[Dict[str, Any]], include_success: bool) -> List[Dict[str, Any]]:
    from scripts.build_trm_tool_dataset import build_pairs, build_selection_pairs

    pairs = build_pairs(logs)
    if include_success:
        pairs.extend(build_selection_pairs(logs))
    return pairs


def run_trm_tool_pipeline(
    repo_root: Path,
    include_success: bool = True,
    train_model: bool = True,
    seq_len: int = 128,
    augmentations: int = 5,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
) -> None:
    trm_root = get_trm_root()
    runtime_dir = trm_root / "datasets" / "runtime"
    tool_corr_dir = trm_root / "datasets" / "tool_correction"
    log_path = runtime_dir / "tool_calls.jsonl"
    raw_pairs_path = tool_corr_dir / "training_pairs_raw.json"
    dataset_dir = tool_corr_dir
    vocab_path = trm_root / "tokenizer" / "tool_policy_vocab.json"
    model_path = trm_root / "models" / "tool_policy" / "trm_tool_policy.pt"

    logs = _load_logs(log_path)
    pairs = _build_raw_pairs(logs, include_success=include_success)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_pairs_path, "w", encoding="utf-8") as fh:
        json.dump(pairs, fh, indent=2)
    print(f"[TRM] Wrote {len(pairs)} training pairs to {raw_pairs_path}")
    if not pairs:
        print("[TRM] No training pairs available. Skipping dataset build.")
        return

    from scripts.build_trm_tool_correction_dataset import ToolCorrectionDatasetBuilder, ToolCorrectionDatasetConfig

    config = ToolCorrectionDatasetConfig(seq_len=seq_len, num_augmentations=augmentations)
    builder = ToolCorrectionDatasetBuilder(config)
    metadata = builder.build_dataset(pairs, dataset_dir)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    builder.vocab.save(vocab_path)
    print(f"[TRM] Dataset ready at {dataset_dir} (examples: {metadata.get('total_puzzles', 0)})")
    print(f"[TRM] Vocabulary saved to {vocab_path}")

    if not train_model:
        return

    from scripts.train_trm_tool_policy import run_training

    run_training(
        data_dir=dataset_dir / "train",
        output_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
