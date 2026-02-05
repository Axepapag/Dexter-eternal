from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch  # type: ignore
except ImportError:
    torch = None


@dataclass
class PlanTemplatePrediction:
    template_id: str
    template_index: int
    confidence: float


class TRMPlanPolicy:
    """Optional TRM-based plan-template classifier."""

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("trm_plan_policy", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.7))
        self.model_path = cfg.get("model_path")
        self.vocab_path = cfg.get("vocab_path")
        self.max_tokens = int(cfg.get("max_tokens", 32))
        self._ready = False
        self._model = None
        self._vocab: Dict[str, int] = {}
        self._template_map: Dict[int, str] = {}
        self._repo_root = Path(__file__).resolve().parent.parent
        if self.enabled:
            self._load_assets()

    def _resolve_path(self, path: Optional[str]) -> Optional[Path]:
        if not path:
            return None
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self._repo_root / candidate

    def _load_assets(self) -> None:
        if torch is None or not self.enabled:
            return
        model_path = self._resolve_path(self.model_path)
        if not model_path or not model_path.exists():
            return

        checkpoint = None
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception:
            return

        vocab = checkpoint.get("vocab") if isinstance(checkpoint, dict) else None
        if vocab:
            self._vocab = vocab
        else:
            vocab_path = self._resolve_path(self.vocab_path)
            if vocab_path and vocab_path.exists():
                try:
                    import json

                    with open(vocab_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    self._vocab = data.get("token2id", data)
                except Exception:
                    self._vocab = {}

        self._template_map = checkpoint.get("template_map", {}) if isinstance(checkpoint, dict) else {}

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            return

        try:
            from core.trm_base import BaseTRM, TRMConfig

            config = checkpoint.get("config", TRMConfig())
            if isinstance(config, dict):
                config = TRMConfig(**config)
            self._model = BaseTRM(config)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
            self._ready = True
        except Exception:
            self._ready = False

    def is_ready(self) -> bool:
        return self.enabled and self._ready and bool(self._vocab) and self._model is not None

    def _encode(self, text: str) -> list[int]:
        tokens = []
        if not text:
            return [0] * self.max_tokens
        words = text.lower().split()
        unk_id = self._vocab.get("<UNK>", 1)
        for word in words[: self.max_tokens]:
            tokens.append(self._vocab.get(word, unk_id))
        if len(tokens) < self.max_tokens:
            tokens.extend([0] * (self.max_tokens - len(tokens)))
        return tokens

    def predict_template(
        self,
        goal: str,
        fallback_index_map: Optional[Dict[int, str]] = None,
    ) -> Optional[PlanTemplatePrediction]:
        if not self.is_ready() or torch is None:
            return None
        tokens = self._encode(goal)
        with torch.no_grad():
            input_tensor = torch.tensor([tokens], dtype=torch.long)
            logits = self._model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            idx = int(torch.argmax(probs, dim=1).item())
            conf = float(probs.max().item())

        template_id = ""
        if self._template_map and idx in self._template_map:
            template_id = self._template_map[idx]
        elif fallback_index_map and idx in fallback_index_map:
            template_id = fallback_index_map[idx]
        if not template_id:
            return None
        return PlanTemplatePrediction(template_id=template_id, template_index=idx, confidence=conf)
