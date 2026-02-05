from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except ImportError:
    torch = None
    nn = None
    F = None


@dataclass
class TRMToolCall:
    name: str
    arguments: Dict[str, Any]
    confidence: float


@dataclass
class ToolTRMConfig:
    vocab_size: int = 512
    seq_len: int = 128
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 2
    H_cycles: int = 3
    L_cycles: int = 4
    dropout: float = 0.1


if nn is None:
    class ToolTRMModel:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Torch is required to use ToolTRMModel.")

        def eval(self) -> "ToolTRMModel":
            return self

else:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight


    class SwiGLU(nn.Module):
        def __init__(self, hidden_size: int, expansion: float = 4.0):
            super().__init__()
            inner_dim = int(hidden_size * expansion)
            self.fc1 = nn.Linear(hidden_size, inner_dim * 2, bias=False)
            self.fc2 = nn.Linear(inner_dim, hidden_size, bias=False)

        def forward(self, x):
            x, gate = self.fc1(x).chunk(2, dim=-1)
            return self.fc2(F.silu(gate) * x)


    class TRMBlock(nn.Module):
        def __init__(self, config: ToolTRMConfig):
            super().__init__()
            self.norm1 = RMSNorm(config.hidden_size)
            self.attn = nn.MultiheadAttention(
                config.hidden_size,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.norm2 = RMSNorm(config.hidden_size)
            self.ffn = SwiGLU(config.hidden_size)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x


    class ToolTRMModel(nn.Module):
        """Tiny Recursive Model for tool-call token prediction."""

        def __init__(self, config: ToolTRMConfig):
            super().__init__()
            self.config = config
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.pos_embed = nn.Embedding(config.seq_len, config.hidden_size)
            self.layers = nn.ModuleList([TRMBlock(config) for _ in range(config.num_layers)])
            self.norm = RMSNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.z_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            batch, length = inputs.shape
            device = inputs.device
            pos = torch.arange(length, device=device).unsqueeze(0)
            x = self.embed(inputs) + self.pos_embed(pos)
            z = self.z_init.unsqueeze(0).unsqueeze(1).expand(batch, length, -1)

            for h in range(self.config.H_cycles):
                for _ in range(self.config.L_cycles):
                    hidden = x + z
                    for layer in self.layers:
                        hidden = layer(hidden)
                    z = hidden if h == self.config.H_cycles - 1 else hidden.detach()

            return self.lm_head(self.norm(z))


class TRMToolPolicy:
    """
    Optional TRM-based tool-call predictor. Uses a TinyRecursiveModels checkpoint
    trained on tool correction datasets.
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("trm_tool_policy", {}) or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.use_for_execution = bool(cfg.get("use_for_execution", False))
        self.shadow_mode = bool(cfg.get("shadow_mode", True))
        self.confidence_threshold = float(cfg.get("confidence_threshold", 0.85))
        self.auto_mode = bool(cfg.get("auto_mode", True))
        self.execute_threshold = float(cfg.get("execute_threshold", 0.9))
        self.shadow_threshold = float(cfg.get("shadow_threshold", 0.6))
        self.mode = str(cfg.get("mode", "hybrid")).lower()
        self.model_path = cfg.get("model_path")
        self.vocab_path = cfg.get("vocab_path")
        self.max_tokens = int(cfg.get("max_tokens", 128))
        self._ready = False
        self._model = None
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._repo_root = Path(__file__).resolve().parent.parent

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.sep_id = 3
        self.err_id = 4
        self.ok_id = 5
        self.tool_id = 6
        self.arg_id = 7
        self.val_id = 8
        self.unk_id = 9
        self.err_other_id = 18

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

        vocab_path = self._resolve_path(self.vocab_path)
        if vocab_path and vocab_path.exists():
            try:
                import json

                with open(vocab_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._vocab = data.get("token2id", data)
            except Exception:
                self._vocab = {}

        if self._vocab:
            self._id_to_token = {idx: token for token, idx in self._vocab.items()}
            self.pad_id = self._vocab.get("<PAD>", self.pad_id)
            self.bos_id = self._vocab.get("<BOS>", self.bos_id)
            self.eos_id = self._vocab.get("<EOS>", self.eos_id)
            self.sep_id = self._vocab.get("<SEP>", self.sep_id)
            self.err_id = self._vocab.get("<ERR>", self.err_id)
            self.ok_id = self._vocab.get("<OK>", self.ok_id)
            self.tool_id = self._vocab.get("<TOOL>", self.tool_id)
            self.arg_id = self._vocab.get("<ARG>", self.arg_id)
            self.val_id = self._vocab.get("<VAL>", self.val_id)
            self.unk_id = self._vocab.get("<UNK>", self.unk_id)
            self.err_other_id = self._vocab.get("<ERR:other>", self.err_other_id)

        model_path = self._resolve_path(self.model_path)
        if not model_path or not model_path.exists():
            return

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception:
            return

        try:
            config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            if isinstance(config, dict):
                trm_config = ToolTRMConfig(**config)
            else:
                trm_config = ToolTRMConfig()
            model = ToolTRMModel(trm_config)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            self._model = model
            self._model.eval()
            self._ready = True
        except Exception:
            self._ready = False

    def is_ready(self) -> bool:
        return self.enabled and self._ready and self._model is not None and bool(self._vocab)

    def _token_id(self, token: str) -> int:
        return self._vocab.get(token, self.unk_id)

    def _tokenize_tool_name(self, tool_name: str) -> List[int]:
        tokens = [self.tool_id]
        if tool_name and "." in tool_name:
            prefix, func = tool_name.split(".", 1)
            tokens.append(self._token_id(f"<TOOL:{prefix}>"))
            tokens.append(self._token_id(f"<FUNC:{func}>"))
        elif tool_name:
            tokens.append(self._token_id(f"<TOOL:{tool_name}>"))
        else:
            tokens.append(self.unk_id)
        return tokens

    def _tokenize_error(self, error_type: str, error_msg: str) -> List[int]:
        tokens = [self.err_id]
        err_token = f"<ERR:{error_type}>"
        tokens.append(self._token_id(err_token) if err_token in self._vocab else self.err_other_id)
        for word in error_msg.lower().split()[:10]:
            if len(word) > 3:
                tokens.append(self._token_id(f"<W:{word[:16]}>"))
        return tokens

    def _tokenize_context(self, text: str, limit: int = 5) -> List[int]:
        tokens = []
        for word in str(text).lower().split()[:limit]:
            if len(word) > 3:
                tokens.append(self._token_id(f"<W:{word[:12]}>"))
        return tokens

    def _pad_sequence(self, tokens: List[int], max_len: int) -> List[int]:
        if len(tokens) > max_len:
            return tokens[:max_len]
        return tokens + [self.pad_id] * (max_len - len(tokens))

    def _tool_failed(self, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if result.get("ok") is False or result.get("success") is False:
            return True
        payload = result.get("result")
        if payload is False:
            return True
        if isinstance(payload, dict) and payload.get("success") is False:
            return True
        return False

    def _error_type_from_result(self, result: Dict[str, Any]) -> str:
        err = (result or {}).get("error", "") or ""
        err_lower = str(err).lower()
        if "timeout" in err_lower:
            return "timeout"
        if "missing" in err_lower and "argument" in err_lower:
            return "missing_argument"
        if "not found" in err_lower and "tool" in err_lower:
            return "tool_not_found"
        if "file not found" in err_lower:
            return "file_not_found"
        if "permission" in err_lower:
            return "permission_error"
        if "json" in err_lower and "parse" in err_lower:
            return "json_parse_error"
        if "connection" in err_lower:
            return "connection_error"
        return "other"

    def _build_input_tokens(
        self,
        task: str,
        skill_id: str,
        error_type: str,
        error_msg: str,
        failed_tool: str,
    ) -> List[int]:
        tokens = [self.bos_id]
        tokens.extend(self._tokenize_error(error_type, error_msg))
        tokens.extend(self._tokenize_tool_name(failed_tool))
        tokens.append(self.sep_id)
        tokens.extend(self._tokenize_context(task, limit=5))
        tokens.extend(self._tokenize_context(skill_id, limit=3))
        seq_len = min(self.max_tokens, getattr(self._model.config, "seq_len", self.max_tokens))
        return self._pad_sequence(tokens, seq_len)

    def _decode_tool_call(
        self,
        token_ids: List[int],
        skill_id: str,
    ) -> Tuple[str, Dict[str, Any], List[int]]:
        tokens = [self._id_to_token.get(tok, "<UNK>") for tok in token_ids]
        start_idx = 0
        if "<OK>" in tokens:
            start_idx = tokens.index("<OK>") + 1
        prefix = ""
        func = ""
        args: Dict[str, Any] = {}
        used_indices: List[int] = []
        expect_arg = False
        expect_val = False
        current_key = ""

        for idx, token in enumerate(tokens[start_idx:], start=start_idx):
            if token in ("<EOS>", "<PAD>"):
                break
            if token.startswith("<TOOL:"):
                prefix = token[6:-1]
                used_indices.append(idx)
                continue
            if token.startswith("<FUNC:"):
                func = token[6:-1]
                used_indices.append(idx)
                continue
            if token == "<ARG>":
                expect_arg = True
                expect_val = False
                current_key = ""
                continue
            if token == "<VAL>":
                expect_val = True
                continue
            if token.startswith("<W:"):
                value = token[3:-1]
                if expect_arg and not current_key:
                    current_key = value
                    continue
                if expect_val and current_key:
                    args[current_key] = value
                    expect_arg = False
                    expect_val = False
                    current_key = ""

        tool_name = ""
        if prefix and func:
            tool_name = f"{prefix}.{func}"
        elif func and skill_id:
            tool_name = f"{skill_id}.{func}"
        elif prefix:
            tool_name = prefix
        return tool_name, args, used_indices

    def _estimate_confidence(self, logits: torch.Tensor, token_ids: List[int], indices: List[int]) -> float:
        if not indices:
            return 0.0
        probs = torch.softmax(logits, dim=-1)
        confs = []
        for idx in indices:
            token_id = token_ids[idx]
            confs.append(float(probs[0, idx, token_id].item()))
        return sum(confs) / len(confs)

    def predict_tool_call(
        self,
        task: str,
        skill_id: str,
        context: Dict[str, Any],
    ) -> Optional[TRMToolCall]:
        if not self.is_ready() or torch is None:
            return None

        last_result = context.get("last_action_result") or {}
        last_call = context.get("last_tool_call") or {}
        failed_tool = ""
        error_msg = ""
        error_type = "other"
        if self._tool_failed(last_result):
            error_type = self._error_type_from_result(last_result)
            error_msg = str(last_result.get("error") or "")
            failed_tool = str(last_call.get("name") or "")

        if self.mode == "correction" and not failed_tool and not error_msg:
            return None

        input_ids = self._build_input_tokens(
            task=task,
            skill_id=skill_id,
            error_type=error_type,
            error_msg=error_msg,
            failed_tool=failed_tool,
        )

        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            logits = self._model(input_tensor)
            pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()

        tool_name, arguments, used_indices = self._decode_tool_call(pred_ids, skill_id)
        if not tool_name:
            return None

        confidence = self._estimate_confidence(logits, pred_ids, used_indices)
        return TRMToolCall(name=tool_name, arguments=arguments, confidence=confidence)

    def decide_mode(self, confidence: float) -> str:
        """
        Decide whether to execute, shadow, or ignore a TRM prediction.
        Returns: "execute", "shadow", or "ignore".
        """
        if not self.enabled:
            return "ignore"
        if self.auto_mode:
            if confidence >= self.execute_threshold:
                return "execute"
            if confidence >= self.shadow_threshold:
                return "shadow"
            return "ignore"
        if self.use_for_execution and confidence >= self.confidence_threshold:
            return "execute"
        if self.shadow_mode:
            return "shadow"
        return "ignore"
