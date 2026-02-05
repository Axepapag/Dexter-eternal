#!/usr/bin/env python3
r"""
trained_trm_loader.py - Loads trained TRMs from the configured TRM root
and integrates them into Dexter's cognitive flow.

These TRMs use the TinyRecursiveModels architecture with:
- Recursive H/L cycles for iterative reasoning
- Carry state persistence across calls
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import yaml

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "core_config.json"


def _load_runtime_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def get_trm_root() -> Path:
    """Resolve the canonical TRM root folder."""
    env_root = os.getenv("DEXTER_TRM_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
    else:
        cfg = _load_runtime_config()
        cfg_root = cfg.get("trm_root") if isinstance(cfg, dict) else None
        root = Path(cfg_root) if cfg_root else BASE_DIR / "dexter_TRMs"

    if not root.is_absolute():
        root = (BASE_DIR / root).resolve()

    # Fallback to legacy location if nothing exists yet
    if not root.exists():
        legacy = BASE_DIR / "models" / "trained_trms"
        if legacy.exists():
            return legacy

    return root


def _load_model_config(model_type: str) -> Dict[str, Any]:
    """Load model config YAML if present under the TRM root."""
    trm_root = get_trm_root()
    cfg_path = trm_root / "configs" / f"model_{model_type}.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class TRMCarryState:
    """Persistent state between TRM calls."""
    z: Optional[torch.Tensor] = None
    steps: int = 0
    
    def is_initialized(self) -> bool:
        return self.z is not None


class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation."""
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        inner_dim = int(hidden_size * expansion)
        self.fc1 = nn.Linear(hidden_size, inner_dim * 2, bias=False)
        self.fc2 = nn.Linear(inner_dim, hidden_size, bias=False)
    
    def forward(self, x):
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * x)


class TRMBlock(nn.Module):
    """Single TRM transformer block."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TinyRecursiveModel(nn.Module):
    """Tiny Recursive Model matching the trained checkpoint format."""
    
    def __init__(
        self,
        vocab_size: int = 16000,
        seq_len: int = 256,
        hidden_size: int = 320,
        num_heads: int = 5,
        num_layers: int = 4,
        H_cycles: int = 2,
        L_cycles: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(seq_len, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TRMBlock(hidden_size, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Carry state initialization
        self.zH_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
        self.zL_init = nn.Parameter(torch.randn(hidden_size) * 0.02)
    
    def forward(
        self,
        inputs: torch.Tensor,
        carry_z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional carry state.
        Returns (logits, new_carry_z)
        """
        B, L = inputs.shape
        device = inputs.device
        
        # Clamp input indices to valid range
        inputs = inputs.clamp(0, self.vocab_size - 1)
        
        # Embeddings
        pos = torch.arange(min(L, self.seq_len), device=device).unsqueeze(0)
        x = self.tok_emb(inputs[:, :self.seq_len]) + self.pos_emb(pos)
        
        # Initialize or use carry
        if carry_z is None:
            z = self.zH_init.unsqueeze(0).unsqueeze(1).expand(B, x.size(1), -1)
        else:
            z = carry_z
        
        # Recursive H/L cycles
        for h in range(self.H_cycles):
            for l in range(self.L_cycles):
                hidden = x + z
                for layer in self.layers:
                    hidden = layer(hidden)
                # Detach between H cycles except last
                if h < self.H_cycles - 1:
                    z = hidden.detach()
                else:
                    z = hidden
        
        # Output
        logits = self.lm_head(self.norm(z))
        return logits, z.detach()


class DexterTRMConfig:
    """Configuration for Dexter's trained TRMs."""
    
    def __init__(self, config_dict: Optional[dict] = None, **kwargs):
        defaults = {
            "vocab_size": 16000,
            "seq_len": 256,
            "hidden_size": 320,
            "num_heads": 5,
            "num_layers": 4,
            "H_cycles": 2,
            "L_cycles": 3,
            "dropout": 0.1,
        }
        config = {**defaults, **(config_dict or {}), **kwargs}
        for k, v in config.items():
            setattr(self, k, v)


class TrainedDexterTRM:
    """Base class for trained Dexter TRMs."""
    
    def __init__(
        self,
        model_type: str,
        device: str = "cpu",
        use_carry: bool = True,
    ):
        self.model_type = model_type
        self.device = device
        self.use_carry = use_carry
        
        # Find model path
        self.model_path = self._find_model_path()
        self.tokenizer_path = self._find_tokenizer_path()
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Load model
        self.model = None
        self.config = None
        self._load_model()
        
        # Carry state
        self.carry = TRMCarryState()
        
        # Stats
        self._call_count = 0
        self._verbose = os.getenv("DEXTER_TRM_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")
    
    def _find_model_path(self) -> Optional[Path]:
        """Find the best model checkpoint."""
        trm_root = get_trm_root()
        model_dir = trm_root / "models" / self.model_type
        candidates = [
            model_dir / "best.pt",
            model_dir / "final.pt",
            model_dir / "last.pt",
        ]
        # Legacy filename fallbacks (if present under TRM root)
        legacy_dir = trm_root / "models"
        candidates.extend([
            legacy_dir / f"{self.model_type}_best.pt",
            legacy_dir / f"{self.model_type}_final.pt",
            legacy_dir / f"{self.model_type}_last.pt",
        ])
        for p in candidates:
            if p.exists():
                return p
        return None
    
    def _find_tokenizer_path(self) -> Optional[Path]:
        """Find tokenizer."""
        trm_root = get_trm_root()
        candidates = [
            trm_root / "tokenizer" / "tokenizer.json",
            trm_root / "models" / "tokenizer.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None
    
    def _load_tokenizer(self) -> Dict[str, int]:
        """Load tokenizer vocabulary."""
        if not self.tokenizer_path or not self.tokenizer_path.exists():
            return {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<SEP>": 3, "<UNK>": 4}
        
        with open(self.tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        vocab = {}
        if "model" in data and "vocab" in data["model"]:
            vocab = data["model"]["vocab"]
        elif "added_tokens" in data:
            for token in data["added_tokens"]:
                vocab[token["content"]] = token["id"]
        
        print(f"[{self.model_type.upper()} TRM] Loaded tokenizer: {len(vocab)} tokens")
        return vocab
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path or not self.model_path.exists():
            print(f"[{self.model_type.upper()} TRM] No model found")
            return
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Get config from checkpoint or YAML fallback
            trm_config = checkpoint.get("trm_config", {})
            if not trm_config:
                trm_config = _load_model_config(self.model_type)
            self.config = DexterTRMConfig(trm_config)
            
            # Create model
            self.model = TinyRecursiveModel(
                vocab_size=self.config.vocab_size,
                seq_len=self.config.seq_len,
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                H_cycles=self.config.H_cycles,
                L_cycles=self.config.L_cycles,
                dropout=self.config.dropout,
            )
            
            # Load weights
            state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
            if isinstance(state_dict, dict):
                self.model.load_state_dict(state_dict, strict=False)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"[{self.model_type.upper()} TRM] Loaded from {self.model_path}")
            print(f"[{self.model_type.upper()} TRM] Config: hidden={self.config.hidden_size}, "
                  f"H/L={self.config.H_cycles}/{self.config.L_cycles}, heads={self.config.num_heads}")
            
        except Exception as e:
            print(f"[{self.model_type.upper()} TRM] Failed to load: {e}")
            self.model = None
    
    def is_ready(self) -> bool:
        return self.model is not None
    
    def tokenize(self, text: str, max_len: Optional[int] = None) -> torch.Tensor:
        """Tokenize text."""
        max_len = max_len or (self.config.seq_len if self.config else 256)
        
        pad_id = self.tokenizer.get("<PAD>", 0)
        bos_id = self.tokenizer.get("<BOS>", 1)
        eos_id = self.tokenizer.get("<EOS>", 2)
        unk_id = self.tokenizer.get("<UNK>", 4)
        
        tokens = [bos_id]
        for word in str(text).lower().split():
            word = word.strip(".,!?;:()[]{}'\"")[:20]
            if word:
                token_id = self.tokenizer.get(word, unk_id)
                tokens.append(token_id)
                if len(tokens) >= max_len - 1:
                    break
        tokens.append(eos_id)
        
        while len(tokens) < max_len:
            tokens.append(pad_id)
        
        return torch.tensor([tokens[:max_len]], dtype=torch.long, device=self.device)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs."""
        id_to_token = {v: k for k, v in self.tokenizer.items()}
        tokens = []
        for idx in token_ids:
            if idx == self.tokenizer.get("<EOS>", 2):
                break
            if idx not in [self.tokenizer.get("<PAD>", 0), self.tokenizer.get("<BOS>", 1)]:
                tokens.append(id_to_token.get(idx, "<UNK>"))
        return " ".join(tokens)
    
    def reset_carry(self):
        """Reset carry state."""
        self.carry = TRMCarryState()
    
    @torch.no_grad()
    def forward(self, input_text: str) -> Dict[str, Any]:
        """Run inference."""
        if not self.is_ready():
            return {"error": "TRM not ready", "output": None}
        
        input_ids = self.tokenize(input_text)
        
        carry_z = self.carry.z if self.use_carry and self.carry.is_initialized() else None
        
        logits, new_z = self.model(input_ids, carry_z)
        
        if self.use_carry:
            self.carry.z = new_z
            self.carry.steps += 1
        
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
        
        probs = F.softmax(logits.squeeze(0), dim=-1)
        confidences = [probs[i, pid].item() for i, pid in enumerate(pred_ids) if i < len(pred_ids)]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        self._call_count += 1
        
        return {
            "output_text": self.decode(pred_ids),
            "output_ids": pred_ids,
            "confidence": avg_conf,
            "carry_active": self.use_carry,
            "steps": self.carry.steps if self.use_carry else 0,
        }


class ToolTRM(TrainedDexterTRM):
    """Tool selection TRM."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("tool", device, use_carry=True)
    
    def select_tool(
        self,
        task: str,
        context: Optional[Dict] = None,
        last_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select the best tool for a task."""
        prompt = f"<TASK> {task}"
        if context:
            prompt += f" <CTX> {json.dumps(context)}"
        if last_error:
            prompt += f" <ERR> {last_error}"
        
        result = self.forward(prompt)
        
        output = result.get("output_text", "")
        tool_name = None
        
        if "<TOOL:" in output:
            start = output.find("<TOOL:") + 6
            end = output.find(">", start)
            if end > start:
                tool_name = output[start:end]
        
        return {
            "tool": tool_name,
            "confidence": result.get("confidence", 0.0),
            "raw_output": output,
            "carry_steps": result.get("steps", 0),
        }
    
    def ingest_result(
        self,
        task: str,
        tool_used: str,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Ingest execution result into carry state.
        This lets the TRM learn from what just happened.
        
        The TRM processes a "result token sequence" and updates its
        carry state, building up context over time.
        """
        # Build result prompt - TRM will process this and update carry
        status = "SUCCESS" if success else "FAIL"
        prompt = f"<RESULT> {task} -> {tool_used} [{status}]"
        if error:
            prompt += f" <ERR:{error[:30]}>"
        if metadata:
            if metadata.get("skill_forged"):
                prompt += f" <FORGED:{metadata.get('skill_forged_name', 'new')}>"
            if metadata.get("retries", 0) > 0:
                prompt += f" <RETRY:{metadata.get('retries')}>"
        
        # Forward pass updates carry state with this experience
        result = self.forward(prompt)
        
        return {
            "ingested": True,
            "carry_steps": result.get("steps", 0),
            "confidence": result.get("confidence", 0.0),
        }


class MemoryTRM(TrainedDexterTRM):
    """Memory retrieval TRM."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("memory", device, use_carry=True)
    
    def query(self, query: str, memory_type: str = "all") -> Dict[str, Any]:
        """Query memory."""
        prompt = f"<MEM:{memory_type.upper()}> {query}"
        
        result = self.forward(prompt)
        
        output = result.get("output_text", "")
        memories = []
        
        if "<SEP>" in output:
            memories = [m.strip() for m in output.split("<SEP>") if m.strip()]
        elif output.strip():
            memories = [output.strip()]
        
        return {
            "query": query,
            "memories": memories,
            "memory_type": memory_type,
            "confidence": result.get("confidence", 0.0),
            "carry_steps": result.get("steps", 0),
        }
    
    def ingest_message(self, sender: str, message: str, msg_type: str, metadata: Dict = None):
        """
        Ingest a message from the communication bus.
        
        Memory TRM learns from all conversations:
        - Stores episodic memories (what happened)
        - Extracts facts and entities
        - Builds knowledge graph connections
        """
        if not self.is_ready():
            return
        
        # Format for memory encoding
        prompt = f"<ENCODE> [{sender}|{msg_type}] {message}"
        if metadata:
            prompt += f" <META> {json.dumps(metadata)}"
        
        try:
            # Forward pass updates carry state - TRM "remembers" this
            result = self.forward(prompt)
            if self._verbose:
                print(f"[MemoryTRM] Ingested: {sender}/{msg_type} (steps={result.get('steps', 0)})", flush=True)
        except Exception as e:
            print(f"[MemoryTRM] Ingest error: {e}", flush=True)


class ReasoningTRM(TrainedDexterTRM):
    """Reasoning TRM."""
    
    def __init__(self, device: str = "cpu"):
        super().__init__("reasoning", device, use_carry=True)
    
    def reason(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate reasoning steps."""
        full_prompt = f"<REASON> {prompt}"
        if context:
            full_prompt += f" <CTX> {json.dumps(context)}"
        
        result = self.forward(full_prompt)
        
        output = result.get("output_text", "")
        steps = []
        
        if "<STEP>" in output:
            parts = output.split("<STEP>")
            steps = [p.strip() for p in parts if p.strip()]
        elif output.strip():
            steps = [output.strip()]
        
        return {
            "prompt": prompt,
            "reasoning_steps": steps,
            "conclusion": steps[-1] if steps else None,
            "confidence": result.get("confidence", 0.0),
            "carry_steps": result.get("steps", 0),
        }
    
    def ingest_message(self, sender: str, message: str, msg_type: str, metadata: Dict = None):
        """
        Ingest a message from the communication bus.
        
        Reasoning TRM learns from request→result patterns:
        - Observes what Dexter asked for
        - Sees how Forge handled it
        - Learns execution patterns and strategies
        """
        if not self.is_ready():
            return
        
        # Format for pattern learning
        prompt = f"<OBSERVE> [{sender}|{msg_type}] {message[:500]}"
        
        # Results are especially valuable for learning patterns
        if msg_type == "result" and metadata:
            success = metadata.get("success", metadata.get("ok", False))
            prompt += f" <OUTCOME:{'SUCCESS' if success else 'FAIL'}>"
        
        try:
            # Forward pass updates carry state - TRM learns patterns
            result = self.forward(prompt)
            if self._verbose:
                print(f"[ReasoningTRM] Observed: {sender}/{msg_type} (steps={result.get('steps', 0)})", flush=True)
        except Exception as e:
            print(f"[ReasoningTRM] Ingest error: {e}", flush=True)


# Global instances
_tool_trm: Optional[ToolTRM] = None
_memory_trm: Optional[MemoryTRM] = None
_reasoning_trm: Optional[ReasoningTRM] = None


def get_tool_trm(device: str = "cpu") -> ToolTRM:
    global _tool_trm
    if _tool_trm is None:
        _tool_trm = ToolTRM(device)
    return _tool_trm


def get_memory_trm(device: str = "cpu") -> MemoryTRM:
    global _memory_trm
    if _memory_trm is None:
        _memory_trm = MemoryTRM(device)
    return _memory_trm


def get_reasoning_trm(device: str = "cpu") -> ReasoningTRM:
    global _reasoning_trm
    if _reasoning_trm is None:
        _reasoning_trm = ReasoningTRM(device)
    return _reasoning_trm


def load_all_trms(device: str = "cpu") -> Dict[str, TrainedDexterTRM]:
    """Load all available TRMs."""
    trms = {}
    
    tool = get_tool_trm(device)
    if tool.is_ready():
        trms["tool"] = tool
    
    memory = get_memory_trm(device)
    if memory.is_ready():
        trms["memory"] = memory
    
    reasoning = get_reasoning_trm(device)
    if reasoning.is_ready():
        trms["reasoning"] = reasoning
    
    print(f"[TRM Loader] Loaded {len(trms)} TRMs: {list(trms.keys())}")
    return trms


if __name__ == "__main__":
    print("Testing TRM Loader...")
    trms = load_all_trms()
    
    if "tool" in trms:
        result = trms["tool"].select_tool("Check system CPU usage")
        print(f"Tool TRM: {result}")
    
    if "memory" in trms:
        result = trms["memory"].query("What projects am I working on?")
        print(f"Memory TRM: {result}")
