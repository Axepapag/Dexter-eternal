#!/usr/bin/env python3
"""
trm_online_trainer.py - Online Training System for Tool TRM

This implements a dual-weight training system:
1. ACTIVE weights - Used for inference, maintains carry state
2. SHADOW weights - Being fine-tuned in background on new data

Periodically, shadow weights are promoted to active:
- Active carry state is discarded (fresh start)
- Shadow weights become active
- Fresh copy of new active is made for next training cycle

This allows the TRM to learn in real-time while maintaining stable inference.
"""

import asyncio
import copy
import json
import os
import threading
import time
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset

from core.trained_trm_loader import (
    TinyRecursiveModel, 
    TrainedDexterTRM, 
    get_tool_trm,
    get_memory_trm,
    get_reasoning_trm,
    get_trm_root,
    TRMCarryState,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = get_trm_root() / "models" / "online"


@dataclass
class TrainingExample:
    """A single training example for online TRM training."""
    input_text: str
    target_text: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    # Legacy / tool-specific fields for compatibility
    intent: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class TrainerStats:
    """Statistics for the online trainer."""
    examples_collected: int = 0
    batches_trained: int = 0
    weight_swaps: int = 0
    last_swap_time: float = 0
    last_train_time: float = 0
    shadow_loss: float = 0.0
    active_accuracy: float = 0.0


class OnlineTRMTrainer:
    """
    Online training system with dual weights.
    
    Architecture:
    - active_model: Used for inference, has carry state
    - shadow_model: Being trained in background
    
    Training loop:
    1. Collect examples from tool executions
    2. When batch_size reached, fine-tune shadow_model
    3. After N batches or time interval, swap weights
    4. Reset carry state after swap (fresh start with new knowledge)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_type: str = "tool",
        base_trm: Optional[TrainedDexterTRM] = None,
    ):
        self.config = config
        self.trainer_cfg = config.get("online_training", {}) or {}
        self.model_type = model_type
        
        # Training parameters
        self.enabled = self.trainer_cfg.get("enabled", True)
        self.batch_size = self.trainer_cfg.get("batch_size", 16)
        self.min_examples_for_train = self.trainer_cfg.get("min_examples", 8)
        self.train_every_n_examples = self.trainer_cfg.get("train_every_n", 16)
        self.swap_after_n_batches = self.trainer_cfg.get("swap_after_n_batches", 10)
        self.swap_every_seconds = self.trainer_cfg.get("swap_every_seconds", 3600)  # 1 hour
        self.learning_rate = self.trainer_cfg.get("lr", 1e-5)
        self.max_buffer_size = self.trainer_cfg.get("max_buffer_size", 1000)
        self.save_checkpoints = self.trainer_cfg.get("save_checkpoints", True)
        self.swap_policy = str(self.trainer_cfg.get("swap_policy", "auto")).lower()
        self._swap_policy_warned = False
        
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Models - will be initialized
        self.active_model: Optional[TinyRecursiveModel] = None
        self.shadow_model: Optional[TinyRecursiveModel] = None
        self.model_config: Optional[Dict] = None
        self.tokenizer: Dict[str, int] = {}
        
        # Carry state for active model
        self.active_carry = TRMCarryState()
        
        # Training buffer
        self.example_buffer: deque = deque(maxlen=self.max_buffer_size)
        self.batches_since_swap = 0
        self.last_swap_time = time.time()
        
        # Stats
        self.stats = TrainerStats()
        
        # Background training
        self._train_lock = threading.Lock()
        self._running = False
        self._train_thread: Optional[threading.Thread] = None
        
        # JSONL log for permanent dataset capture
        self.dataset_log_path = get_trm_root() / "datasets" / f"online_{self.model_type}.jsonl"
        
        # Initialize from existing TRM if provided
        if base_trm and base_trm.is_ready():
            self._init_from_trm(base_trm)
    
    def _init_from_trm(self, trm: TrainedDexterTRM):
        """Initialize dual weights from existing TRM."""
        if not trm.model:
            return
            
        # Store config
        self.model_config = {
            "vocab_size": trm.config.vocab_size,
            "seq_len": trm.config.seq_len,
            "hidden_size": trm.config.hidden_size,
            "num_heads": trm.config.num_heads,
            "num_layers": trm.config.num_layers,
            "H_cycles": trm.config.H_cycles,
            "L_cycles": trm.config.L_cycles,
        }
        self.tokenizer = trm.tokenizer
        
        # Clone to active and shadow
        self.active_model = copy.deepcopy(trm.model).to(self.device)
        self.shadow_model = copy.deepcopy(trm.model).to(self.device)
        
        # Set modes
        self.active_model.eval()
        self.shadow_model.train()
        
        print(f"[Online Trainer] Initialized dual weights (hidden={self.model_config['hidden_size']})")
    
    def start(self):
        """Start background training loop."""
        if not self.enabled or self._running:
            return
            
        self._running = True
        self._train_thread = threading.Thread(target=self._training_loop, daemon=True)
        self._train_thread.start()
        print(f"[Online Trainer:{self.model_type}] Background training started")
    
    def stop(self):
        """Stop training loop."""
        self._running = False
        if self._train_thread:
            self._train_thread.join(timeout=5)
        print(f"[Online Trainer:{self.model_type}] Stopped. Stats: {self.stats}")
    
    def record_execution(
        self,
        intent: str,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a tool execution for training.
        Called after every tool call.
        """
        if not self.enabled:
            return
            
        example = TrainingExample(
            input_text=intent,
            target_text=tool_name,
            success=success,
            context=context or {},
            intent=intent,
            tool_name=tool_name,
            arguments=arguments,
        )
        
        # Add to buffer
        self.example_buffer.append(example)
        self.stats.examples_collected += 1
        
        # Log to permanent dataset (never lost)
        self._log_to_dataset(example)
    
    def observe_teacher_decision(
        self,
        intent: str,
        teacher_tool: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a Teacher's decision for TRM learning.
        
        This is the PRIMARY way the TRM learns - by observing the Teacher.
        The TRM is an infant observer. Every Teacher decision is:
        1. Logged to the training dataset
        2. Compared against TRM's prediction (for accuracy tracking)
        3. Used for online weight updates
        
        Args:
            intent: The raw input from Dexter
            teacher_tool: The tool the Teacher chose
            metadata: Additional info (TRM's prediction, confidence, etc.)
        """
        if not self.enabled:
            return
        
        metadata = metadata or {}
        trm_prediction = metadata.get("trm_prediction")
        trm_was_correct = metadata.get("trm_was_correct", False)
        
        # Create training example from Teacher decision
        # Success is True because Teacher's decisions are authoritative
        example = TrainingExample(
            input_text=intent,
            target_text=teacher_tool,
            success=True,  # Teacher decisions are ground truth
            context={
                "source": "teacher_observation",
                "trm_predicted": trm_prediction,
                "trm_was_correct": trm_was_correct,
                "trm_confidence": metadata.get("trm_confidence", 0),
            },
            intent=intent,
            tool_name=teacher_tool,
            arguments={},  # Args are structured by Teacher, not our concern here
        )
        
        # Add to buffer for training
        self.example_buffer.append(example)
        self.stats.examples_collected += 1
        
        # Log to permanent dataset
        self._log_to_dataset(example)
        
        # Track observation accuracy
        self.stats.__dict__.setdefault("teacher_observations", 0)
        self.stats.__dict__["teacher_observations"] += 1
        if trm_was_correct:
            self.stats.__dict__.setdefault("correct_observations", 0)
            self.stats.__dict__["correct_observations"] += 1
        
        print(f"[Online Trainer:{self.model_type}] Observed: '{intent[:40]}...' -> {teacher_tool}", flush=True)

    def record_text_example(
        self,
        input_text: str,
        target_text: Optional[str] = None,
        success: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a generic text example for non-tool TRMs (memory/reasoning).
        If target_text is None, uses input_text (auto-encoding).
        """
        if not self.enabled:
            return
        if not input_text:
            return
        example = TrainingExample(
            input_text=input_text,
            target_text=target_text or input_text,
            success=success,
            context=context or {},
        )
        self.example_buffer.append(example)
        self.stats.examples_collected += 1
        self._log_to_dataset(example)
    
    def _log_to_dataset(self, example: TrainingExample):
        """Append to permanent JSONL dataset."""
        try:
            self.dataset_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dataset_log_path, "a", encoding="utf-8") as f:
                record = {
                    "input": example.input_text,
                    "target": example.target_text,
                    "success": example.success,
                    "ts": example.timestamp,
                    "ctx": example.context,
                    "model_type": self.model_type,
                    # Legacy fields for tool pipelines
                    "intent": example.intent,
                    "tool": example.tool_name,
                    "args": example.arguments,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def infer_with_carry(self, intent: str) -> Tuple[torch.Tensor, float]:
        """
        Run inference on active model with carry state.
        Returns (logits, confidence)
        """
        if not self.active_model:
            return None, 0.0
            
        self.active_model.eval()
        
        # Tokenize
        tokens = self._tokenize(intent)
        inputs = torch.tensor([tokens], device=self.device)
        
        with torch.no_grad():
            # Use carry state
            carry = self.active_carry.z if self.active_carry.is_initialized() else None
            logits, new_carry = self.active_model(inputs, carry_z=carry)
            
            # Update carry
            self.active_carry.z = new_carry
            self.active_carry.steps += 1
        
        # Compute confidence
        probs = F.softmax(logits[:, -1, :], dim=-1)
        confidence = probs.max().item()
        
        return logits, confidence
    
    def _tokenize(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """Simple tokenization."""
        if max_len is None:
            max_len = int(self.model_config.get("seq_len", 256)) if self.model_config else 256
        tokens = []
        for word in text.lower().split():
            if word in self.tokenizer:
                tokens.append(self.tokenizer[word])
            else:
                tokens.append(self.tokenizer.get("<UNK>", 4))
        
        # Pad or truncate
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        
        return tokens
    
    def _training_loop(self):
        """Background training loop."""
        while self._running:
            try:
                # Check if we should train
                if len(self.example_buffer) >= self.min_examples_for_train:
                    self._train_batch()
                
                # Check if we should swap weights
                if self._should_auto_swap():
                    should_swap_batches = self.batches_since_swap >= self.swap_after_n_batches
                    should_swap_time = time.time() - self.last_swap_time >= self.swap_every_seconds
                    
                    if should_swap_batches or should_swap_time:
                        self._swap_weights()
                
                # Sleep before next check
                time.sleep(1)
                
            except Exception as e:
                print(f"[Online Trainer] Error: {e}")
                time.sleep(5)
    
    def _train_batch(self):
        """Train shadow model on collected examples."""
        if not self.shadow_model or len(self.example_buffer) < self.min_examples_for_train:
            return
        
        with self._train_lock:
            # Collect batch
            batch_examples = list(self.example_buffer)[:self.batch_size]
            
            # Prepare training data
            inputs = []
            targets = []
            
            for ex in batch_examples:
                input_text = ex.input_text or ex.intent
                target_text = ex.target_text or ex.tool_name
                if not input_text or not target_text:
                    continue
                # Create input/target tokens
                input_tokens = self._tokenize(input_text)
                target_tokens = self._tokenize(target_text)
                
                inputs.append(input_tokens)
                targets.append(target_tokens)
            
            if not inputs:
                return
            
            # Convert to tensors
            inputs_t = torch.tensor(inputs, device=self.device)
            targets_t = torch.tensor(targets, device=self.device)
            
            # Train step
            self.shadow_model.train()
            optimizer = torch.optim.AdamW(
                self.shadow_model.parameters(), 
                lr=self.learning_rate,
                weight_decay=0.01,
            )
            
            optimizer.zero_grad()
            
            logits, _ = self.shadow_model(inputs_t)
            
            # Simple cross-entropy loss
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets_t[:, 1:].reshape(-1),
                ignore_index=0,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.shadow_model.parameters(), 1.0)
            optimizer.step()
            
            self.stats.shadow_loss = loss.item()
            self.stats.batches_trained += 1
            self.stats.last_train_time = time.time()
            self.batches_since_swap += 1
            
            # Clear used examples
            for _ in range(min(len(batch_examples), len(self.example_buffer))):
                if self.example_buffer:
                    self.example_buffer.popleft()
            
            print(f"[Online Trainer:{self.model_type}] Trained batch {self.stats.batches_trained} (loss={loss.item():.4f})")
    
    def _swap_weights(self):
        """
        Promote shadow weights to active.
        - Active carry state is DISCARDED (fresh start)
        - Shadow becomes active
        - Fresh copy of active becomes new shadow
        """
        if not self.shadow_model or not self.active_model:
            return
        
        with self._train_lock:
            # 1. Save checkpoint of current shadow (the weights we're promoting)
            if self.save_checkpoints:
                self._save_checkpoint("promoted")
            
            # 2. Swap: shadow -> active
            old_active = self.active_model
            self.active_model = self.shadow_model
            self.active_model.eval()
            
            # 3. DISCARD carry state (fresh start with new knowledge)
            self.active_carry = TRMCarryState()
            
            # 4. Create new shadow from new active
            self.shadow_model = copy.deepcopy(self.active_model)
            self.shadow_model.train()
            
            # 5. Delete old active
            del old_active
            
            # 6. Update stats
            self.stats.weight_swaps += 1
            self.stats.last_swap_time = time.time()
            self.last_swap_time = time.time()
            self.batches_since_swap = 0
            
            print(f"[Online Trainer:{self.model_type}] ✓ Weight swap #{self.stats.weight_swaps} - carry state reset")
    
    def _save_checkpoint(self, tag: str = ""):
        """Save checkpoint of current shadow weights."""
        try:
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "model": self.shadow_model.state_dict(),
                "trm_config": self.model_config,
                "model_type": self.model_type,
                "stats": {
                    "examples_collected": self.stats.examples_collected,
                    "batches_trained": self.stats.batches_trained,
                    "weight_swaps": self.stats.weight_swaps,
                },
                "timestamp": time.time(),
            }
            
            filename = f"{self.model_type}_online_{tag}_{int(time.time())}.pt"
            path = CHECKPOINTS_DIR / filename
            torch.save(checkpoint, path)
            
            # Also save as "latest"
            latest_path = CHECKPOINTS_DIR / f"{self.model_type}_online_latest.pt"
            torch.save(checkpoint, latest_path)
            
            print(f"[Online Trainer:{self.model_type}] Saved checkpoint: {filename}")
            
        except Exception as e:
            print(f"[Online Trainer] Checkpoint save failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "model_type": self.model_type,
            "examples_collected": self.stats.examples_collected,
            "batches_trained": self.stats.batches_trained,
            "weight_swaps": self.stats.weight_swaps,
            "buffer_size": len(self.example_buffer),
            "carry_steps": self.active_carry.steps,
            "shadow_loss": self.stats.shadow_loss,
            "batches_since_swap": self.batches_since_swap,
        }
    
    def force_swap(self):
        """Force an immediate weight swap."""
        print("[Online Trainer] Forcing weight swap...")
        self._swap_weights()

    def _should_auto_swap(self) -> bool:
        if self.swap_policy in ("auto", "time", "time+batch", "batch"):
            return True
        if self.swap_policy in ("manual", "none", "off"):
            return False
        if not self._swap_policy_warned:
            print(f"[Online Trainer:{self.model_type}] Unknown swap_policy '{self.swap_policy}', defaulting to auto.", flush=True)
            self._swap_policy_warned = True
        return True


# Global instances
_online_trainers: Dict[str, OnlineTRMTrainer] = {}


def get_online_trainer(model_type: str = "tool") -> Optional[OnlineTRMTrainer]:
    """Get global online trainer instance for a given TRM type."""
    return _online_trainers.get(model_type)


def _resolve_trm_types(config: Dict[str, Any]) -> List[str]:
    cfg = config.get("online_training", {}) or {}
    trm_types = cfg.get("trm_types")
    if isinstance(trm_types, list) and trm_types:
        return [str(t).lower() for t in trm_types]
    # Backward-compatible default
    return ["tool"]


def init_online_trainers(config: Dict[str, Any]) -> Dict[str, OnlineTRMTrainer]:
    """Initialize and return the global online trainers."""
    global _online_trainers
    trm_types = _resolve_trm_types(config)
    created: Dict[str, OnlineTRMTrainer] = {}

    for trm_type in trm_types:
        base_trm = None
        if trm_type == "tool":
            base_trm = get_tool_trm()
        elif trm_type == "memory":
            base_trm = get_memory_trm()
        elif trm_type == "reasoning":
            base_trm = get_reasoning_trm()
        else:
            continue

        trainer = OnlineTRMTrainer(config, model_type=trm_type, base_trm=base_trm)
        created[trm_type] = trainer

    _online_trainers = created
    return _online_trainers


def init_online_trainer(config: Dict[str, Any]) -> OnlineTRMTrainer:
    """Backward-compatible: initialize and return the tool trainer."""
    trainers = init_online_trainers(config)
    return trainers.get("tool")
