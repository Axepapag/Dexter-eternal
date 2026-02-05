#!/usr/bin/env python3
"""
train_librarian.py - Trains a Tiny Recursive Model (TRM) to classify goals into skills.
Used by SkillLibrarian for sub-cortical retrieval.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
# Fix module pathing
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.trm_base import TRMConfig, BaseTRM

class LibrarianDataset(Dataset):
    def __init__(self, data_path: Path, vocab_size: int, seq_len: int):
        self.data = []
        self.skill_map = {}
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.seq_len = seq_len
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if item["skill_id"] not in self.skill_map:
                        self.skill_map[item["skill_id"]] = len(self.skill_map)
                    
                    # Tokenize goal
                    tokens = self._tokenize(item["goal"])
                    self.data.append((tokens, self.skill_map[item["skill_id"]]))
                except Exception:
                    continue
                
        print(f"[Dataset] Loaded {len(self.data)} examples and {len(self.skill_map)} skills.")

    def _tokenize(self, text: str):
        words = text.lower().split()
        ids = []
        for w in words:
            if w not in self.vocab and len(self.vocab) < 1024:
                self.vocab[w] = len(self.vocab)
            ids.append(self.vocab.get(w, self.vocab["<UNK>"]))
        
        # Pad/Truncate
        if len(ids) < self.seq_len:
            ids += [0] * (self.seq_len - len(ids))
        else:
            ids = ids[:self.seq_len]
        return np.array(ids, dtype=np.int32)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train_librarian_model(repo_root: Path, data_path: Optional[Path] = None, save_path: Optional[Path] = None, epochs: int = 50):
    if data_path is None:
        data_path = repo_root / "memory" / "librarian_training_data.jsonl"
    if save_path is None:
        save_path = repo_root / "models" / "librarian_core.pt"
    
    if not data_path.exists():
        print(f"[Trainer] Data path {data_path} not found. Skipping.")
        return False

    dataset = LibrarianDataset(data_path, vocab_size=1024, seq_len=32)
    if len(dataset) == 0:
        print("[Trainer] Dataset empty. Skipping.")
        return False
        
    dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
    
    config = TRMConfig(num_classes=len(dataset.skill_map))
    model = BaseTRM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"[Trainer] Training Librarian TRM for {len(dataset.skill_map)} skills ({len(dataset)} examples)...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} loss: {total_loss/len(dataloader):.4f}")
            
    # Save model + Metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": dataset.vocab,
        "skill_map": dataset.skill_map,
        "config": config
    }
    
    # Ensure dir exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"[Trainer] Offline Intelligence Upgraded: {save_path}")
    return True

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    train_librarian_model(repo_root)
