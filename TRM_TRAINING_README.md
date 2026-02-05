# TRM Training System for Dexter

## Overview

This system transforms raw Dexter operational data into powerful trained **Tiny Recursive Models (TRMs)** that can operate as fast, local, intelligent subsystems.

### What Are TRMs?

TRMs are small neural networks (7M-50M parameters) that use **recursive reasoning** - they think by iteratively refining their internal state across multiple cycles:

- **H-cycles (High-level)**: Strategic reasoning about the problem
- **L-cycles (Low-level)**: Tactical refinement of solutions

Unlike LLMs which are stateless (fresh context every call), TRMs can maintain **working memory** (`z_H`, `z_L`) across conversation turns - enabling truly stateful cognition.

---

## Architecture

```
Raw Data (dexter_TRMs/datasets/raw/)
    │
    ├── d2_memory.json        (conversations, facts, entities)
    ├── brain.db              (SQLite: facts, triples, patterns, history)
    ├── training_data_fast.jsonl  (UI interactions, tool calls)
    └── art_*.json            (tool execution artifacts)
    │
    ▼
Dataset Generator (generate_trm_training_data.py)
    │
    ├── Memory Dataset        (query → relevant memories)
    ├── Tool Dataset          (task → tool selection)
    └── Reasoning Dataset     (goal → plan steps)
    │
    ▼
Training Pipeline (train_all_trms.py)
    │
    ├── Memory TRM            (256 hidden, 2 layers, H3/L4 cycles)
    ├── Tool TRM              (192 hidden, 2 layers, H2/L3 cycles)
    └── Reasoning TRM         (256 hidden, 3 layers, H4/L4 cycles)
    │
    ▼
Trained Models (dexter_TRMs/models/)
    │
    ├─► MemoryTRM Wrapper     (stateful memory retrieval)
    ├─► ToolTRM Wrapper       (tool selection with error correction)
    └─► ReasoningTRM Wrapper  (stateful planning & reasoning)
```

---

## Quick Start

### Step 1: Build Everything

```bash
python build_trm_brain.py
```

This single command:
1. Generates datasets from all raw data
2. Trains all three TRMs
3. Saves models ready for integration

**Time:** ~2-4 hours depending on your hardware

### Step 2: Use the Trained TRMs

```python
from trained_trm_wrappers import MemoryTRM, ToolTRM, ReasoningTRM

# Load trained models
memory = MemoryTRM.from_checkpoint("dexter_TRMs/models/memory/best.pt", use_carry=True)
tool = ToolTRM.from_checkpoint("dexter_TRMs/models/tool/best.pt", use_carry=False)
reasoning = ReasoningTRM.from_checkpoint("dexter_TRMs/models/reasoning/best.pt", use_carry=True)

# Use Memory TRM (retrieves relevant context)
result = memory.retrieve("What website projects do I have?")
print(result["memories"])  # List of relevant facts/episodes
print(result["confidence"])  # 0.0-1.0

# Use Tool TRM (selects appropriate tool)
result = tool.select_tool(
    task="Check system status",
    available_tools=["system_ops.get_info", "file_system.list"]
)
print(result["selected_tool"])  # "system_ops.get_info"

# Use Reasoning TRM (generates plans)
result = reasoning.plan("Build an income-generating website")
print(result["plan_steps"])  # ["analyze market", "design layout", "implement payment"...]
print(result["complexity"])  # "simple", "moderate", "complex"

# Stateful reasoning (multi-step)
completed = []
for i in range(5):
    step_result = reasoning.next_step("Build website", completed)
    print(f"Step {i+1}: {step_result['next_step']}")
    completed.append(step_result['next_step'])

# Reset carry when starting new conversation
memory.reset_carry()
reasoning.reset_carry()
```

---

## TRM Details

### Memory TRM

**Purpose:** Retrieve relevant memories from knowledge graph

**Training Data:**
- d2_memory.json (conversations, facts, entities, topics)
- brain.db (facts table, triples table, patterns table, history table)

**Architecture:**
- Hidden: 256
- Layers: 2
- H-cycles: 3 (deep retrieval reasoning)
- L-cycles: 4 (fine-grained relevance)
- Vocab: ~8,000 tokens

**Stateful Mode:** Yes
- Carry (`z_H`, `z_L`) persists across calls
- Enables conversation-contextual memory retrieval

**Input Format:**
```
<MEM_QUERY> {user query} context: {optional context}
```

**Output Format:**
```
<MEM_FACT> {fact 1} <SEP> {fact 2} <SEP> {fact 3}
```

---

### Tool TRM

**Purpose:** Select the best tool for a task, with error correction and context-aware decision making

**Training Data:**
- art_*.json (tool execution results)
- training_data_fast.jsonl (UI context → tool calls)
- Error patterns for correction learning

**Architecture:**
- Hidden: 192
- Layers: 2
- H-cycles: 2 (task understanding)
- L-cycles: 3 (tool matching)

**Stateful Mode:** **YES - CRITICAL**
- Carry state tracks tool execution history
- Learns from previous errors in the session
- Contextual tool selection based on conversation flow

**Input Format:**
```
<TOOL_CALL> task: {description} available: [{tools}] context: {state}
<TOOL_ERROR> {previous error} <TOOL_CALL> task: {description}  # For correction
```

**Output Format:**
```
<TOOL_RESULT> {tool_name}  # Success
<TOOL_ERROR> {tool_name} {error}  # Predicted failure
```

---

### Reasoning TRM

**Purpose:** Generate plans, make decisions, think through problems

**Training Data:**
- d2_memory.json (summaries → task patterns)
- brain.db (patterns table as reasoning templates)
- Synthetic goal → plan pairs

**Architecture:**
- Hidden: 256
- Layers: 3
- H-cycles: 4 (deep strategic reasoning)
- L-cycles: 4 (tactical step refinement)

**Stateful Mode:** Yes
- Carry accumulates reasoning context
- Enables multi-step plan refinement

**Input Format:**
```
<REASON_TASK> {goal} state: {current state} constraints: [{constraints}]
```

**Output Format:**
```
<REASON_PLAN> step1 then step2 then step3...
<REASON_STEP> next specific action
```

---

## Stateful Operation (REQUIRED)

### ALL TRMs Are Stateful - Architecture Principle

```python
# ALL TRMs default to stateful mode
trm = MemoryTRM.from_checkpoint("memory_best.pt")    # Stateful
trm = ToolTRM.from_checkpoint("tool_best.pt")        # Stateful  
trm = ReasoningTRM.from_checkpoint("reasoning_best.pt")  # Stateful

# Carry state persists across calls
result1 = trm.retrieve("What projects?")
# Internal state (z_H, z_L) now encodes "projects" context

result2 = trm.retrieve("Tell me more")  
# TRM "remembers" we were talking about projects!

result3 = trm.retrieve("Which one is most profitable?")
# TRM understands "which one" refers to projects!
```

### Why Stateful is Mandatory

1. **Memory TRM**: Must track conversation context to retrieve relevant memories
2. **Tool TRM**: Must remember previous tool attempts/errors to avoid repetition
3. **Reasoning TRM**: Must accumulate reasoning context across planning steps

### Reset Protocol

```python
# Start of new conversation - reset ALL TRMs
def reset_all_trms():
    memory_trm.reset_carry()
    tool_trm.reset_carry()
    reasoning_trm.reset_carry()
    print("All TRM states reset for new conversation")

# Call when:
# - User starts new conversation
# - Topic completely changes
# - You want to clear working memory
```

### Stateful Architecture Benefits

- **Contextual Awareness**: TRMs understand "it", "that", "those" references
- **Error Learning**: Tool TRM remembers what failed and adjusts
- **Plan Continuity**: Reasoning TRM maintains plan context across steps
- **Conversation Flow**: Memory TRM builds understanding over time

---

## Training from Scratch

### Just Generate Datasets

```bash
python generate_trm_training_data.py
```

Outputs:
- `dexter_TRMs/datasets/offline/memory/`
- `dexter_TRMs/datasets/offline/tool/`
- `dexter_TRMs/datasets/offline/reasoning/`
- `dexter_TRMs/datasets/offline/unified_vocab.json`

### Just Train Models

```bash
python train_all_trms.py
```

### Train Individual TRMs

```python
from train_all_trms import train_trm

train_trm(
    trm_type="memory",
    data_path=Path("dexter_TRMs/datasets/offline/memory/memory_training_data.npz"),
    vocab_size=8000,
    epochs=30,
    batch_size=16,
    hidden_size=256,
    H_cycles=3,
    L_cycles=4,
)
```

---

## Integration into New Dexter Architecture

### Suggested Architecture

```python
class UnifiedConsciousness:
    def __init__(self):
        # Stateful TRMs (carry persists)
        self.memory_trm = MemoryTRM.from_checkpoint("memory_best.pt", use_carry=True)
        self.reasoning_trm = ReasoningTRM.from_checkpoint("reasoning_best.pt", use_carry=True)
        
        # Stateless TRMs (fresh each call)
        self.tool_trm = ToolTRM.from_checkpoint("tool_best.pt", use_carry=False)
        
        # Cloud LLM for complex reasoning
        self.llm_think_tank = LLMThinkTank()
    
    async def process_input(self, user_input: str):
        # 1. Retrieve memories (stateful)
        memory_result = self.memory_trm.retrieve(user_input)
        
        # 2. Quick tool selection (stateless)
        tool_result = self.tool_trm.select_tool(user_input, available_tools)
        
        # 3. Assess complexity
        if tool_result["confidence"] > 0.9:
            # Fast path: Execute tool immediately
            return await self.execute_tool(tool_result["selected_tool"])
        
        # 4. Complex reasoning (stateful)
        plan_result = self.reasoning_trm.plan(user_input)
        
        # 5. Rich context for LLM
        context = self.assemble_context(
            base_bundle=get_base_bundle(),
            memories=memory_result["memories"],
            plan=plan_result["plan_steps"],
            tool_suggestion=tool_result["selected_tool"],
        )
        
        # 6. LLM for nuanced response
        response = await self.llm_think_tank.chat(context)
        
        return response
    
    def reset_conversation(self):
        """Call when user starts new conversation."""
        self.memory_trm.reset_carry()
        self.reasoning_trm.reset_carry()
```

---

## Performance Benchmarks

### Inference Speed (CPU)

- **Memory TRM**: ~50-100ms per query
- **Tool TRM**: ~30-60ms per selection
- **Reasoning TRM**: ~80-150ms per plan

### Inference Speed (GPU/CUDA)

- **Memory TRM**: ~5-10ms per query
- **Tool TRM**: ~3-5ms per selection
- **Reasoning TRM**: ~8-15ms per plan

### Model Sizes

- **Memory TRM**: ~15MB
- **Tool TRM**: ~8MB
- **Reasoning TRM**: ~18MB

**Total**: ~40MB for complete local brain!

---

## Troubleshooting

### "Module not found" errors

```bash
pip install torch numpy tqdm
```

### Training is slow

- TRMs train faster on GPU. Install PyTorch with CUDA:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```

- Reduce epochs in `train_all_trms.py` for faster iteration

### Low accuracy after training

- Check dataset size in logs (need 1000+ examples per TRM)
- Increase epochs: `epochs=100` in training config
- Adjust learning rate: `lr=5e-5` for fine-tuning

### Out of memory during training

- Reduce batch size: `batch_size=8` or `batch_size=4`
- Reduce hidden size: `hidden_size=128`

---

## Files Reference

| File | Purpose |
|------|---------|
| `build_trm_brain.py` | Master orchestration script |
| `generate_trm_training_data.py` | Dataset generation from raw data |
| `train_all_trms.py` | Training pipeline for all TRMs |
| `trained_trm_wrappers.py` | Ready-to-use TRM classes |
| `core/trm_base.py` | Base TRM architecture |
| `TinyRecursiveModels/` | Full TRM paper implementation |

---

## Next Steps

1. **Run the builder**: `python build_trm_brain.py`
2. **Test the TRMs**: `python trained_trm_wrappers.py`
3. **Integrate** into your new unified consciousness architecture
4. **Add more training data** → retrain → improve accuracy
5. **Experiment** with stateful vs stateless modes

---

## Credits

- **TRM Architecture**: Based on "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)
- **HRM Foundation**: Hierarchical Reasoning Model (Wang et al., 2025)
- **Implementation**: Dexter Gliksbot Cognitive Architecture

---

**Ready to build your TRM-powered brain?**

```bash
python build_trm_brain.py
```
