# TRM Architecture Mandate: ALL TRMs MUST BE STATEFUL

## Core Principle

**Every TRM in the Dexter cognitive architecture is STATEFUL. Period.**

There are no exceptions. There are no stateless TRMs. The `use_carry=False` mode does not exist in production.

---

## Why Statefulness is Mandatory

### The TRM's Core Innovation

The Tiny Recursive Model's entire advantage over standard neural networks comes from its **recursive state evolution**:

- **H-state (z_H)**: High-level reasoning context
- **L-state (z_L)**: Low-level working memory

These states evolve through H-cycles and L-cycles, enabling the model to "think" by iteratively refining its internal representations.

### Without Statefulness, TRMs Are Useless

A stateless TRM is just a small, weak feedforward network:
- No working memory
- No context accumulation
- No learning within a session
- Just pattern matching - inferior to even simple heuristics

### With Statefulness, TRMs Are Powerful

Stateful TRMs possess:
- **Working memory** across conversation turns
- **Contextual understanding** that deepens over time
- **Error correction** that learns from mistakes
- **Reasoning chains** that build on previous thoughts

---

## Implementation Requirements

### 1. Default to Stateful

```python
# CORRECT - Stateful by default
class AnyTRM(TrainedTRM):
    @classmethod
    def from_checkpoint(cls, path, device="cpu", use_carry=True):
        return cls(path, vocab_path, device, use_carry=True)

# WRONG - Never do this
class AnyTRM(TrainedTRM):
    @classmethod  
    def from_checkpoint(cls, path, device="cpu", use_carry=False):  # HERESY
        return cls(path, vocab_path, device, use_carry=False)  # CRITICAL ERROR
```

### 2. Carry State Management

Every TRM maintains:
```python
@dataclass
class CarryState:
    z_H: torch.Tensor  # High-level reasoning state
    z_L: torch.Tensor  # Low-level working memory
    step_count: int    # How many steps we've reasoned
```

### 3. Reset Protocol

```python
# At conversation start
def initialize_session():
    memory_trm.reset_carry()    # Fresh memory context
    tool_trm.reset_carry()      # Fresh tool history
    reasoning_trm.reset_carry() # Fresh reasoning chains

# At conversation end
def terminate_session():
    # Optional: Save carry states for long-term learning
    save_carry_states()
    
    # Reset for hygiene
    memory_trm.reset_carry()
    tool_trm.reset_carry()
    reasoning_trm.reset_carry()
```

### 4. Cross-TRM State Sharing

```python
class UnifiedConsciousness:
    def synchronize_carry_states(self):
        """
        Allow TRMs to share state context.
        Memory TRM's z_H informs Reasoning TRM's z_H.
        """
        # Memory carry → Reasoning carry (context alignment)
        self.reasoning_trm.carry.z_H += (
            self.memory_trm.carry.z_H * 0.1  # Soft update
        )
        
        # Reasoning carry → Tool carry (intent alignment)
        self.tool_trm.carry.z_H += (
            self.reasoning_trm.carry.z_H * 0.1
        )
```

---

## TRM-Specific Statefulness

### Memory TRM Statefulness

**What it remembers:**
- Conversation topics and themes
- User preferences and patterns
- Retrieved memory context
- Query-to-memory mappings

**State evolution:**
```
Turn 1: Retrieve "projects"
  z_H encodes: "projects" concept space
  
Turn 2: Retrieve "which is profitable"
  z_H now encodes: "projects" + "profitability" + relationship
  Uses "which" → refers to projects (contextual understanding!)
```

### Tool TRM Statefulness

**What it remembers:**
- Previously attempted tools
- Success/failure patterns
- Error types and frequencies
- User's preferred tool workflows

**State evolution:**
```
Attempt 1: Try "shell.run"
  Fails with "permission denied"
  z_H encodes: "shell" + "permission error"
  
Attempt 2: Try "powershell.execute"
  z_L carries: "previous permission error"
  Selects tool with elevated privileges
  Learns from failure without explicit re prompting!
```

### Reasoning TRM Statefulness

**What it remembers:**
- Current plan and goals
- Completed steps
- Failed approaches
- Alternative paths explored
- Reasoning chains

**State evolution:**
```
Step 1: Plan "Build website"
  z_H encodes: website concept + initial approach

Step 2: Next step after "analyze market"
  z_H carries: "market analysis complete"
  Next step: "design layout" (continues naturally)
  
Step 5: Plan refinement
  z_L accumulates: steps 1-4 context
  Can backtrack and replan based on evolved understanding
```

---

## Anti-Patterns (NEVER DO THESE)

### Anti-Pattern 1: Stateless TRM

```python
# NEVER
class ToolTRM:
    def select_tool(self, task):
        # Fresh model call
        return model.predict(task)  # No carry - useless!
```

### Anti-Pattern 2: Manual State Management

```python
# NEVER - Don't manually prompt instead of using carry
def get_tool_with_context(task, previous_errors):
    prompt = f"Task: {task}\nPrevious errors: {previous_errors}"
    return trm.predict(prompt)  # Wrong! Use carry instead
    
# Correct
trm.select_tool(task)  # Carry already has error history!
```

### Anti-Pattern 3: Resetting Too Frequently

```python
# NEVER - Don't reset between every call
for query in queries:
    result = trm.retrieve(query)
    trm.reset_carry()  # DESTROYS context!
    
# Correct
for query in queries:
    result = trm.retrieve(query)  # Carry accumulates!
# Reset only at conversation boundary
trm.reset_carry()
```

---

## Stateful Architecture Benefits

### 1. Emergent Contextual Understanding

Stateful TRMs naturally develop understanding through state evolution:
- Start with: "Website project"
- Evolve to: "Website project for client X with deadline Y using tech Z"
- Without explicit prompting!

### 2. Implicit Error Correction

Tool TRM state captures failure patterns:
- Carry state after 3 failed `shell.run` attempts
- Implicitly encodes: "shell commands failing, try alternatives"
- 4th call automatically explores different approaches

### 3. Reasoning Continuity

Reasoning TRM maintains train of thought:
- Step 1: "Analyze market"
- Step 5: "Wait, market analysis suggested different approach"
- Can backtrack and replan coherently

### 4. Conversation Coherence

Memory TRM builds topic models:
- Query 1: "What about Python?"
- Query 5: "How do I install it?"
- TRM understands "it" = Python from carry state

---

## Verification Checklist

Before any TRM is considered complete:

- [ ] Uses `use_carry=True` by default
- [ ] Initializes carry in `__init__`
- [ ] Updates carry in every `forward()` call
- [ ] Provides `reset_carry()` method
- [ ] Carry state is inspectable for debugging
- [ ] Can save/load carry for persistence
- [ ] Documentation states "This TRM is stateful"

---

## Code Review Gates

Any PR introducing a TRM must pass:

1. **Statefulness Gate**: `use_carry` defaults to True
2. **Carry Evolution Gate**: Carry is updated in forward pass
3. **Reset Gate**: `reset_carry()` method exists
4. **Documentation Gate**: Docstring mentions statefulness
5. **Architecture Gate**: Reviewer confirms no stateless modes

---

## Consequences of Violation

Violating the "all TRMs stateful" mandate:

1. **Cognitive Fracture**: TRMs can't maintain context
2. **Context Loss**: Every call starts fresh (like LLMs)
3. **Error Repetition**: Tool TRM repeats failed attempts
4. **Plan Amnesia**: Reasoning TRM forgets the plan mid-execution
5. **User Frustration**: "I told you that already!" syndrome

---

## Summary

**All TRMs are stateful. Always. No exceptions.**

The `z_H` and `z_L` carry states are not optional features - they are the essence of what makes a TRM a TRM. Without them, we have a small dumb neural network. With them, we have a thinking subsystem that can maintain context, learn within sessions, and build understanding over time.

**Stateful TRMs are the neurons of Dexter's brain. Treat them accordingly.**

---

## Quick Reference

```python
# Creating TRMs - All stateful
memory = MemoryTRM.from_checkpoint(path)      # Stateful ✓
tool = ToolTRM.from_checkpoint(path)          # Stateful ✓  
reasoning = ReasoningTRM.from_checkpoint(path) # Stateful ✓

# Using TRMs - Carry persists
trm.retrieve("What projects?")      # Carry evolves
trm.retrieve("Tell me more")        # Has context!

# Resetting - Only at boundaries
trm.reset_carry()  # New conversation only

# Cross-TRM sync
consciousness.synchronize_carry_states()  # Share context
```

**Stateful. Always. Forever.**
