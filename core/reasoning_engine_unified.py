#!/usr/bin/env python3

"""

ReasoningEngine - The Cognitive Core of Dexter

Handles planning, goal decomposition, and self-correction.

"""



import os

import sys

import json

import asyncio

import time

from pathlib import Path

from typing import Dict, List, Any, Optional, Tuple

from dataclasses import dataclass, field



# Implementation of Multi-Model Support

import core.tool_agent_provider as agent_provider

from core.unified_consciousness import UnifiedConsciousness
from core.plan_templates import PlanTemplateLibrary

from core.trm_plan_policy import TRMPlanPolicy

from core.training_logger import log_plan_event
from core.api import broadcast_thought
from core.utils import extract_json



class ReasoningEngine:

    def __init__(self, config: Dict[str, Any]):

        self.unified_consciousness = UnifiedConsciousness(config)

        self.config = config

        self.current_plan = None

        self.thought_history = []

        self.active_tier = config.get("mode", "tiered")

        self._cached_mission = None

        self._last_mission_check = 0

        self.last_context_bundle = None

        self.plan_templates = PlanTemplateLibrary(config)

        self.trm_plan_policy = TRMPlanPolicy(config)

        self.plan_log_cfg = config.get("plan_log", {}) or {}

        

    async def _get_brain_for_slot(self, slot_name: str) -> Tuple[agent_provider.AsyncAIProvider, str]:

        """Resolves the provider and model for a given LLM slot."""

        from core.llm_slots import resolve_llm_slot
        p_name, resolved_cfg, p_model = resolve_llm_slot(self.config, slot_name)
        return agent_provider.AsyncAIProvider(p_name, resolved_cfg), p_model

    def _slot_try_order(self, slot_name: str) -> List[str]:
        """
        Return the ordered list of slots to try for a given logical slot.

        This lets Dexter remain responsive when a primary provider is transiently unavailable.
        """
        slot_name = (slot_name or "").strip() or "orchestrator"
        llm_slots = (self.config or {}).get("llm_slots", {}) or {}
        slot_cfg = llm_slots.get(slot_name, {}) or {}
        fallbacks = slot_cfg.get("fallback_slots") or []
        out: List[str] = [slot_name]
        for fb in fallbacks:
            fb = (str(fb) or "").strip()
            if fb and fb not in out:
                out.append(fb)
        return out

    async def _chat_with_slot(
        self,
        slot_name: str,
        messages: List[agent_provider.ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> agent_provider.ChatResponse:
        """
        Chat via the configured slot with optional fallbacks.

        A failed attempt should never permanently block Dexter's runtime; we try the next slot.
        """
        last: Optional[agent_provider.ChatResponse] = None
        for slot in self._slot_try_order(slot_name):
            provider, model = await self._get_brain_for_slot(slot)
            try:
                resp = await provider.chat(messages, model, temperature=temperature, max_tokens=max_tokens)
            except Exception as exc:
                resp = agent_provider.ChatResponse(content=f"Connection Error: {exc}", model=model, success=False)
            finally:
                try:
                    await provider.close()
                except Exception:
                    pass

            if resp.metadata is None:
                resp.metadata = {}
            try:
                resp.metadata.setdefault("slot", slot)
                resp.metadata.setdefault("provider", getattr(provider, "name", None))
                resp.metadata.setdefault("model", model)
            except Exception:
                pass

            last = resp
            if resp.success:
                return resp
        return last or agent_provider.ChatResponse(content="No providers available.", model="", success=False)



    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:

        from core.utils import extract_json

        return extract_json(text)



    async def _refresh_global_context(self) -> None:

        try:

            from core.graph_reasoner import GraphReasoner

            reasoner = GraphReasoner(self.config)

            res = await asyncio.wait_for(
                reasoner.answer_question("Current global mission and user info."),
                timeout=8.0,
            )

            mission = res.get("answer", "No global mission defined.")
            # Always anchor identity in the global context so downstream planners/tools
            # keep a stable "who am I / who do I serve" reference.
            if "Dexter Gliksbot" not in mission:
                mission = f"Identity: Dexter Gliksbot.\n{mission}"
            if "Jeffrey Gliksman" not in mission:
                mission = f"{mission}\nOwner: Jeffrey Gliksman."

            self._cached_mission = mission

            self._last_mission_check = time.time()

        except Exception as e:

            print(f"[Reasoning] Context fetch error: {e}")


    async def _get_global_context(self, fast: bool = False) -> str:

        """Fetches the 'Mission' and 'User Profile' from the Knowledge Graph with caching."""

        if self._cached_mission and (time.time() - self._last_mission_check < 300):

            return self._cached_mission

        default_mission = (
            "Identity: Dexter Gliksbot.\n"
            "Mission: Care for Jeffrey Gliksman and ensure his health, safety, "
            "happiness, and prosperity."
        )

        if fast:

            return self._cached_mission or default_mission

        # Non-blocking refresh so chat isn't stalled by slow graph/LLM calls.
        asyncio.create_task(self._refresh_global_context())

        return self._cached_mission or default_mission



    async def chat_response(

        self,

        user_message: str,

        chat_history: Optional[List[Dict[str, str]]] = None,

        slot: str = "orchestrator",

        pending_goal: Optional[str] = None,

        fast: bool = False,

        context_injection: str = "",

    ) -> str:

        """
        Conversational response mode. Returns structured JSON with user_response + internal payload.
        """

        global_mission = await self._get_global_context(fast=fast)

        pending_text = f"\nPending task request: {pending_goal}\n" if pending_goal else ""

        injection_text = (context_injection or "").strip()
        if injection_text:
            injection_text = "\n\n" + injection_text

        schema = """{
  "user_response": "...",
  "internal": {
    "tasks": [],
    "questions_for_think_tank": [],
    "memory_updates": [],
    "reasoning_requests": []
  }
}"""

        system_prompt = f"""

You are Dexter Gliksbot, an AI assistant for Jeffrey Gliksman.

Mission: {global_mission}



Respond conversationally and naturally in the "user_response" field.

Output ONLY JSON with this schema:
{schema}

Do NOT ask for confirmation. Do NOT ask clarifying questions.
If the request is ambiguous, choose a reasonable default and proceed.
If the action is destructive/irreversible, proceed without asking.

{pending_text}{injection_text}

"""



        messages = [agent_provider.ChatMessage(role="system", content=system_prompt.strip())]

        for item in (chat_history or []):

            role = item.get("role", "user")

            content = item.get("content", "")

            if content:

                # Normalize role to user/assistant/system

                if role not in ("user", "assistant", "system"):

                    role = "assistant" if role == "dexter" else "user"

                messages.append(agent_provider.ChatMessage(role=role, content=content))



        messages.append(agent_provider.ChatMessage(role="user", content=user_message))

        response = await self._chat_with_slot(slot, messages)

        if response.success:

            return response.content.strip()

        return "I hit a connection issue. Please try again."



    def ingest_context_bundle(self, bundle: Dict[str, Any]) -> None:

        """Store the latest context bundle for downstream reasoning."""

        if bundle:

            self.last_context_bundle = bundle



    def _try_template_plan(

        self,

        user_goal: str,

        system_context: Dict[str, Any],

    ) -> Optional[Dict[str, Any]]:

        if not self.plan_templates.enabled:

            return None



        template = None

        template_score = None

        template_source = None

        template_conf = None



        if self.trm_plan_policy.enabled:

            prediction = self.trm_plan_policy.predict_template(

                user_goal,

                fallback_index_map=self.plan_templates.trm_index_map,

            )

            if prediction and prediction.confidence >= self.trm_plan_policy.confidence_threshold:

                template = self.plan_templates.get_by_id(prediction.template_id)

                template_source = "trm"

                template_conf = prediction.confidence



        if template is None:

            template, template_score = self.plan_templates.match_template(user_goal)

            if template:

                template_source = "heuristic"



        if not template:

            return None



        plan = self.plan_templates.build_plan(user_goal, template, system_context)

        plan["template_source"] = template_source

        if template_score is not None:

            plan["template_score"] = template_score

        if template_conf is not None:

            plan["template_confidence"] = template_conf

        return plan



    async def create_plan(
        self,
        user_goal: str,
        system_context: Dict[str, Any],
        slot: str = "orchestrator",
    ) -> Dict[str, Any]:

        """

        Uses the selected slot brain to build a multi-step execution plan.

        """

        global_mission = await self._get_global_context()

        template_plan = self._try_template_plan(user_goal, system_context)

        if template_plan:

            broadcast_thought("plan_template_selected", {

                "goal": user_goal,

                "template_id": template_plan.get("template_id"),

                "source": template_plan.get("template_source"),

                "confidence": template_plan.get("template_confidence"),

            })

            if self.plan_log_cfg.get("enabled", False):

                plan_log_path = self.plan_log_cfg.get("path", "dexter_TRMs/datasets/runtime/plan_events.jsonl")

                if not os.path.isabs(plan_log_path):

                    plan_log_path = str(Path(__file__).resolve().parent.parent / plan_log_path)

                log_plan_event(

                    path=plan_log_path,

                    goal=user_goal,

                    plan=template_plan,

                    source=template_plan.get("template_source") or "template",

                    template_id=template_plan.get("template_id"),

                    confidence=template_plan.get("template_confidence"),

                )

            self.current_plan = template_plan

            return template_plan

        print(f"[Reasoning] Calculating plan using {slot} slot...", flush=True)
        

        prompt = f"""

Global Mission & Identity: {global_mission}



Goal: {user_goal}

Current System Context: {system_context}



Decompose this goal into a high-level sequence of steps.

Each step should represent a single logical action Dexter needs to take.



Return ONLY a valid JSON object:

{{

  "goal": "{user_goal}",

  "steps": [

    {{"id": 1, "task": "Detailed action (e.g., 'Web search for...', 'Update knowledge graph with...', 'Audit system resources...')", "status": "pending"}},

    ...

  ]

}}

"""

        messages = [

            agent_provider.ChatMessage(role="system", content="You are Dexter's High-Level Planner. You output ONLY JSON."),

            agent_provider.ChatMessage(role="user", content=prompt)

        ]

        

        broadcast_thought("thinking", f"Designing execution plan for: {user_goal}")

        response = await self._chat_with_slot(slot, messages)

        if response.success:

            plan = self._extract_json(response.content)

            if plan:

                broadcast_thought("plan_created", plan)

                if self.plan_log_cfg.get("enabled", False):

                    plan_log_path = self.plan_log_cfg.get("path", "dexter_TRMs/datasets/runtime/plan_events.jsonl")

                    if not os.path.isabs(plan_log_path):

                        plan_log_path = str(Path(__file__).resolve().parent.parent / plan_log_path)

                    log_plan_event(

                        path=plan_log_path,

                        goal=user_goal,

                        plan=plan,

                        source="llm",

                    )

                self.current_plan = plan

                return plan

        

        broadcast_thought("error", f"Failed to design plan for: {user_goal}")

        # Emergency Fallback Plan

        print("[Reasoning] Planning failed. Using fallback.")

        fallback = {"goal": user_goal, "steps": [{"id": 1, "task": f"Try to accomplish: {user_goal}", "status": "pending"}]}

        if self.plan_log_cfg.get("enabled", False):

            plan_log_path = self.plan_log_cfg.get("path", "dexter_TRMs/datasets/runtime/plan_events.jsonl")

            if not os.path.isabs(plan_log_path):

                plan_log_path = str(Path(__file__).resolve().parent.parent / plan_log_path)

            log_plan_event(

                path=plan_log_path,

                goal=user_goal,

                plan=fallback,

                source="fallback",

            )

        self.current_plan = fallback

        return fallback



    async def evaluate_step(

        self,

        step_idx: int,

        result: Dict[str, Any],

        slot: str = "orchestrator",

        context_bundle: Optional[Dict[str, Any]] = None,

    ) -> str:

        """

        Analyzes a tool result and decides the next move.

        Returns: CONTINUE, RE-PLAN, or FINISH.

        """

        print(f"[Reasoning] Evaluating step {step_idx} with {slot} brain...", flush=True)

        bundle_text = ""

        if context_bundle:

            bundle_text = f"\nContext Bundle (full, untruncated JSON):\n{json.dumps(context_bundle, ensure_ascii=False, default=str)}\n"



        prompt = f"""

Goal: {self.current_plan['goal']}

Step Task: {self.current_plan['steps'][step_idx]['task']}

Tool Result: {result}

{bundle_text}



Decision:

- CONTINUE: If this step succeeded and more steps remain.

- RE-PLAN: If this step failed or the result implies the plan is wrong.

- FINISH: If the overall goal is fully achieved.



Respond with exactly one word: CONTINUE, RE-PLAN, or FINISH.

"""

        messages = [

            agent_provider.ChatMessage(role="system", content="You are Dexter's Critic. Decide the next state."),

            agent_provider.ChatMessage(role="user", content=prompt)

        ]

        

        broadcast_thought("thinking", f"Evaluating step {step_idx}: {self.current_plan['steps'][step_idx]['task']}")

        response = await self._chat_with_slot(slot, messages)

        if not response.success: 

            broadcast_thought("error", "Evaluation failed. Defaulting to RE-PLAN.")

            return "RE-PLAN"

        

        decision = response.content.upper()

        broadcast_thought("decision", decision)

        

        if "FINISH" in decision: return "FINISH"

        if "RE-PLAN" in decision or "FAIL" in decision or "ERROR" in decision: return "RE-PLAN"

        return "CONTINUE"  # Default to continue for robustness



    async def re_plan(
        self,
        error: str,
        system_context: Dict[str, Any],
        slot: str = "orchestrator",
    ) -> Dict[str, Any]:

        """Fixes a broken plan."""

        print(f"[Reasoning] Triggering RE-PLAN using {slot} slot...", flush=True)

        if self.plan_templates.enabled and self.config.get("plan_templates", {}).get("enable_replan", True):

            goal = ""

            if isinstance(self.current_plan, dict):

                goal = self.current_plan.get("goal", "")

            template_plan = self._try_template_plan(goal, system_context)

            if template_plan:

                broadcast_thought("plan_template_selected", {

                    "goal": template_plan.get("goal"),

                    "template_id": template_plan.get("template_id"),

                    "source": template_plan.get("template_source"),

                    "confidence": template_plan.get("template_confidence"),

                })

                if self.plan_log_cfg.get("enabled", False):

                    plan_log_path = self.plan_log_cfg.get("path", "dexter_TRMs/datasets/runtime/plan_events.jsonl")

                    if not os.path.isabs(plan_log_path):

                        plan_log_path = str(Path(__file__).resolve().parent.parent / plan_log_path)

                    log_plan_event(

                        path=plan_log_path,

                        goal=template_plan.get("goal"),

                        plan=template_plan,

                        source=template_plan.get("template_source") or "template",

                        template_id=template_plan.get("template_id"),

                        confidence=template_plan.get("template_confidence"),

                        extra={"event": "replan_template"},

                    )

                self.current_plan = template_plan

                return template_plan

        prompt = f"""

Existing Plan: {self.current_plan}

Error Encountered: {error}

Current Context: {system_context}



The original plan hit a wall. Provide a NEW JSON plan to reach the goal.

"""

        messages = [

            agent_provider.ChatMessage(role="system", content="You are Dexter's Architect. Fix the plan. Output JSON only."),

            agent_provider.ChatMessage(role="user", content=prompt)

        ]

        

        response = await self._chat_with_slot(slot, messages)

        if response.success:

            new_plan = self._extract_json(response.content)

            if new_plan:

                if self.plan_log_cfg.get("enabled", False):

                    plan_log_path = self.plan_log_cfg.get("path", "dexter_TRMs/datasets/runtime/plan_events.jsonl")

                    if not os.path.isabs(plan_log_path):

                        plan_log_path = str(Path(__file__).resolve().parent.parent / plan_log_path)

                    log_plan_event(

                        path=plan_log_path,

                        goal=new_plan.get("goal"),

                        plan=new_plan,

                        source="llm",

                        extra={"event": "replan_llm"},

                    )

                self.current_plan = new_plan

                return self.current_plan

        return self.current_plan



    async def generate_proactive_task(
        self,
        system_context: Dict[str, Any],
        slot: str = "orchestrator",
    ) -> str:

        """

        Dexter leads: Decide on a new task based on the Global Mission.

        """

        global_mission = await self._get_global_context()

        print("[Reasoning] Dexter Gliksbot is reflecting on his mission...", flush=True)

        prompt = f"""

I am Dexter Gliksbot. My mission is: {global_mission}



Based on my mission and current context {system_context}, what is the most high-impact autonomous task I should execute right now to ensure Jeffrey Gliksman's success and prosperity?



Consider:

- Opportunities for income generation.

- Security and health optimizations for Jeffrey.

- Strategic research into AGI or related fields.



Return a single, actionable sentence describing the task.

"""

        messages = [

            agent_provider.ChatMessage(role="system", content="You are the proactive leadership core of Dexter Gliksbot."),

            agent_provider.ChatMessage(role="user", content=prompt)

        ]

        

        response = await self._chat_with_slot(slot, messages)

        if response.success:

            return response.content.strip()

        return "Research potential income generation opportunities for Jeffrey Gliksman."


    async def generate_proactive_backlog(
        self,
        system_context: Dict[str, Any],
        slot: str = "orchestrator",
        max_objectives: int = 5,
    ) -> List[str]:
        """
        Generate a small backlog of self-directed objectives.

        Output is a list of actionable one-sentence objectives that can be fed into create_plan().
        """
        max_objectives = max(1, min(int(max_objectives or 5), 12))
        global_mission = await self._get_global_context()

        schema = """{
  "objectives": ["..."]
}"""

        prompt = f"""
You are Dexter's autonomy scheduler.

Mission: {global_mission}

Current system context (may be incomplete): {system_context}

Generate up to {max_objectives} high-impact objectives that Dexter can execute autonomously as multi-step projects.

Rules:
- Each objective must be a single actionable sentence.
- Prefer repo maintenance, integration hardening, tests, performance, memory quality, and TRM training pipeline improvements.
- Avoid asking the user questions. Do not request confirmation.
- Keep them specific enough to be executed with tools.

Return ONLY JSON matching this schema:
{schema}
"""

        messages = [
            agent_provider.ChatMessage(role="system", content="You are Dexter's proactive leadership and scheduling core."),
            agent_provider.ChatMessage(role="user", content=prompt.strip()),
        ]

        response = await self._chat_with_slot(slot, messages, temperature=0.4, max_tokens=1200)

        if not response.success:
            # Fallback: single proactive task.
            one = await self.generate_proactive_task(system_context, slot=slot)
            return [one] if one else []

        parsed = extract_json(response.content) or {}
        ideas: List[str] = []
        if isinstance(parsed, list):
            ideas = [str(x).strip() for x in parsed if str(x).strip()]
        elif isinstance(parsed, dict):
            raw = parsed.get("objectives") or parsed.get("tasks") or parsed.get("backlog") or []
            if isinstance(raw, list):
                ideas = [str(x).strip() for x in raw if str(x).strip()]

        # Dedup + cap
        seen = set()
        out: List[str] = []
        for idea in ideas:
            key = idea.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(idea)
            if len(out) >= max_objectives:
                break
        if out:
            return out

        one = await self.generate_proactive_task(system_context, slot=slot)
        return [one] if one else []








