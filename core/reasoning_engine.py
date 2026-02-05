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
from core.api import broadcast_thought
from core.plan_templates import PlanTemplateLibrary
from core.trm_plan_policy import TRMPlanPolicy
from core.training_logger import enqueue_plan_event, log_plan_event

class ReasoningEngine:
    def __init__(self, config: Dict[str, Any]):
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
            self._cached_mission = mission
            self._last_mission_check = time.time()
        except Exception as e:
            print(f"[Reasoning] Context fetch error: {e}")

    async def _get_global_context(self, fast: bool = False) -> str:
        """Fetches the 'Mission' and 'User Profile' from the Knowledge Graph with caching."""
        if self._cached_mission and (time.time() - self._last_mission_check < 300):
            return self._cached_mission
        default_mission = (
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
    ) -> str:
        """
        Conversational response mode. Returns a natural language reply (no JSON).
        """
        global_mission = await self._get_global_context(fast=fast)
        provider, model = await self._get_brain_for_slot(slot)

        pending_text = f"\nPending task request: {pending_goal}\n" if pending_goal else ""
        system_prompt = f"""
You are Dexter Gliksbot, an AI assistant for Jeffrey Gliksman.
Mission: {global_mission}

Respond conversationally and naturally. Do NOT mention planning or tool calls.
Do NOT output JSON. Do NOT ask for confirmation or clarifying questions.
If the user requests an action, proceed without confirmation.
{pending_text}
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

        response = await provider.chat(messages, model)
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
                enq_ok = enqueue_plan_event(
                    path=plan_log_path,
                    goal=user_goal,
                    plan=template_plan,
                    source=template_plan.get("template_source") or "template",
                    template_id=template_plan.get("template_id"),
                    confidence=template_plan.get("template_confidence"),
                )
                if not enq_ok:
                    asyncio.create_task(
                        asyncio.to_thread(
                            log_plan_event,
                            path=plan_log_path,
                            goal=user_goal,
                            plan=template_plan,
                            source=template_plan.get("template_source") or "template",
                            template_id=template_plan.get("template_id"),
                            confidence=template_plan.get("template_confidence"),
                        )
                    )
            self.current_plan = template_plan
            return template_plan
        print(f"[Reasoning] Calculating plan using {slot} slot...", flush=True)
        provider, model = await self._get_brain_for_slot(slot)
        
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
        response = await provider.chat(messages, model)
        if response.success:
            plan = self._extract_json(response.content)
            if plan:
                broadcast_thought("plan_created", plan)
                if self.plan_log_cfg.get("enabled", False):
                    plan_log_path = self.plan_log_cfg.get("path", "dexter_TRMs/datasets/runtime/plan_events.jsonl")
                    if not os.path.isabs(plan_log_path):
                        plan_log_path = str(Path(__file__).resolve().parent.parent / plan_log_path)
                    enq_ok = enqueue_plan_event(
                        path=plan_log_path,
                        goal=user_goal,
                        plan=plan,
                        source="llm",
                    )
                    if not enq_ok:
                        asyncio.create_task(
                            asyncio.to_thread(
                                log_plan_event,
                                path=plan_log_path,
                                goal=user_goal,
                                plan=plan,
                                source="llm",
                            )
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
            enq_ok = enqueue_plan_event(
                path=plan_log_path,
                goal=user_goal,
                plan=fallback,
                source="fallback",
            )
            if not enq_ok:
                asyncio.create_task(
                    asyncio.to_thread(
                        log_plan_event,
                        path=plan_log_path,
                        goal=user_goal,
                        plan=fallback,
                        source="fallback",
                    )
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
        provider, model = await self._get_brain_for_slot(slot)
        
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
        response = await provider.chat(messages, model)
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
                    enq_ok = enqueue_plan_event(
                        path=plan_log_path,
                        goal=template_plan.get("goal"),
                        plan=template_plan,
                        source=template_plan.get("template_source") or "template",
                        template_id=template_plan.get("template_id"),
                        confidence=template_plan.get("template_confidence"),
                        extra={"event": "replan_template"},
                    )
                    if not enq_ok:
                        asyncio.create_task(
                            asyncio.to_thread(
                                log_plan_event,
                                path=plan_log_path,
                                goal=template_plan.get("goal"),
                                plan=template_plan,
                                source=template_plan.get("template_source") or "template",
                                template_id=template_plan.get("template_id"),
                                confidence=template_plan.get("template_confidence"),
                                extra={"event": "replan_template"},
                            )
                        )
                self.current_plan = template_plan
                return template_plan
        provider, model = await self._get_brain_for_slot(slot)
        
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
        
        response = await provider.chat(messages, model)
        if response.success:
            new_plan = self._extract_json(response.content)
            if new_plan:
                if self.plan_log_cfg.get("enabled", False):
                    plan_log_path = self.plan_log_cfg.get("path", "dexter_TRMs/datasets/runtime/plan_events.jsonl")
                    if not os.path.isabs(plan_log_path):
                        plan_log_path = str(Path(__file__).resolve().parent.parent / plan_log_path)
                    enq_ok = enqueue_plan_event(
                        path=plan_log_path,
                        goal=new_plan.get("goal"),
                        plan=new_plan,
                        source="llm",
                        extra={"event": "replan_llm"},
                    )
                    if not enq_ok:
                        asyncio.create_task(
                            asyncio.to_thread(
                                log_plan_event,
                                path=plan_log_path,
                                goal=new_plan.get("goal"),
                                plan=new_plan,
                                source="llm",
                                extra={"event": "replan_llm"},
                            )
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
        provider, model = await self._get_brain_for_slot(slot)
        
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
        
        response = await provider.chat(messages, model)
        if response.success:
            return response.content.strip()
        return "Research potential income generation opportunities for Jeffrey Gliksman."
