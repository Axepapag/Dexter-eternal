import os
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
import json
import aiohttp
import time

from core.response_tank import ResponseTank, get_global_tank
from core.tool_agent_provider import AsyncAIProvider, ChatMessage, resolve_provider_config


class LLMRole(Enum):
    GENERAL = "general"
    REASONING = "reasoning"
    FACT_CHECKER = "fact_checker"
    CREATIVE = "creative"
    CRITICAL = "critical"


@dataclass
class LLMSlotConfig:
    name: str
    url: str
    api_key_env: str
    model: str
    enabled: bool = False
    role: LLMRole = LLMRole.GENERAL
    timeout: float = 30.0
    max_retries: int = 2
    provider_name: Optional[str] = None
    provider_cfg: Optional[Dict[str, Any]] = None
    
    def get_api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


@dataclass
class LLMInsight:
    source: str
    timestamp: float
    insights: List[Dict[str, Any]]
    questions_for_dexter: List[str]
    memory_needs: List[Dict[str, Any]]
    confidence: float
    raw_response: str


class LLMAdvisor:
    def __init__(
        self,
        config: LLMSlotConfig,
        response_tank: ResponseTank
    ):
        self.config = config
        self._tank = response_tank
        self._session: Optional[aiohttp.ClientSession] = None
        self._provider: Optional[AsyncAIProvider] = None
        self._active = config.enabled
    
    async def start(self):
        if not self._active:
            return
        if self.config.provider_name and self.config.provider_cfg:
            self._provider = AsyncAIProvider(self.config.provider_name, self.config.provider_cfg)
            print(f"[LLM Subconscious] {self.config.name}: Online (model: {self.config.model})")
            return

        if self.config.get_api_key() is None:
            print(f"[LLM Subconscious] {self.config.name}: API key not found, disabling")
            self._active = False
            return

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        print(f"[LLM Subconscious] {self.config.name}: Online (model: {self.config.model})")
    
    async def stop(self):
        if self._provider:
            try:
                await self._provider.close()
            except Exception:
                pass
            self._provider = None
        if self._session:
            await self._session.close()
            self._session = None
    
    async def query(
        self,
        context: Dict[str, Any],
        specific_questions: Optional[List[str]] = None
    ) -> Optional[LLMInsight]:
        if not self._active:
            return None
        if not self._provider and not self._session:
            return None
        
        system_prompt = (
            "You are a Subconscious advisor in Dexter's brain."
            " Produce minimal, high-signal contributions as JSON artifacts."
            " Output ONLY JSON (no prose, no markdown)."
            "\n\nSchema:\n"
            "{\n"
            "  \"insights\": [\n"
            "    {\"type\": \"insight|opportunity|strategy|risk|fact\", \"content\": \"...\", \"confidence\": 0.0}\n"
            "  ],\n"
            "  \"questions_for_dexter\": [\"...\"],\n"
            "  \"memory_needs\": [{\"type\": \"triple|episode|pattern\", \"content\": \"...\", \"confidence\": 0.0}]\n"
            "}\n"
            "Rules: keep items short; include confidence; avoid repetition." 
        )
        user_prompt = self._build_user_prompt(context, specific_questions)
        
        try:
            response_data = await self._call_llm(system_prompt, user_prompt)
            insight = self._parse_response(response_data)
            await self._publish_insight(insight)
            return insight
        except Exception as e:
            print(f"[LLM Subconscious] {self.config.name}: Query failed - {e}")
            return None
    
    def _build_user_prompt(
        self,
        context: Dict[str, Any],
        specific_questions: Optional[List[str]]
    ) -> str:
        parts = []
        
        if "user_input" in context:
            parts.append(f"USER INPUT: {context['user_input']}")

        bundle_context = context.get("bundle_context")
        if bundle_context:
            parts.append("\nPERSISTENT CONTEXT:\n" + str(bundle_context))
        
        if specific_questions:
            parts.append(f"\\nSPECIFIC QUESTIONS:")
            for q in specific_questions:
                parts.append(f"- {q}")
        
        return "\\n\\n".join(parts)
    
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, Any]:
        if self._provider:
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
            response = await self._provider.chat(messages, self.config.model, temperature=0.7, max_tokens=2000)
            return {"choices": [{"message": {"content": response.content}}]}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.get_api_key()}"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        for attempt in range(self.config.max_retries):
            try:
                base = self.config.url.rstrip("/")
                # Ollama Cloud uses /api/chat, not /chat/completions
                endpoint = f"{base}/api/chat" if "ollama.com" in base else f"{base}/chat/completions"
                started = time.perf_counter()
                print(f"[LLM Subconscious] {self.config.name}: Requesting {self.config.model} at {endpoint}...", flush=True)

                async with self._session.post(
                    endpoint,
                    headers=headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    elapsed = time.perf_counter() - started
                    print(f"[LLM Subconscious] {self.config.name}: Response {response.status} in {elapsed:.2f}s", flush=True)
                    return await response.json()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))
        
        raise RuntimeError("Max retries exceeded")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> LLMInsight:
        try:
            content = response_data["choices"][0]["message"]["content"]
            
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {
                    "insights": [{"type": "reasoning", "content": content, "confidence": 0.7}],
                    "questions_for_dexter": [],
                    "memory_needs": []
                }
            
            return LLMInsight(
                source=self.config.name,
                timestamp=time.time(),
                insights=parsed.get("insights", []),
                questions_for_dexter=parsed.get("questions_for_dexter", []),
                memory_needs=parsed.get("memory_needs", []),
                confidence=sum(i.get("confidence", 0.7) for i in parsed.get("insights", [])) / max(1, len(parsed.get("insights", []))),
                raw_response=content
            )
        except Exception as e:
            return LLMInsight(
                source=self.config.name,
                timestamp=time.time(),
                insights=[{"type": "reasoning", "content": str(response_data), "confidence": 0.5}],
                questions_for_dexter=[],
                memory_needs=[],
                confidence=0.5,
                raw_response=str(response_data)
            )
    
    async def _publish_insight(self, insight: LLMInsight):
        await self._tank.publish(
            source=f"llm_{self.config.name}",
            content={
                "type": "llm_insight",
                "insight": {
                    "source": insight.source,
                    "timestamp": insight.timestamp,
                    "insights": insight.insights,
                    "questions_for_dexter": insight.questions_for_dexter,
                    "memory_needs": insight.memory_needs,
                    "confidence": insight.confidence
                }
            },
            priority="medium"
        )


class LLMSubconscious:
    def __init__(self, config: Dict[str, Any]):
        self._root_config = config
        self._config = config.get("llm_subconscious", {})
        self._slots: Dict[str, LLMAdvisor] = {}
        self._tank = get_global_tank()
        self._initialize_slots()
    
    def _initialize_slots(self):
        slot_configs = self._config.get("slots", [])
        for slot_cfg in slot_configs:
            provider_name = slot_cfg.get("provider_name")
            provider_cfg = None
            if provider_name:
                _, provider_cfg = resolve_provider_config(self._root_config, provider_name)
            slot_config = LLMSlotConfig(
                name=slot_cfg.get("name", "unnamed"),
                url=slot_cfg.get("url", ""),
                api_key_env=slot_cfg.get("api_key_env", ""),
                model=slot_cfg.get("model", ""),
                enabled=slot_cfg.get("enabled", False),
                provider_name=provider_name,
                provider_cfg=provider_cfg,
            )
            self._slots[slot_config.name] = LLMAdvisor(slot_config, self._tank)
    
    async def start(self):
        enabled_count = 0
        for advisor in self._slots.values():
            await advisor.start()
            if advisor._active:
                enabled_count += 1
        
        print(f"[LLM Subconscious] {enabled_count} advisors online")
    
    async def stop(self):
        for advisor in self._slots.values():
            await advisor.stop()
        print("[LLM Subconscious] All advisors offline")
    
    async def broadcast(
        self,
        context: Dict[str, Any],
        specific_questions: Optional[List[str]] = None,
        timeout: float = 5.0,
        min_insights: int = 1
    ) -> List[LLMInsight]:
        tasks = [
            advisor.query(context, specific_questions)
            for advisor in self._slots.values()
            if advisor._active
        ]
        
        if not tasks:
            print("[LLM Subconscious] No enabled advisors")
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        insights = []
        for result in results:
            if isinstance(result, LLMInsight):
                insights.append(result)
            elif isinstance(result, Exception):
                print(f"[LLM Subconscious] Query error: {result}")
        
        print(f"[LLM Subconscious] Received {len(insights)} insights")
        return insights
    
    async def query_specific(
        self,
        slot_name: str,
        context: Dict[str, Any],
        specific_questions: Optional[List[str]] = None
    ) -> Optional[LLMInsight]:
        advisor = self._slots.get(slot_name)
        if not advisor:
            print(f"[LLM Subconscious] Slot {slot_name} not found")
            return None
        
        return await advisor.query(context, specific_questions)
    
    def get_enabled_slots(self) -> List[str]:
        return [
            name for name, advisor in self._slots.items()
            if advisor._active
        ]
    
    def is_available(self) -> bool:
        return any(advisor._active for advisor in self._slots.values())
