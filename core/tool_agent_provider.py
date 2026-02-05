import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# Optional imports for specific providers
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from ollama import AsyncClient as OllamaAsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaAsyncClient = None


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    content: str
    model: str
    success: bool = True
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


def load_tool_agent_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def resolve_provider_config(config: Dict[str, Any], provider_name: str = None) -> Tuple[str, Dict[str, Any]]:
    """
    Resolves the configuration for a specific provider.
    Priority:
    1. 'providers' dictionary in core_config.json
    2. 'custom_providers' in legacy config
    """
    if not provider_name:
        base_provider = config.get("provider", {})
        provider_name = base_provider.get("name", "openai")

    # Check new 'providers' structure
    providers = config.get("providers", {})
    if provider_name in providers:
        merged = dict(providers[provider_name])
        merged["name"] = provider_name
        # Map 'type' to 'provider_type' for internal compatibility
        if "type" in merged:
            merged["provider_type"] = merged["type"]
        return provider_name, merged

    # Legacy support
    custom_providers = config.get("custom_providers", {})
    if provider_name in custom_providers:
        merged = dict(custom_providers[provider_name])
        merged["name"] = provider_name
        return provider_name, merged
        
    return provider_name, {"name": provider_name}


def determine_model(provider_config: Dict[str, Any], base_config: Dict[str, Any]) -> str:
    # 1. Check explicit override in base config
    if base_config.get("provider", {}).get("model"):
        return base_config["provider"]["model"]
    # 2. Check provider specific model
    model = provider_config.get("model")
    if model:
        return model
    # 3. Check provider specific models list (first one)
    models = provider_config.get("models") or []
    return models[0] if models else "gpt-4o-mini"


class AsyncAIProvider:
    """Async multi-provider AI client (OpenAI, Ollama, Gemini, Bedrock, Vertex)."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config or {}
        self.provider_type = self.config.get("provider_type", "openai_compatible")
        self.api_key = self._get_api_key()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ollama_client = None
        self._aws_bedrock_client = None

    def _get_api_key(self) -> Optional[str]:
        if self.config.get("api_key_env"):
            return os.getenv(self.config["api_key_env"])
        return self.config.get("api_key")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout_total = float(self.config.get("timeout_sec", 60))
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=timeout_total,
                    connect=timeout_total,
                    sock_connect=timeout_total,
                    sock_read=timeout_total,
                )
            )
        return self._session

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        started = time.perf_counter()
        timeout_sec = float(self.config.get("timeout_sec", 60))

        async def _run_with_timeout(coro: asyncio.Future) -> ChatResponse:
            try:
                return await asyncio.wait_for(coro, timeout=timeout_sec)
            except asyncio.TimeoutError:
                return ChatResponse(
                    content=f"Timeout after {timeout_sec:.0f}s",
                    model=model,
                    success=False,
                )
            except Exception as exc:
                return ChatResponse(
                    content=f"Provider Error: {exc}",
                    model=model,
                    success=False,
                )

        ptype = self.provider_type.lower()
        
        if ptype == "ollama":
            resp = await _run_with_timeout(self._chat_ollama(messages, model, temperature, max_tokens))
            elapsed = time.perf_counter() - started
            extra = f" err={resp.content[:120]!r}" if not resp.success else ""
            print(f"[LLM] {self.name}/{model} ({ptype}) done in {elapsed:.2f}s (ok={resp.success}){extra}", flush=True)
            return resp
        elif ptype == "ollama_cloud":
            resp = await _run_with_timeout(self._chat_ollama_cloud(messages, model, temperature, max_tokens))
            elapsed = time.perf_counter() - started
            extra = f" err={resp.content[:120]!r}" if not resp.success else ""
            print(f"[LLM] {self.name}/{model} ({ptype}) done in {elapsed:.2f}s (ok={resp.success}){extra}", flush=True)
            return resp
        elif ptype in ("google_gemini", "vertex_maas", "google_vertex"):
             # "vertex_maas" often uses the same requests structure as Vertex AI or a raw endpoint. 
             # For simpler integration, if it provides an absolute URL, we treat it as a REST call.
             if "googleapis.com" in self.config.get("base_url", ""):
                 resp = await _run_with_timeout(self._chat_vertex_rest(messages, model, temperature, max_tokens))
                 elapsed = time.perf_counter() - started
                 extra = f" err={resp.content[:120]!r}" if not resp.success else ""
                 print(f"[LLM] {self.name}/{model} ({ptype}) done in {elapsed:.2f}s (ok={resp.success}){extra}", flush=True)
                 return resp
             resp = await _run_with_timeout(self._chat_gemini(messages, model, temperature, max_tokens))
             elapsed = time.perf_counter() - started
             extra = f" err={resp.content[:120]!r}" if not resp.success else ""
             print(f"[LLM] {self.name}/{model} ({ptype}) done in {elapsed:.2f}s (ok={resp.success}){extra}", flush=True)
             return resp
        elif ptype == "aws_bedrock":
            resp = await _run_with_timeout(self._chat_aws_bedrock(messages, model, temperature, max_tokens))
            elapsed = time.perf_counter() - started
            extra = f" err={resp.content[:120]!r}" if not resp.success else ""
            print(f"[LLM] {self.name}/{model} ({ptype}) done in {elapsed:.2f}s (ok={resp.success}){extra}", flush=True)
            return resp
        
        # Default to OpenAI Compatible (NVIDIA, Groq, DeepSeek, LocalAI, etc.)
        resp = await _run_with_timeout(self._chat_openai_compatible(messages, model, temperature, max_tokens))
        elapsed = time.perf_counter() - started
        extra = f" err={resp.content[:120]!r}" if not resp.success else ""
        print(f"[LLM] {self.name}/{model} (openai_compatible) done in {elapsed:.2f}s (ok={resp.success}){extra}", flush=True)
        return resp

    async def _chat_openai_compatible(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        session = await self._get_session()
        base_url = self.config.get("base_url", "https://api.openai.com/v1")
        if base_url.endswith("/"):
            base_url = base_url[:-1]
            
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Handle special provider headers or adjustments here if needed
        # NVIDIA NIM is OpenAI compatible but sometimes strict on endpoints
        endpoint = f"{base_url}/chat/completions"
        print(f"[_chat_openai] Requesting {model} at {endpoint}...", flush=True)

        try:
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                print(f"[_chat_openai] Response status {resp.status} from {endpoint}", flush=True)
                if resp.status != 200:
                    error_text = await resp.text()
                    return ChatResponse(
                        content=f"API Error ({resp.status}): {error_text}", 
                        model=model, 
                        success=False
                    )
                
                data = await resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return ChatResponse(content="Empty response from API", model=model, success=False)
                    
                message = choices[0].get("message", {}) or {}
                content = message.get("content")
                if not content:
                    # Some OpenAI-compatible providers (e.g., NVIDIA NIM) return content in reasoning_content or text.
                    content = (
                        message.get("reasoning_content")
                        or choices[0].get("text")
                        or ""
                    )
                return ChatResponse(
                    content=content,
                    model=data.get("model", model),
                    success=True,
                    finish_reason=choices[0].get("finish_reason"),
                    usage=data.get("usage"),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ChatResponse(content=f"Connection Error: {exc}", model=model, success=False)

    async def _chat_ollama(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        if not OLLAMA_AVAILABLE:
            return ChatResponse(content="Ollama Python client (ollama) not installed", model=model, success=False)

        try:
            if not self._ollama_client:
                host = self.config.get("base_url", "http://localhost:11434")
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                client_timeout = int(float(self.config.get("timeout_sec", 30)))
                self._ollama_client = OllamaAsyncClient(host=host, headers=headers, timeout=client_timeout)
                
            formatted = [{"role": m.role, "content": m.content} for m in messages]
            options = {"temperature": temperature}
            # NOTE: Ollama Cloud currently returns empty content when num_predict is set.
            # Avoid passing max_tokens for cloud to keep responses non-empty.
                
            print(f"[_chat_ollama] Requesting model {model} at {self._ollama_client._client.base_url}...", flush=True)
            response = await self._ollama_client.chat(
                model=model,
                messages=formatted,
                options=options,
            )
            print(f"[_chat_ollama] Response received.", flush=True)
            content = response.message.content if response.message else ""
            return ChatResponse(content=content, model=model, success=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ChatResponse(content=f"Ollama Error: {exc}", model=model, success=False)

    async def _chat_ollama_cloud(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """
        Chat with Ollama Cloud REST API at {base_url}/api/chat.

        We deliberately use aiohttp here instead of the `ollama` Python client because we've
        observed cases where timeouts/cancellation don't interrupt the underlying request
        promptly, which can stall Dexter's interactive loop.
        """
        try:
            session = await self._get_session()
            base_url = (self.config.get("base_url") or "https://ollama.com").rstrip("/")
            endpoint = f"{base_url}/api/chat"

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            formatted = [{"role": m.role, "content": m.content} for m in messages]
            payload: Dict[str, Any] = {
                "model": model,
                "messages": formatted,
                "stream": False,
                "options": {"temperature": temperature},
            }
            # NOTE: Ollama Cloud returns empty content when num_predict is set.
            # Do not pass max_tokens/num_predict for cloud.

            print(f"[_chat_ollama_cloud] Requesting {model} at {endpoint}...", flush=True)
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                raw = await resp.text()
                if resp.status != 200:
                    return ChatResponse(
                        content=f"Ollama Cloud Error ({resp.status}): {raw[:2000]}",
                        model=model,
                        success=False,
                    )

            print(f"[_chat_ollama_cloud] Response received.", flush=True)

            # Normal mode: JSON object. Some servers may still return NDJSON; handle both.
            try:
                data = json.loads(raw) if raw.strip() else {}
                if isinstance(data, dict):
                    msg = data.get("message") or {}
                    if isinstance(msg, dict) and (msg.get("content") is not None):
                        return ChatResponse(content=str(msg.get("content") or ""), model=model, success=True)
                    if data.get("response") is not None:
                        return ChatResponse(content=str(data.get("response") or ""), model=model, success=True)
                    return ChatResponse(content=str(data), model=model, success=True)
            except Exception:
                pass

            # NDJSON fallback: accumulate deltas.
            parts: List[str] = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                if isinstance(evt, dict):
                    msg = evt.get("message") or {}
                    if isinstance(msg, dict) and msg.get("content"):
                        parts.append(str(msg["content"]))
                    elif evt.get("response"):
                        parts.append(str(evt["response"]))
            return ChatResponse(content="".join(parts), model=model, success=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ChatResponse(content=f"Ollama Cloud Error: {exc}", model=model, success=False)

    async def _chat_gemini(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        session = await self._get_session()
        base_url = self.config.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        
        contents = []
        for msg in messages:
            if msg.role == "system":
                # Gemini often handles system prompts differently or as context
                # For basic compatibility, we prepend to user message or use proper 'system' role if supported by version
                # Here we just treat it as user part for safety or skip if strictly system-instruction oriented
                # A safer bet for general Gemini API is "user" and "model" roles solely.
                role = "user"
            else:
                role = "user" if msg.role == "user" else "model"
            
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        key_param = f"?key={self.api_key}" if self.api_key else ""
        endpoint = f"{base_url}/models/{model}:generateContent{key_param}"

        try:
            async with session.post(endpoint, json=payload) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return ChatResponse(content=f"Gemini API Error: {error}", model=model, success=False)
                
                data = await resp.json()
                # Parse robustly
                try:
                    candidates = data.get("candidates", [])
                    if not candidates:
                         return ChatResponse(content="No candidates returned", model=model, success=False)
                    
                    content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return ChatResponse(content=content, model=model, success=True)
                except (IndexError, AttributeError):
                    return ChatResponse(content=f"Unexpected Gemini response format: {data}", model=model, success=False)
                    
        except Exception as exc:
             return ChatResponse(content=f"Gemini Error: {exc}", model=model, success=False)

    async def _chat_vertex_rest(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Interact with Vertex AI / MaaS endpoints via raw REST."""
        session = await self._get_session()
        endpoint = self.config.get("base_url")
        if not endpoint:
            return ChatResponse(content="Missing base_url for Vertex provider", model=model, success=False)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Format for typical Vertex Chat prediction (Publisher Model)
        # Structure often: { "instances": [ { "messages": [...] } ], "parameters": { ... } }
        # NOTE: Verify specific model format (Moonshot may differ). Using standard Vertex format here.
        
        vertex_messages = []
        for m in messages:
            vertex_messages.append({
                "role": m.role,  # often "user" or "assistant"
                "content": m.content
            })

        payload = {
            "instances": [
                { "messages": vertex_messages }
            ],
            "parameters": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens or 2048
            }
        }

        try:
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    return ChatResponse(content=f"Vertex Error ({resp.status}): {err}", model=model, success=False)
                
                data = await resp.json()
                # Extract prediction
                try:
                    predictions = data.get("predictions", [])
                    if predictions:
                        # Depending on model, it might be text or structured
                        content = predictions[0].get("candidates", [{}])[0].get("content") or str(predictions[0])
                        return ChatResponse(content=str(content), model=model, success=True)
                    else:
                        return ChatResponse(content=f"No predictions: {data}", model=model, success=False)
                except Exception as parse_err:
                     return ChatResponse(content=f"Parse Error: {parse_err} | Data: {data}", model=model, success=False)

        except Exception as exc:
            return ChatResponse(content=f"Vertex Request Failed: {exc}", model=model, success=False)

    async def _chat_aws_bedrock(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        if not AWS_AVAILABLE:
            return ChatResponse(content="Boto3 not installed", model=model, success=False)
        
        region = self.config.get("region", "us-east-1")
        
        # We invoke strictly in a thread to avoid blocking the async loop
        import asyncio
        return await asyncio.to_thread(self._invoke_bedrock_sync, messages, model, temperature, max_tokens, region)

    def _invoke_bedrock_sync(
        self, 
        messages: List[ChatMessage], 
        model: str, 
        temperature: float, 
        max_tokens: Optional[int],
        region: str
    ) -> ChatResponse:
        try:
            if not self._aws_bedrock_client:
                # Assuming environment variables for AWS_ACCESS_KEY_ID etc are set
                # or user has configured ~/.aws/credentials
                self._aws_bedrock_client = boto3.client("bedrock-runtime", region_name=region)

            # Claude 3 format (Messages API)
            bedrock_messages = []
            system_prompts = []
            
            for m in messages:
                if m.role == "system":
                    system_prompts.append({"text": m.content})
                else:
                    role = "user" if m.role == "user" else "assistant"
                    bedrock_messages.append({
                        "role": role,
                        "content": [{"text": m.content}]
                    })
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or 2000,
                "messages": bedrock_messages,
                "temperature": temperature,
            }
            if system_prompts:
                body["system"] = system_prompts

            response = self._aws_bedrock_client.invoke_model(
                modelId=model,
                body=json.dumps(body)
            )
            
            resp_body = json.loads(response.get("body").read())
            content_block = resp_body.get("content", [])
            text_content = "".join([item.get("text", "") for item in content_block if item.get("type") == "text"])
            
            return ChatResponse(
                content=text_content,
                model=model,
                success=True,
                usage=resp_body.get("usage")
            )

        except Exception as exc:
            return ChatResponse(content=f"Bedrock Error: {exc}", model=model, success=False)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
