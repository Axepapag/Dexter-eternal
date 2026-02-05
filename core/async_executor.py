#!/usr/bin/env python3
"""
async_executor.py - Parallel tool execution engine for Dexter
Spawns a worker pool and executes skill tools concurrently.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import core.tool_executor as tool_executor


@dataclass
class ToolExecutionRequest:
    tool_name: str
    arguments: Dict[str, Any]
    meta: Dict[str, Any]
    future: asyncio.Future


class AsyncToolExecutor:
    def __init__(
        self,
        max_workers: int = 20,
        result_handler: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Awaitable[None]]] = None,
    ):
        self.max_workers = max(1, int(max_workers))
        self._queue: asyncio.Queue[Optional[ToolExecutionRequest]] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._started = False
        self._result_handler = result_handler

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for idx in range(self.max_workers):
            self._workers.append(asyncio.create_task(self._worker(idx)))

    async def shutdown(self) -> None:
        if not self._started:
            return
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def submit(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> asyncio.Future:
        if not self._started:
            await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = ToolExecutionRequest(
            tool_name=tool_name,
            arguments=arguments or {},
            meta=meta or {},
            future=future,
        )
        await self._queue.put(request)
        return future

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        future = await self.submit(tool_name, arguments, meta=meta)
        return await future

    async def _worker(self, worker_id: int) -> None:
        while True:
            request = await self._queue.get()
            if request is None:
                break
            started = time.monotonic()
            result: Dict[str, Any]
            try:
                result = await tool_executor.execute_tool(request.tool_name, request.arguments)
            except Exception as exc:
                result = {
                    "ok": False,
                    "success": False,
                    "error": str(exc),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }

            if self._result_handler:
                try:
                    await self._result_handler(request.meta, result)
                except Exception:
                    pass

            if not request.future.cancelled():
                request.future.set_result(result)
