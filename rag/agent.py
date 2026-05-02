from __future__ import annotations

import json
from typing import Any

from .config import RAGConfig
from .mcp.runtime import MCPRuntime
from .mcp.schemas import MCPToolSpec
from .orchestrator import ModularRAGOrchestrator
from .platform.kb_catalog import KnowledgeBaseCatalogItem, build_catalog_prompt
from .prompts import AGENT_SYSTEM_PROMPT
from .runtime import RuntimeStore
from .schemas import StructuredRAGResponse
from .tools.retrieve import RetrievalTool


class ToolCallingRAGAgent:
    def __init__(
        self,
        orchestrator: ModularRAGOrchestrator | None = None,
        config: RAGConfig | None = None,
        *,
        orchestrators_by_kb: dict[str, ModularRAGOrchestrator] | None = None,
        available_kbs: list[dict[str, Any]] | None = None,
        default_kb_ids: list[str] | None = None,
        mcp_tools: list[MCPToolSpec] | None = None,
        mcp_runtime: MCPRuntime | None = None,
    ):
        self.config = config or RAGConfig()
        self.orchestrator = orchestrator or ModularRAGOrchestrator(self.config)
        self.orchestrators_by_kb = orchestrators_by_kb or {"default": self.orchestrator}
        self.available_kbs = list(available_kbs or [])
        self.available_kb_ids = {
            str(item.get("kb_id"))
            for item in self.available_kbs
            if item.get("kb_id")
        }
        self.default_kb_ids = list(default_kb_ids or self.available_kb_ids or ["default"])
        self.messages: list[dict[str, Any]] = []
        self.runtime = RuntimeStore()
        self.traces: list[dict[str, Any]] = []
        self.mcp_tools = [tool for tool in list(mcp_tools or []) if tool.enabled]
        self.mcp_runtime = mcp_runtime

    def run(
        self,
        query: str,
        *,
        history: list[str] | None = None,
        topk: int | None = None,
        max_rounds: int | None = None,
    ) -> StructuredRAGResponse:
        active_topk = topk or self.config.default_topk
        active_max_rounds = max_rounds or self.config.max_rounds
        conversation_history = list(history or [])
        self._reset_runtime(query, conversation_history)

        if not self.orchestrator.llm.available:
            return self._fallback_run(
                query,
                history=conversation_history,
                topk=active_topk,
            )

        self.messages = self._build_initial_messages(
            query,
            conversation_history,
            active_topk,
            active_max_rounds,
        )

        for round_index in range(1, active_max_rounds + 1):
            response = self.orchestrator.llm.chat_with_tools(
                messages=self.messages,
                tools=self._tool_schemas(),
                max_tokens=1600,
            )
            self.messages.append(response.assistant_message)

            if not response.tool_calls:
                return self._auto_finish(
                    query=query,
                    reason="Agent returned plain text instead of calling finish.",
                    response_text=response.content,
                    tool_rounds=round_index,
                )

            for tool_call in response.tool_calls:
                result = self._execute_tool(
                    tool_name=tool_call.name,
                    arguments=tool_call.arguments,
                    history=conversation_history,
                    default_topk=active_topk,
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                self.traces.append(
                    {
                        "round": round_index,
                        "tool": tool_call.name,
                        "agent_decision": self._extract_agent_decision(
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            assistant_content=response.content,
                        ),
                        "arguments": tool_call.arguments,
                        "result": self._summarize_result(tool_call.name, result),
                    }
                )
                if tool_call.name == "finish":
                    return self._to_structured_response(
                        result,
                        round_index=round_index,
                        tool_rounds=round_index,
                    )

        return self._auto_finish(
            query=query,
            reason="Reached the maximum number of tool-call rounds before finish.",
            response_text=self.runtime.last_valid_answer_content(),
            tool_rounds=active_max_rounds,
        )

    def _reset_runtime(self, query: str = "", history: list[str] | None = None) -> None:
        self.messages = []
        self.runtime.reset(query=query, history=history)
        self.traces = []

    def _build_initial_messages(
        self,
        query: str,
        history: list[str],
        topk: int,
        max_rounds: int,
    ) -> list[dict[str, Any]]:
        history_text = "\n".join(history[-8:]) if history else "(empty)"
        kb_catalog_text = self._kb_catalog_prompt()
        user_message = (
            f"用户问题：\n{query}\n\n"
            f"对话记忆：\n{history_text}\n\n"
            f"可用知识库：\n{kb_catalog_text}\n\n"
            f"工作约束：\n"
            f"- topk: {topk}\n"
            f"- 如果用户是在询问上一轮/刚才的回答、之前问过什么等对话回顾问题，直接调用 generate(source_mode=\"memory\")，不要为了形式检索知识库。\n"
            f"- 最大工具调用轮数: {max_rounds}\n"
            f"- 必须使用工具完成生成；只有需要知识库证据时才先检索。\n"
            f"- 调用 retrieve 时，请从可用知识库中选择 kb_ids；如果不确定，可选择多个；不要选择未列出的 kb_id。\n"
            f"- 每次生成答案草稿后，判断是继续检索、重新生成，还是 finish。\n"
        )
        if self.mcp_tools:
            mcp_tool_lines = "\n".join(
                f"- {tool.function_name}: {tool.description}"
                for tool in self.mcp_tools
            )
            user_message += (
                "\n\nAvailable MCP tools:\n"
                f"{mcp_tool_lines}\n\n"
                "MCP usage policy:\n"
                "- Prefer local retrieve first for knowledge-base questions.\n"
                "- Use web-search MCP tools when local retrieval is empty, weak, unavailable, or the user asks for current web information.\n"
                "- If an MCP tool returns retrieval_id, call generate with that retrieval_id before finish.\n"
            )
        return [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _tool_schemas(self) -> list[dict[str, Any]]:
        decision_schema = self._decision_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve",
                    "description": (
                        "执行检索。需要 query 改写或真正多意图拆解时，自己写 plans。不要拆解简单单意图问题。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": decision_schema,
                            "query": {"type": "string"},
                            "kb_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "从可用知识库中选择一个或多个 kb_id。",
                            },
                            "plans": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "retrieval_modes": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "enum": ["semantic", "keyword", "grep"],
                                            },
                                        },
                                    },
                                    "required": ["query", "retrieval_modes"],
                                },
                            },
                            "retrieval_modes": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["semantic", "keyword", "grep"],
                                },
                            },
                            "topk": {"type": "integer", "minimum": 1},
                        },
                        "required": ["decision", "query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate",
                    "description": "基于检索证据、对话记忆或二者结合生成/重新生成答案。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": decision_schema,
                            "query": {"type": "string"},
                            "source_mode": {
                                "type": "string",
                                "enum": ["retrieval", "memory", "mixed"],
                                "description": (
                                    "答案来源。retrieval=必须使用检索 chunks；"
                                    "memory=只使用对话记忆，适合询问刚才/上一轮内容；"
                                    "mixed=结合检索 chunks 和对话记忆。默认 retrieval。"
                                ),
                            },
                            "retrieval_id": {
                                "type": "string",
                                "description": "retrieval id，或使用 'all' 表示基于目前所有检索证据生成。",
                            },
                            "retrieval_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "instruction": {"type": "string"},
                        },
                        "required": ["decision", "query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish",
                    "description": (
                        "提交最终结构化结果并结束 agent 循环。必须使用 generate 产生的 answer_id；本工具不接收答案正文。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": decision_schema,
                            "answer_id": {"type": "string"},
                            "query": {"type": "string"},
                            "grounded": {"type": "boolean"},
                            "completeness": {"type": "string", "enum": ["yes", "no"]},
                            "if_multi_turn": {"type": "boolean"},
                            "rationale": {"type": "string"},
                            "next_focus": {"type": "string"},
                            "retrieval_id": {"type": "string"},
                            "retrieval_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "decision",
                            "query",
                            "grounded",
                            "completeness",
                            "if_multi_turn",
                            "rationale",
                        ],
                    },
                },
            },
        ]
        tools.extend(tool.to_openai_tool() for tool in self.mcp_tools)
        return tools

    def _decision_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "description": "ReAct-lite 决策摘要。只写可观测短摘要，不要写完整思维链。",
            "properties": {
                "stage": {
                    "type": "string",
                    "description": "当前阶段，如 evidence_gap_check / answer_generation / final_evaluation。",
                },
                "rationale": {
                    "type": "string",
                    "description": "一句话说明为什么现在调用这个工具。",
                },
                "expected_gain": {
                    "type": "string",
                    "description": "一句话说明这一步预期补足什么。",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "对这一步有用性的估计。",
                },
            },
            "required": ["stage", "rationale", "expected_gain", "confidence"],
        }

    def _extract_agent_decision(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        assistant_content: str,
    ) -> dict[str, Any]:
        decision = arguments.get("decision")
        if isinstance(decision, dict):
            return {
                "tool": tool_name,
                "stage": str(decision.get("stage") or ""),
                "rationale": str(decision.get("rationale") or ""),
                "expected_gain": str(decision.get("expected_gain") or ""),
                "confidence": self._safe_float(decision.get("confidence")),
                "assistant_summary": assistant_content.strip() if assistant_content else "",
            }
        return {
            "tool": tool_name,
            "stage": "",
            "rationale": assistant_content.strip() if assistant_content else "",
            "expected_gain": "",
            "confidence": 0.0,
            "assistant_summary": assistant_content.strip() if assistant_content else "",
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _execute_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        history: list[str],
        default_topk: int,
    ) -> dict[str, Any]:
        if tool_name == "retrieve":
            return self._execute_retrieve(arguments, default_topk)
        if tool_name == "generate":
            return self._execute_generate(arguments, history)
        if tool_name == "finish":
            return self._execute_finish(arguments)
        if self.mcp_runtime and self.mcp_runtime.has_tool(tool_name):
            return self._execute_mcp_tool(tool_name, arguments)
        return {"error": f"Unknown tool: {tool_name}"}

    def _execute_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.mcp_runtime:
            return {"error": "MCP runtime is not configured.", "mcp_tool": tool_name}
        result = self.mcp_runtime.call_tool(tool_name, arguments)
        retrieval = result.pop("retrieval", None)
        if retrieval:
            retrieval_id = self.runtime.put_retrieval(retrieval)
            hits = list(retrieval.get("fused_hits", []))
            result["retrieval_id"] = retrieval_id
            result["retrieved_chunk_ids"] = [hit.chunk_id for hit in hits]
            result["used_queries"] = list(retrieval.get("used_queries", []))
            result["retrieval_budget"] = dict(retrieval.get("retrieval_budget", {}))
        return result

    def _execute_retrieve(self, arguments: dict[str, Any], default_topk: int) -> dict[str, Any]:
        selected_kb_ids = self._effective_kb_ids(arguments.get("kb_ids"))
        retrievals_by_kb: dict[str, dict[str, Any]] = {}
        for kb_id in selected_kb_ids:
            orchestrator = self.orchestrators_by_kb.get(kb_id)
            if not orchestrator:
                continue
            retrievals_by_kb[kb_id] = orchestrator.retrieve_tool.run(
                query=str(arguments.get("query") or ""),
                plans=arguments.get("plans"),
                retrieval_modes=arguments.get("retrieval_modes"),
                topk=int(arguments.get("topk") or default_topk),
            )
        retrieval = self._merge_kb_retrievals(
            retrievals_by_kb,
            query=str(arguments.get("query") or ""),
        )
        retrieval_id = self.runtime.put_retrieval(retrieval)
        hits = list(retrieval["fused_hits"])
        return {
            "retrieval_id": retrieval_id,
            "selected_kb_ids": selected_kb_ids,
            "query": str(retrieval["query"]),
            "used_queries": list(retrieval["used_queries"]),
            "retrieved_chunk_ids": [hit.chunk_id for hit in hits],
            "retrieval_budget": dict(retrieval.get("retrieval_budget", {})),
            "chunks": RetrievalTool.summarize_hits(hits),
            "retriever_runs": [
                self._serialize_retriever_run(item)
                for item in retrieval["retrieval_results"]
            ],
        }

    def _execute_generate(self, arguments: dict[str, Any], history: list[str]) -> dict[str, Any]:
        source_mode = self._normalize_source_mode(arguments.get("source_mode"))
        retrieval_id_arg = arguments.get("retrieval_id")
        retrieval_ids_arg = arguments.get("retrieval_ids")
        retrievals = (
            []
            if source_mode == "memory"
            else self._resolve_generation_retrievals(retrieval_id_arg, retrieval_ids_arg)
        )
        if not retrievals and source_mode == "retrieval":
            has_explicit_retrieval_ref = bool(retrieval_id_arg or retrieval_ids_arg)
            has_last_retrieval = bool(self.runtime.globals.get("sys.last_retrieval_id"))
            query_for_memory_check = str(
                arguments.get("query") or self.runtime.globals.get("sys.query") or ""
            )
            if (
                not has_explicit_retrieval_ref
                and not has_last_retrieval
                and history
                and self._looks_like_memory_query(query_for_memory_check)
            ):
                source_mode = "memory"
            else:
                return {
                    "error": f"Unknown retrieval reference: {retrieval_id_arg or retrieval_ids_arg}",
                    "available_retrieval_ids": list(self.runtime.retrievals),
                    "source_mode": source_mode,
                }
        if not retrievals and source_mode == "mixed":
            source_mode = "memory"
        if not retrievals and source_mode not in {"memory", "mixed"}:
            return {
                "error": f"Unknown retrieval reference: {retrieval_id_arg or retrieval_ids_arg}",
                "available_retrieval_ids": list(self.runtime.retrievals),
                "source_mode": source_mode,
            }
        retrieval_ids = [str(item["retrieval_id"]) for item in retrievals]
        hits = self._merge_retrieval_hits(retrievals)

        query = str(
            arguments.get("query")
            or (retrievals[-1]["query"] if retrievals else self.runtime.globals.get("sys.query"))
            or ""
        )
        generation = self.orchestrator.generate_once(
            query=query,
            hits=hits,
            history=history,
            instruction=(
                str(arguments.get("instruction")).strip()
                if arguments.get("instruction") is not None
                else None
            ),
            source_mode=source_mode,
        )
        response_text = generation.answer
        retrieved_chunk_ids = [hit.chunk_id for hit in hits]
        used_queries = self._merge_used_queries(retrievals)
        active_retrieval_id = (
            retrieval_ids[-1]
            if len(retrieval_ids) == 1
            else ("merged" if retrieval_ids else None)
        )
        is_valid = bool(response_text.strip()) and not generation.used_fallback
        answer_id = self.runtime.put_answer(
            {
                "content": response_text,
                "query": query,
                "source_mode": source_mode,
                "retrieval_id": active_retrieval_id,
                "retrieval_ids": retrieval_ids,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "used_queries": used_queries,
                "llm_context": generation.llm_context,
                "llm_raw_output": generation.raw_output,
                "used_fallback": generation.used_fallback,
                "error": generation.error,
                "is_valid": is_valid,
            },
            mark_latest=is_valid,
        )
        return {
            "answer_id": answer_id,
            "response": response_text,
            "query": query,
            "source_mode": source_mode,
            "retrieval_id": active_retrieval_id,
            "retrieval_ids": retrieval_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "used_queries": used_queries,
            "grounded_hint": bool(hits) or (source_mode == "memory" and bool(history)),
            "is_valid": is_valid,
            "used_fallback": generation.used_fallback,
            "error": generation.error,
            "llm_context": generation.llm_context,
            "llm_raw_output": generation.raw_output,
        }

    @staticmethod
    def _normalize_source_mode(value: Any) -> str:
        source_mode = str(value or "retrieval").strip().lower()
        if source_mode not in {"retrieval", "memory", "mixed"}:
            return "retrieval"
        return source_mode

    @staticmethod
    def _looks_like_memory_query(query: str) -> bool:
        markers = (
            "刚才",
            "上一轮",
            "上轮",
            "之前",
            "前面",
            "历史",
            "对话",
            "回答是啥",
            "回答是什么",
            "我问了什么",
            "说了什么",
            "总结一下刚",
        )
        return any(marker in query for marker in markers)

    def _resolve_generation_retrievals(
        self,
        retrieval_id_arg: Any,
        retrieval_ids_arg: Any,
    ) -> list[dict[str, Any]]:
        if str(retrieval_id_arg or "").lower() == "all":
            return self.runtime.get_retrievals()
        if isinstance(retrieval_ids_arg, list) and retrieval_ids_arg:
            return self.runtime.get_retrievals([str(item) for item in retrieval_ids_arg])
        retrieval_id = str(
            retrieval_id_arg
            or self.runtime.globals.get("sys.last_retrieval_id")
            or ""
        )
        retrieval = self.runtime.get_retrieval(retrieval_id)
        return [retrieval] if retrieval is not None else []

    def _merge_retrieval_hits(self, retrievals: list[dict[str, Any]]) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for retrieval in retrievals:
            for hit in retrieval.get("fused_hits", []):
                if hit.chunk_id in seen:
                    continue
                seen.add(hit.chunk_id)
                merged.append(hit)
        return merged

    def _merge_used_queries(self, retrievals: list[dict[str, Any]]) -> list[str]:
        used_queries: list[str] = []
        seen: set[str] = set()
        for retrieval in retrievals:
            for query in retrieval.get("used_queries", []):
                query_text = str(query)
                if query_text in seen:
                    continue
                seen.add(query_text)
                used_queries.append(query_text)
        return used_queries

    def _execute_finish(self, arguments: dict[str, Any]) -> dict[str, Any]:
        answer_id = str(
            arguments.get("answer_id")
            or self.runtime.globals.get("sys.last_valid_answer_id")
            or self.runtime.globals.get("sys.last_answer_id")
            or ""
        )
        answer = self.runtime.get_answer(answer_id)
        if not answer or not answer.get("is_valid"):
            answer = self.runtime.get_last_valid_answer()
            if answer:
                answer_id = str(self.runtime.globals.get("sys.last_valid_answer_id") or "")
        answer_source_mode = str((answer or {}).get("source_mode") or "retrieval")
        retrieval_ids_arg = arguments.get("retrieval_ids")
        retrieval_id = str(
            arguments.get("retrieval_id")
            or (answer or {}).get("retrieval_id")
            or (None if answer_source_mode == "memory" else self.runtime.globals.get("sys.last_retrieval_id"))
            or ""
        )
        retrievals = (
            []
            if answer_source_mode == "memory"
            else self._resolve_generation_retrievals(retrieval_id, retrieval_ids_arg)
        )
        response_text = str(
            (answer or {}).get("content")
            or self.runtime.last_valid_answer_content()
            or "No valid generated answer is available. Please call generate successfully before finish."
        )
        retrieved_chunk_ids = list((answer or {}).get("retrieved_chunk_ids") or [])
        used_queries = list((answer or {}).get("used_queries") or [])
        if not retrieved_chunk_ids and retrievals:
            retrieved_chunk_ids = [hit.chunk_id for hit in self._merge_retrieval_hits(retrievals)]
        if not used_queries and retrievals:
            used_queries = self._merge_used_queries(retrievals)

        payload = self.orchestrator.finish_tool.finish(
            response=response_text,
            query=str(arguments.get("query") or self.runtime.globals.get("sys.query") or ""),
            grounded=bool(arguments.get("grounded")),
            completeness=str(arguments.get("completeness") or "no"),
            if_multi_turn=bool(arguments.get("if_multi_turn")),
            rationale=str(arguments.get("rationale") or ""),
            next_focus=(
                str(arguments.get("next_focus")).strip()
                if arguments.get("next_focus") is not None
                else None
            ),
            retrieved_chunk_ids=retrieved_chunk_ids,
            used_queries=used_queries,
            tool_rounds=len(self.traces) + 1,
        )
        payload["answer_id"] = answer_id or None
        payload["retrieval_id"] = retrieval_id or None
        payload["retrieval_ids"] = list(retrieval_ids_arg or (answer or {}).get("retrieval_ids") or [])
        payload["source_mode"] = answer_source_mode
        return payload

    def _summarize_result(self, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "retrieve":
            return {
                "retrieval_id": result.get("retrieval_id"),
                "selected_kb_ids": result.get("selected_kb_ids", []),
                "retrieved_chunk_ids": result.get("retrieved_chunk_ids", []),
                "used_queries": result.get("used_queries", []),
                "retrieval_budget": result.get("retrieval_budget", {}),
            }
        if tool_name == "generate":
            return {
                "answer_id": result.get("answer_id"),
                "source_mode": result.get("source_mode"),
                "retrieval_id": result.get("retrieval_id"),
                "retrieval_ids": result.get("retrieval_ids", []),
                "retrieved_chunk_ids": result.get("retrieved_chunk_ids", []),
                "used_queries": result.get("used_queries", []),
                "is_valid": result.get("is_valid", False),
                "used_fallback": result.get("used_fallback", False),
                "error": result.get("error"),
                "llm_context": result.get("llm_context"),
                "llm_raw_output": result.get("llm_raw_output"),
                "response_preview": str(result.get("response") or "")[:220],
            }
        if tool_name == "finish":
            return {
                "answer_id": result.get("answer_id"),
                "source_mode": result.get("source_mode"),
                "retrieval_id": result.get("retrieval_id"),
                "grounded": result.get("grounded"),
                "completeness": result.get("completeness"),
                "if_multi_turn": result.get("if_multi_turn"),
            }
        return result

    def _kb_catalog_prompt(self) -> str:
        items = [
            KnowledgeBaseCatalogItem(
                kb_id=str(item.get("kb_id") or ""),
                name=str(item.get("name") or ""),
                description=str(item.get("description") or ""),
                language=str(item.get("language") or "zh"),
                retrieval_modes=[str(mode) for mode in item.get("retrieval_modes", [])],
                domain=str(item.get("domain") or ""),
                tags=[str(tag) for tag in item.get("tags", [])],
                examples=[str(example) for example in item.get("examples", [])],
                priority=int(item.get("priority") or 0),
            )
            for item in self.available_kbs
        ]
        return build_catalog_prompt(items)

    def _effective_kb_ids(self, requested: Any) -> list[str]:
        requested_ids = (
            [str(item) for item in requested if str(item) in self.available_kb_ids]
            if isinstance(requested, list)
            else []
        )
        effective = requested_ids or [
            kb_id for kb_id in self.default_kb_ids
            if kb_id in self.orchestrators_by_kb
        ]
        if not effective:
            effective = list(self.orchestrators_by_kb)[:1]
        return effective

    def _merge_kb_retrievals(
        self,
        retrievals_by_kb: dict[str, dict[str, Any]],
        *,
        query: str,
    ) -> dict[str, Any]:
        fused_hits: list[Any] = []
        seen: set[str] = set()
        retrieval_results: list[dict[str, Any]] = []
        used_queries: list[str] = []
        retrieval_budget: dict[str, Any] = {"selected_kb_ids": list(retrievals_by_kb)}
        for kb_id, retrieval in retrievals_by_kb.items():
            for result in retrieval.get("retrieval_results", []):
                retrieval_results.append({"kb_id": kb_id, "result": result})
            for used_query in retrieval.get("used_queries", []):
                if used_query not in used_queries:
                    used_queries.append(used_query)
            for hit in retrieval.get("fused_hits", []):
                hit.metadata["kb_id"] = kb_id
                key = f"{kb_id}:{hit.chunk_id}"
                if key in seen:
                    continue
                seen.add(key)
                fused_hits.append(hit)
            retrieval_budget[kb_id] = dict(retrieval.get("retrieval_budget", {}))
        return {
            "query": query,
            "used_queries": used_queries or [query],
            "retrieval_results": retrieval_results,
            "fused_hits": fused_hits,
            "retrieval_budget": retrieval_budget,
        }

    @staticmethod
    def _serialize_retriever_run(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            kb_id = item.get("kb_id")
            result = item.get("result")
        else:
            kb_id = None
            result = item
        return {
            "kb_id": kb_id,
            "retriever": result.retriever,
            "query": result.query,
            "chunk_ids": [hit.chunk_id for hit in result.hits],
        }

    def _fallback_run(
        self,
        query: str,
        *,
        history: list[str],
        topk: int,
    ) -> StructuredRAGResponse:
        retrieval = self.orchestrator.retrieve_once(
            query,
            retrieval_modes=self.orchestrator.default_retrieval_modes(query),
            topk=topk,
        )
        hits = list(retrieval["fused_hits"])
        generation = self.orchestrator.generate_once(
            query=query,
            hits=hits,
            history=history,
        )
        response_text = generation.answer
        judge = self.orchestrator.answer_judge.judge(
            query=query,
            response=response_text,
            hits=hits,
            subqueries=[],
            round_index=1,
            max_rounds=1,
        )
        payload = self.orchestrator.finish_tool.finish(
            response=response_text,
            query=query,
            grounded=judge.grounded,
            completeness=judge.completeness,
            if_multi_turn=False,
            rationale="Fallback path without tool-calling LLM.",
            next_focus=None,
            retrieved_chunk_ids=[hit.chunk_id for hit in hits],
            used_queries=list(retrieval["used_queries"]),
            tool_rounds=1,
        )
        return StructuredRAGResponse(
            response=str(payload["response"]),
            query=str(payload["query"]),
            grounded=bool(payload["grounded"]),
            retrieved_chunk_ids=list(payload["retrieved_chunk_ids"]),
            completeness=str(payload["completeness"]),
            if_multi_turn=bool(payload["if_multi_turn"]),
            rationale=str(payload["rationale"]),
            next_focus=None,
            relevance_score=judge.relevance_score,
            support_score=judge.support_score,
            round=1,
            tool_rounds=1,
            used_queries=list(payload["used_queries"]),
            traces=[
                {
                    "round": 1,
                    "tool": "generate",
                    "arguments": {"query": query, "fallback_path": True},
                    "result": {
                        "used_fallback": generation.used_fallback,
                        "error": generation.error,
                        "llm_context": generation.llm_context,
                        "llm_raw_output": generation.raw_output,
                        "response_preview": response_text[:220],
                    },
                }
            ],
        )

    def _auto_finish(
        self,
        *,
        query: str,
        reason: str,
        response_text: str,
        tool_rounds: int,
    ) -> StructuredRAGResponse:
        answer = self.runtime.get_last_valid_answer()
        response_text = response_text or str((answer or {}).get("content") or "")
        retrieval_ids = list((answer or {}).get("retrieval_ids") or [])
        retrievals = self.runtime.get_retrievals(retrieval_ids) if retrieval_ids else []
        if not retrievals:
            retrieval = self.runtime.get_retrieval()
            retrievals = [retrieval] if retrieval is not None else []
        hits = self._merge_retrieval_hits(retrievals)
        used_queries = list((answer or {}).get("used_queries") or [])
        if not used_queries:
            used_queries = self._merge_used_queries(retrievals)
        judge = self.orchestrator.answer_judge.judge(
            query=query,
            response=response_text,
            hits=hits,
            subqueries=[],
            round_index=tool_rounds,
            max_rounds=tool_rounds,
        )
        payload = self.orchestrator.finish_tool.finish(
            response=response_text or "Agent stopped before producing an answer.",
            query=query,
            grounded=judge.grounded,
            completeness=judge.completeness,
            if_multi_turn=judge.completeness != "yes",
            rationale=reason,
            next_focus=(
                "Supplement retrieval or regenerate the answer."
                if judge.completeness != "yes"
                else None
            ),
            retrieved_chunk_ids=[hit.chunk_id for hit in hits],
            used_queries=used_queries,
            tool_rounds=tool_rounds,
        )
        return self._to_structured_response(
            payload,
            round_index=tool_rounds,
            tool_rounds=tool_rounds,
            relevance_score=judge.relevance_score,
            support_score=judge.support_score,
        )

    def _to_structured_response(
        self,
        payload: dict[str, Any],
        *,
        round_index: int,
        tool_rounds: int,
        relevance_score: float = 0.0,
        support_score: float = 0.0,
    ) -> StructuredRAGResponse:
        return StructuredRAGResponse(
            response=str(payload["response"]),
            query=str(payload["query"]),
            grounded=bool(payload["grounded"]),
            retrieved_chunk_ids=list(payload["retrieved_chunk_ids"]),
            completeness=str(payload["completeness"]),
            if_multi_turn=bool(payload["if_multi_turn"]),
            rationale=str(payload.get("rationale") or ""),
            next_focus=(
                str(payload["next_focus"])
                if payload.get("next_focus") is not None
                else None
            ),
            relevance_score=relevance_score,
            support_score=support_score,
            round=round_index,
            tool_rounds=tool_rounds,
            used_queries=list(payload.get("used_queries", [])),
            traces=list(self.traces),
        )
