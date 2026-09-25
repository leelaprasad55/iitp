"""
LangGraph support assistant.

Nodes:
    classify_intent -> retrieve_and_answer | direct_answer
"""

from __future__ import annotations

import json
import os
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

try:
    from .embeddings import retrieve_top_k
    from .models import AssistantResponse
    from .prompts import SCHEMA_REPAIR_INSTRUCTION, SYSTEM_PROMPT, build_user_prompt
except ImportError:
    from embeddings import retrieve_top_k
    from models import AssistantResponse
    from prompts import SCHEMA_REPAIR_INSTRUCTION, SYSTEM_PROMPT, build_user_prompt

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "track",
    "cancel",
    "gift card",
    "giftcard",
    "support hours",
    "support hour",
]

DIRECT_ANSWER_TEXT = "I can only answer questions about Zepto policies right now."
MOCK_CONFIDENCE = 1.0


class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


def mock_llm_enabled() -> bool:
    """MOCK_LLM is on when unset or set to 1/true/yes."""
    value = os.getenv("MOCK_LLM")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _keyword_intent(query: str) -> str:
    lowered = query.lower()
    is_policy = any(keyword in lowered for keyword in POLICY_KEYWORDS)
    return "policy_question" if is_policy else "general_question"


def classify_intent(state: GraphState) -> GraphState:
    """Route case-insensitively by policy keywords in every generation mode."""
    state["intent"] = _keyword_intent(state["query"])
    return state


def retrieve_and_answer(state: GraphState) -> GraphState:
    """Embed query, search ChromaDB for top 3 chunks, then generate an answer."""
    hits = retrieve_top_k(state["query"], k=3)
    sources = []
    for hit in hits:
        if hit["source"] not in sources:
            sources.append(hit["source"])

    if mock_llm_enabled():
        top = hits[0] if hits else None
        if top is None:
            snippet = "no matching policy chunk was found."
        else:
            snippet = " ".join(str(top["text"]).split())[:200]
        answer = f"Based on the retrieved context: {snippet}"
        response = AssistantResponse(
            answer=answer,
            sources=sources,
            confidence=MOCK_CONFIDENCE,
        )
    else:
        response = _real_llm_answer(state["query"], hits)

    state["answer"] = response.answer
    state["sources"] = response.sources
    state["confidence"] = response.confidence
    return state


def direct_answer(state: GraphState) -> GraphState:
    """Deterministic non-policy response in MOCK_LLM mode."""
    if mock_llm_enabled():
        response = AssistantResponse(
            answer=DIRECT_ANSWER_TEXT,
            sources=[],
            confidence=MOCK_CONFIDENCE,
        )
    else:
        response = _real_llm_direct(state["query"])
    state["answer"] = response.answer
    state["sources"] = response.sources
    state["confidence"] = response.confidence
    return state


def route_intent(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    if state.get("intent") == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


def _chat_completion(messages: list[dict]) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("No API key configured for MOCK_LLM=0")

    import httpx

    if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")
    else:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    payload = {"model": model, "messages": messages, "temperature": 0}
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def _real_llm_classify(query: str) -> str | None:
    try:
        content = _chat_completion(
            [
                {
                    "role": "system",
                    "content": "Classify the user question as policy_question or general_question. Reply with one of those two labels only.",
                },
                {"role": "user", "content": query},
            ]
        )
        label = content.strip().lower().replace(" ", "_")
        if "policy" in label:
            return "policy_question"
        if "general" in label:
            return "general_question"
    except Exception:
        return None
    return None


def _real_llm_direct(query: str) -> AssistantResponse:
    try:
        content = _chat_completion(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Customer question: {query}\n"
                        "No retrieved policy context is available. "
                        "Respond with JSON only using the required schema."
                    ),
                },
            ]
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Customer question: {query}"},
        ]
        return _parse_with_retries(content, messages)
    except Exception:
        return AssistantResponse(
            answer=DIRECT_ANSWER_TEXT,
            sources=[],
            confidence=0.5,
        )


def _parse_with_retries(content: str, messages: list[dict]) -> AssistantResponse:
    last_error = None
    text = content
    for attempt in range(3):
        try:
            data = json.loads(text)
            return AssistantResponse.model_validate(data)
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                break
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": SCHEMA_REPAIR_INSTRUCTION},
            ]
            text = _chat_completion(messages)
    raise RuntimeError(f"LLM response failed Pydantic validation after retries: {last_error}")


def _real_llm_answer(query: str, hits: list[dict]) -> AssistantResponse:
    """
    Optional OpenAI-compatible / Groq chat completion.

    Reads OPENAI_API_KEY or GROQ_API_KEY from env. Never hardcodes keys.
    Retries up to 2 extra times if Pydantic validation fails.
    """
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")):
        top = hits[0] if hits else None
        snippet = " ".join(str(top["text"]).split())[:200] if top else "no context"
        return AssistantResponse(
            answer=f"Based on the retrieved context: {snippet}",
            sources=[h["source"] for h in hits],
            confidence=0.5,
        )

    context_blocks = [f"{h['source']}: {h['text']}" for h in hits]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, context_blocks)},
    ]
    last_error = None
    content = None
    for attempt in range(3):
        if attempt > 0:
            messages.append({"role": "user", "content": SCHEMA_REPAIR_INSTRUCTION})
        try:
            content = _chat_completion(messages)
            messages.append({"role": "assistant", "content": content})
            data = json.loads(content)
            return AssistantResponse.model_validate(data)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"LLM response failed Pydantic validation after retries: {last_error}")


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


APP_GRAPH = build_graph()


def ask(query: str) -> AssistantResponse:
    result = APP_GRAPH.invoke(
        {
            "query": query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
        }
    )
    return AssistantResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
    )


if __name__ == "__main__":
    try:
        from .embeddings import ingest_to_chroma
    except ImportError:
        from embeddings import ingest_to_chroma

    ingest_to_chroma()
    print(ask("What is the return policy?").model_dump())
    print(ask("What is the capital of India?").model_dump())
