"""
Shared chat utilities: OpenAI chat with tools (voice) and RAG-style process_query (query + history -> answer, sources, intent).
"""
import json
import logging
import os

from backend.voice_prompt import build_voice_system_message

logger = logging.getLogger(__name__)

# --- OpenAI chat with tools (used by /api/chat for voice) ---

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use when the user asks about recent events, facts, or anything you need to look up. Provide a clear search query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to run (e.g. 'weather Tokyo today', 'latest news about X')"},
            },
            "required": ["query"],
        },
    },
}


def tavily_search(query: str) -> str:
    """Run a Tavily web search and return a string summary for the model. Uses TAVILY_API_KEY."""
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return "Web search is not configured (TAVILY_API_KEY not set)."
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="basic", max_results=5, include_answer=True)
        parts = []
        if response.get("answer"):
            parts.append(response["answer"])
        results = response.get("results") or []
        if results:
            parts.append("Sources:")
            for r in results[:5]:
                title = r.get("title") or "No title"
                url = r.get("url") or ""
                content = (r.get("content") or "")[:500]
                parts.append(f"- {title} ({url}): {content}")
        return "\n\n".join(parts) if parts else "No results found."
    except Exception as e:
        logger.warning("Tavily search error: %s", e)
        return f"Web search failed: {e!s}"


END_CONVERSATION_TOOL_DESCRIPTIONS = {
    "ja": "Call this ONLY when the user unambiguously says they want to end (e.g. さようなら、ありがとうございました、以上です、もう結構です). Speech recognition can mishear: if the phrase is short or could be a misheard question, do NOT call this—respond normally. When calling: reply with a brief thank you in Japanese, then call this tool.",
    "en": "Call this ONLY when the user unambiguously says they want to end (e.g. goodbye, that's all for now, I'm done thanks, no more questions). Speech recognition can mishear: if the phrase is short or could be a misheard question, do NOT call this—respond normally. When calling: reply with a brief thank you in English, then call this tool.",
}


def _end_conversation_tool(lang: str):
    desc = END_CONVERSATION_TOOL_DESCRIPTIONS.get(lang) or END_CONVERSATION_TOOL_DESCRIPTIONS["en"]
    return {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": desc,
            "parameters": {"type": "object", "properties": {}},
        },
    }


END_CONVERSATION_DEFAULT = {
    "ja": "お話しできてありがとうございました。またね。",
    "en": "Thank you for talking with me. Goodbye!",
}


def run_chat_with_tools(client, messages: list, lang: str) -> tuple[str, bool]:
    """
    Run chat completion with tools; handle tool_calls in a loop (max 5 rounds).
    Returns (final_content, end_conversation).
    """
    tools = [_end_conversation_tool(lang), WEB_SEARCH_TOOL]
    end_conversation = False
    max_rounds = 5
    content = ""
    for _ in range(max_rounds):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=2048,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            if end_conversation and not content:
                content = END_CONVERSATION_DEFAULT.get(lang) or END_CONVERSATION_DEFAULT["en"]
            return content, end_conversation
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        assistant_msg["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ]
        messages.append(assistant_msg)
        for tc in tool_calls:
            name = getattr(tc.function, "name", None) or ""
            args_str = getattr(tc.function, "arguments", None) or "{}"
            if name == "end_conversation":
                end_conversation = True
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "ok"})
            elif name == "web_search":
                try:
                    args = json.loads(args_str)
                    query = args.get("query") or ""
                except Exception:
                    query = args_str
                result = tavily_search(query)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Unknown tool."})
    if end_conversation and not content:
        content = END_CONVERSATION_DEFAULT.get(lang) or END_CONVERSATION_DEFAULT["en"]
    return content or "I'm sorry, I hit a limit. Please try again.", end_conversation


def run_openai_chat(client, messages: list, lang: str, verbosity: str | None = None) -> tuple[str, bool]:
    """
    Build system message, prepend to messages, and run chat with tools.
    Returns (assistant_content, end_conversation).
    """
    system_content = build_voice_system_message(lang, verbosity=verbosity)
    full_messages = [{"role": "system", "content": system_content}] + list(messages)
    return run_chat_with_tools(client, full_messages, lang)


# --- RAG-style chat (query + history -> answer, sources, intent) ---

def classify_intent(query: str, history: list[dict]) -> str:
    """
    Classify user intent: SEARCH (knowledge base search) or META (e.g. summary/about).
    Uses a simple OpenAI call when OPENAI_API_KEY is set; otherwise defaults to SEARCH.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return "SEARCH"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Based on the user message, reply with exactly one word: SEARCH or META. "
            "Use META only if they are asking for a summary of the knowledge base, what's in it, or meta-information. "
            "Use SEARCH for any question about content, facts, or lookups.\nUser: " + query
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        if "META" in text:
            return "META"
        return "SEARCH"
    except Exception as e:
        logger.warning("classify_intent error: %s", e)
        return "SEARCH"


def hybrid_search(query: str, top_k: int = 3) -> list:
    """
    RAG search: embed query and run vector/semantic search.
    When Azure Search is configured (AZURE_SEARCH_* env), uses it; otherwise returns [].
    """
    # Optional: Azure AI Search. Requires azure-search-documents and OPENAI for embeddings.
    endpoint = (os.getenv("AZURE_SEARCH_ENDPOINT") or "").strip()
    index_name = (os.getenv("AZURE_SEARCH_INDEX") or "").strip()
    embed_deployment = (os.getenv("OPENAI_EMBED_DEPLOYMENT") or "").strip()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not (endpoint and index_name and embed_deployment and api_key):
        return []
    try:
        from openai import OpenAI
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.models import VectorizedQuery, QueryType

        openai_client = OpenAI(api_key=api_key)
        emb_resp = openai_client.embeddings.create(input=query, model=embed_deployment)
        emb = emb_resp.data[0].embedding
        search_key = (os.getenv("AZURE_SEARCH_KEY") or "").strip()
        if not search_key:
            return []
        credential = AzureKeyCredential(search_key)
        search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
        vq = VectorizedQuery(vector=emb, k_nearest_neighbors=10, fields="content_vector")
        results = list(
            search_client.search(
                search_text=query,
                vector_queries=[vq],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "my-semantic-config"),
                top=top_k,
            )
        )
        return [dict(r) for r in results]
    except ImportError:
        logger.warning("Azure Search not available (install azure-search-documents).")
        return []
    except Exception as e:
        logger.warning("hybrid_search error: %s", e)
        return []


def get_knowledge_summary() -> list:
    """
    Return knowledge-base summary / meta info for META intent.
    Override or extend when you have a real summary source.
    """
    return [{"content": "Knowledge base summary is not configured. Add get_knowledge_summary data or use SEARCH.", "title": "Summary"}]


def generate_answer(query: str, data: list, history: list[dict], intent: str) -> str:
    """
    Generate an answer using OpenAI from query, retrieved data, and chat history.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return "OpenAI is not configured (OPENAI_API_KEY not set)."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        context_parts = []
        for i, d in enumerate(data):
            if isinstance(d, dict):
                content = d.get("content") or d.get("content_vector") or str(d)
                if isinstance(content, list):
                    content = "(vector)"
                context_parts.append(f"[{i + 1}] {content}")
            else:
                context_parts.append(str(d))
        context = "\n\n".join(context_parts) if context_parts else "No retrieved context."
        system = (
            "You are a helpful assistant. Answer based on the following context when available. "
            "If the context is empty or irrelevant, say so briefly. Be concise."
        )
        messages = [{"role": "system", "content": f"{system}\n\nContext:\n{context}"}]
        for h in history:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "") or ""})
        messages.append({"role": "user", "content": query})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("generate_answer error")
        return f"Sorry, I couldn't generate an answer: {e!s}"


def process_query(query: str, history: list[dict]) -> dict:
    """
    RAG-style pipeline: classify intent -> retrieve (hybrid_search or get_knowledge_summary) -> generate answer.
    Returns {"answer": str, "sources": list[str], "intent": str}.
    """
    intent = classify_intent(query, history)
    data = []
    if intent == "SEARCH":
        data = hybrid_search(query)
    elif intent == "META":
        data = get_knowledge_summary()
    ans = generate_answer(query, data, history, intent)
    sources = []
    for d in (data or []):
        if isinstance(d, dict):
            name = d.get("metadata_storage_name") or d.get("title") or "Unknown"
            if name not in sources:
                sources.append(name)
    return {"answer": ans, "sources": list(sources), "intent": intent}
