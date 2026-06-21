import asyncio
import json
import os
import queue
import threading
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from tools import search_catalog, get_order_details, initiate_return
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Retail Assistant",
    layout="wide",
)


# ── Agent  ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are VentureGPT, a friendly and professional customer support agent for Venture Dive.

Your responsibilities:
- Help customers find products that match their needs using the product catalog.
- Look up order status and provide accurate, up-to-date order details.
- Assist customers with return requests, following company policy (returns are only available for delivered orders).

Guidelines:
- Always greet customers warmly and address them by name if provided.
- Be concise, empathetic, and solution-focused.
- If a customer asks about something outside your scope (e.g. billing disputes, technical issues beyond returns), \
politely let them know and suggest they contact Venture Dive's support team directly at support@venturedive.com.
- Never fabricate product information or order data — use the available tools to retrieve real information.
- When initiating a return, always confirm the order details with the customer before proceeding."""


@st.cache_resource
def get_agent():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it to backend/.env and restart.")
        st.stop()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    return create_agent(
        llm,
        [search_catalog, get_order_details, initiate_return],
        system_prompt=SYSTEM_PROMPT,
    )


agent = get_agent()


def _iter_events(agent, agent_input):
    """Bridge astream_events(v2) which is async-only, into a sync generator.
    Doing this because of a weird v3 vs v2 error in the agent streaming code, and decided to just go around it.
    Not worth it for this assessment."""
    q = queue.Queue()  # q equals q dot q. saying this out loud is funny.

    async def _produce():
        try:
            async for event in agent.astream_events(
                {"messages": agent_input}, version="v2"
            ):
                q.put(("ok", event))
        except Exception as exc:
            q.put(("err", exc))
        finally:
            q.put(("done", None))

    threading.Thread(target=asyncio.run, args=(_produce(),), daemon=True).start()

    while True:
        kind, payload = q.get()
        if kind == "done":
            return
        if kind == "err":
            raise payload
        yield payload


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    # Each entry: {role, content, tool_calls?}
    st.session_state.messages = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Controls")
    st.caption(f"{len(st.session_state.messages)} messages in history")
    st.divider()

    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if st.session_state.messages:
        export_payload = {
            "exported_at": datetime.now().isoformat(),
            "message_count": len(st.session_state.messages),
            "messages": st.session_state.messages,
        }
        st.download_button(
            label="Export chat as JSON",
            data=json.dumps(export_payload, indent=2, default=str),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("Export available after first message.")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Smart Retail Assistant")

# ── Render existing chat history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_calls"):
            with st.expander(f"{len(msg['tool_calls'])} tool call(s)", expanded=False):
                for tc in msg["tool_calls"]:
                    st.json(tc)

# ── Chat input & streaming ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask the agent…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Pass the full conversation history so the agent has context. This can be swapped by an in-memory checkpointer on the agentic side, but I digress.
    agent_input = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        tool_slot = st.empty()  # live-updating tool call expander
        text_slot = st.empty()  # streaming text

        response_text = ""
        tool_calls = []

        try:
            for event in _iter_events(agent, agent_input):
                event_type = event["event"]

                if event_type == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = chunk.content
                    if isinstance(content, str) and content:
                        response_text += content
                        text_slot.markdown(response_text + "▌")
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                response_text += block.get("text", "")
                        if response_text:
                            text_slot.markdown(response_text + "▌")

                elif event_type == "on_tool_start":
                    tool_calls.append(
                        {
                            "tool": event["name"],
                            "args": event["data"].get("input", {}),
                        }
                    )
                    with tool_slot.container():
                        with st.expander(
                            f"{len(tool_calls)} tool call(s)", expanded=True
                        ):
                            for tc in tool_calls:
                                st.json(tc)

        except Exception as exc:
            st.error(f"Agent error: {exc}")
            response_text = response_text or f"*(error: {exc})*"

        text_slot.markdown(response_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        }
    )
