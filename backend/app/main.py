import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

# Force local loopback proxy bypass for Windows
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent_graph import build_agent_graph

DB_PATH = "data/database.db"
compiled_graph = None

os.makedirs("data", exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global compiled_graph
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        await checkpointer.setup()
        compiled_graph = build_agent_graph(checkpointer=checkpointer)
        yield

app = FastAPI(title="Claude Gemma Local AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schemas
class AttachmentPayload(BaseModel):
    type: str  # 'image' or 'text'
    data: str  # Base64 string for images or raw text string
    name: str

class SessionCreate(BaseModel):
    title: Optional[str] = "New Chat"

class StreamChatRequest(BaseModel):
    session_id: str
    prompt: str
    attachment: Optional[AttachmentPayload] = None


# --- Helper to safely extract display text from any message content ---
def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return " ".join(text_parts)
    return str(content)


# --- API Endpoints ---

@app.get("/api/sessions")
async def get_sessions():
    if not os.path.exists(DB_PATH):
        return []

    async with aiosqlite.connect(DB_PATH) as db:
        try:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            )
            if not await cursor.fetchone():
                return []

            cursor = await db.execute(
                "SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
            )
            rows = await cursor.fetchall()

            sessions = []
            for r in rows:
                thread_id = r[0]
                title = None

                if compiled_graph:
                    try:
                        state_snapshot = await compiled_graph.aget_state({"configurable": {"thread_id": thread_id}})
                        if state_snapshot and state_snapshot.values:
                            msgs = state_snapshot.values.get("messages", [])
                            if msgs:
                                raw_text = extract_text_from_content(getattr(msgs[0], "content", ""))
                                if raw_text.strip():
                                    title = raw_text.strip()[:28] + ("..." if len(raw_text.strip()) > 28 else "")
                    except Exception:
                        pass

                if not title:
                    clean_hex = thread_id.replace("session-", "")[:8]
                    title = f"Chat ({clean_hex})"

                sessions.append({"id": thread_id, "title": title})

            return sessions
        except Exception as e:
            print(f"Error reading sessions: {e}")
            return []


@app.post("/api/sessions")
async def create_session(payload: SessionCreate):
    new_id = f"session-{uuid.uuid4().hex[:8]}"
    return {
        "id": new_id, 
        "title": payload.title or "New Chat"
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if os.path.exists(DB_PATH):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
            await db.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
            await db.commit()
    return {"success": True}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    if not compiled_graph:
        return []

    config = {"configurable": {"thread_id": session_id}}
    try:
        state_snapshot = await compiled_graph.aget_state(config)
        messages = state_snapshot.values.get("messages", []) if state_snapshot.values else []

        result = []
        for idx, m in enumerate(messages):
            role = "user" if m.__class__.__name__ == "HumanMessage" else "assistant"
            display_text = extract_text_from_content(m.content)

            result.append({
                "id": f"msg-{idx}",
                "sessionId": session_id,
                "role": role,
                "content": display_text
            })
        return result
    except Exception:
        return []


@app.post("/api/chat/stream")
async def stream_chat(payload: StreamChatRequest):
    if not compiled_graph:
        raise HTTPException(status_code=503, detail="Graph engine not ready")

    config = {"configurable": {"thread_id": payload.session_id}}
    prompt_text = payload.prompt.strip() or "Describe this attachment."

    # Process image attachment into clean text or base64 format
    if payload.attachment and payload.attachment.type == "image":
        raw_b64 = payload.attachment.data
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]

        # Use clean standard message with inline base64 image dict
        user_message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{raw_b64}"
                }
            ]
        )
    elif payload.attachment and payload.attachment.type == "text":
        combined = f"{prompt_text}\n\n--- Attachment: {payload.attachment.name} ---\n{payload.attachment.data}"
        user_message = HumanMessage(content=combined)
    else:
        user_message = HumanMessage(content=prompt_text)

    input_state = {"messages": [user_message]}

    async def event_generator():
        try:
            async for event in compiled_graph.astream_events(input_state, config=config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        yield chunk
        except Exception as e:
            # Yield error string directly so frontend displays error and completes stream
            yield f"\n[Ollama Error: {str(e)}]"

    return StreamingResponse(event_generator(), media_type="text/event-stream")