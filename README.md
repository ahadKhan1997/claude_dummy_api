# Claude Gemma Local AI - FastAPI & LangGraph Backend

Production-grade Python backend service powering a Claude-style local personal AI assistant using **FastAPI**, **LangGraph**, **LangChain**, and local **Ollama Gemma 4**.

---

## 🌟 Feature Highlights

- **LangGraph Checkpoint History**: Uses `AsyncSqliteSaver` to automatically persist session checkpoints, states, and chat histories in SQLite without manual SQL overhead.
- **Local Ollama Integration**: Powered by `ChatOllama` (`langchain-ollama`) pointing directly to local Gemma 4 models on loopback `127.0.0.1:11434`.
- **Zero-Disk Ephemeral File Processing**: Processes image Base64 payloads and text file attachments in-memory during stream generation without clogging local hard drives.
- **Server-Sent Events (SSE) Streaming**: Streams token output token-by-token directly using LangGraph `astream_events` version 2 API.
- **Network Resilience**: Bypasses system proxies and VPN interfaces (`NO_PROXY`) to guarantee local socket resolution on Windows/macOS/Linux.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn
- **Orchestration**: [LangGraph](https://www.langchain.com/langgraph) + [LangChain](https://www.langchain.com/)
- **LLM Provider**: [langchain-ollama](https://github.com/langchain-ai/langchain-folder) / Ollama Gemma 4
- **Persistence**: `langgraph-checkpoint-sqlite` + `aiosqlite`
- **Validation**: Pydantic v2

---

## 📂 Project Architecture

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application, CORS, lifespan, & REST/SSE endpoints
│   └── agent_graph.py   # LangGraph state graph & ChatOllama LLM setup
├── data/
│   └── database.db      # SQLite checkpoint database (auto-created at runtime)
├── requirements.txt     # Python dependencies
└── run.py               # Server startup entry point
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python**: 3.10, 3.11, 3.12, or 3.13
- **Ollama**: Installed and running locally (`ollama run gemma4`)

### 2. Environment Setup

Create and activate a virtual environment inside `backend/`:

```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Ollama Service

Ensure Ollama is running and accessible:

```bash
ollama run gemma4
```

Verify that `http://127.0.0.1:11434` returns `"Ollama is running"`.

### 5. Run the Backend Server

```bash
python run.py
```

The API server will run at `http://127.0.0.1:8000`.

---

## 🔌 API Reference

### Sessions & History

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/sessions` | Fetch all historical chat sessions with title previews |
| `POST` | `/api/sessions` | Create a new chat session thread |
| `DELETE` | `/api/sessions/{session_id}` | Delete a chat thread and purge its SQLite checkpoints |
| `GET` | `/api/sessions/{session_id}/messages` | Retrieve message history for a specific thread |

### Real-Time Streaming

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat/stream` | Stream model tokens (SSE) for a prompt & optional attachment |

#### Stream Request Payload Example

```json
{
  "session_id": "session-4cecfdd2",
  "prompt": "Analyze this image and summary code.",
  "attachment": {
    "type": "image",
    "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "name": "screenshot.png"
  }
}
```

---

## 🧹 Maintenance & Disk Cleanup

All uploaded attachments are processed strictly in RAM during execution and discarded upon stream completion.

To purge stored conversation histories manually:
1. **Per-Session**: Delete sessions via the UI sidebar or `DELETE /api/sessions/{session_id}`.
2. **Full Reset**: Stop `python run.py` and delete `backend/data/database.db`. A new database will be automatically initialized when restarting.