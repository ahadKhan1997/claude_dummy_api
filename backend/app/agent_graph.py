from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_INSTRUCTION = SystemMessage(
    content="You are a helpful AI assistant powered by Gemma 4. Answer concisely, accurately, and assist with any queries."
)

# 2. Node Definition
def create_agent_node(llm: BaseChatModel):
    async def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = state["messages"]
        full_messages = [SYSTEM_INSTRUCTION] + list(messages)
        response = await llm.ainvoke(full_messages)
        return {"messages": [response]}

    return call_model

# 3. Graph Builder
def build_agent_graph(llm: BaseChatModel | None = None, checkpointer=None):
    if llm is None:
        # Initialize local Ollama Gemma 4 directly via langchain_ollama
        llm = ChatOllama(
            model="gemma4",
            base_url="http://127.0.0.1:11434",
            temperature=0.7,
        )

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", create_agent_node(llm))
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)