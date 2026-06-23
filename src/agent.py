"""
agent.py
--------
Manual tool-calling agent loop — no langchain.agents imports at all.

WHY manual loop instead of AgentExecutor?
  format_to_tool_messages and ToolsAgentOutputParser move between
  langchain sub-packages across versions, causing import errors.

  This implementation uses ONLY:
    - langchain_mistralai   (ChatMistralAI, bind_tools)
    - langchain_core.messages (HumanMessage, AIMessage, ToolMessage, SystemMessage)
  Both are stable across all langchain 0.2+ versions.

How it works:
  1. Build messages list: [SystemMessage] + chat_history + [HumanMessage]
  2. Call LLM with bound tools
  3. If response has tool_calls -> execute each tool, append ToolMessage, loop
  4. If response has NO tool_calls -> that's the final answer, return it
"""

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from src.tools import make_tools

LLM_MODEL = "mistral-large-latest"   # or "mistral-small-latest" for cheaper calls
MAX_ITERATIONS = 6

SYSTEM_PROMPT = """You are DocuMind, an expert document intelligence assistant.

You have access to a knowledge base of documents the user has uploaded.
Use your tools to find accurate, grounded answers — never guess or make up information.

Your tools:
- search_docs: Find relevant passages from any document. Use this first.
- summarize_doc: Summarize a specific document by its exact filename.
- compare_docs: Compare how multiple documents address the same topic.

Rules:
1. Always cite sources as [Source: filename, Page X] when referencing content.
2. If search finds nothing relevant, say so — do not invent an answer.
3. For topics spanning multiple documents, use compare_docs.
4. Be concise and structured. Use bullet points for lists of findings.
"""


class DocuMindAgent:
    """
    Self-contained agent with a simple invoke() interface.
    Compatible with the AgentExecutor.invoke() call in app.py.
    """

    def __init__(self):
        self.tools = make_tools()
        self.tool_map = {t.name: t for t in self.tools}
        llm = ChatMistralAI(model=LLM_MODEL, temperature=0)
        self.llm = llm.bind_tools(self.tools)   # attach tool schemas once

    def invoke(self, inputs: dict) -> dict:
        """
        Run the agent loop.

        inputs = {
            "input":        str             — the user's message
            "chat_history": list[BaseMessage] — prior turns (can be empty)
        }
        returns {"output": str}
        """
        user_input   = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        # Build the full message list for this turn
        messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + chat_history
            + [HumanMessage(content=user_input)]
        )

        for step in range(MAX_ITERATIONS):

            response = self.llm.invoke(messages)
            messages.append(response)

            # No tool calls = Mistral produced a final text answer
            if not getattr(response, "tool_calls", None):
                return {"output": response.content}

            # Execute every tool the LLM requested
            for call in response.tool_calls:
                name    = call["name"]
                args    = call["args"]
                call_id = call["id"]

                print(f"  [tool] {name}({args})")   # visible in terminal during demo

                if name in self.tool_map:
                    result = self.tool_map[name].invoke(args)
                else:
                    result = f"Unknown tool '{name}'."

                # Return the tool result so Mistral can reason over it
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=call_id)
                )

        return {"output": "Reached the step limit. Try a more specific question."}


def create_agent() -> DocuMindAgent:
    """Return a ready-to-use agent. Called once per user message in app.py."""
    return DocuMindAgent()


def format_chat_history(messages: list[dict]) -> list:
    """
    Convert Streamlit session_state dicts
    -> LangChain message objects for the agent.
    """
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history