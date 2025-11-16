
from llama_index.llms.gemini import Gemini
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.agent.workflow import ToolCallResult, AgentStream
from llama_index.core.storage.chat_store import SimpleChatStore

from src.agent.kg_retrieval_engine import KGRetrievalEngine


import os
from dotenv import load_dotenv
load_dotenv()

class ChatAgent:
    def __init__(self, uuid: str, llm=None, tools=None):
        self.llm = Gemini(
            model="models/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            ) if llm is None else llm
        self.kg_engine = KGRetrievalEngine(llm=self.llm)
        self.query_engine_tools = [self.kg_engine.graph_tool] if tools is None else tools
        self.agent = ReActAgent(llm=self.llm, tools=self.query_engine_tools)
        chat_store = SimpleChatStore()

        self.chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=chat_store,
            chat_store_key=uuid,
        )

        self.chatCtx = Context(self.agent)
        
    
    async def chat_stream(self, user_input: str) -> str:
        handler = self.agent.run(user_input, ctx=self.chatCtx, memory=self.chat_memory)
        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                print(
                    f"Call {ev.tool_name} with args {ev.tool_kwargs}\nReturned: {ev.tool_output}"
                )
            elif isinstance(ev, AgentStream):
                print(ev.delta, end="", flush=True)

        response = await handler
        return response