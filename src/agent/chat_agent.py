from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import sql
from workflows.context.state_store import DictState


from llama_index.llms.gemini import Gemini
from llama_index.llms.gemini.base import GEMINI_MODELS
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS as OPENAI_MODELS
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.agent.workflow import ToolCallResult, AgentStream
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.llms.ollama import Ollama

from typing import Optional

from workflows.events import InternalDispatchEvent

from src.agent.sql_retrieval_engine import SQLRetrievalEngine
from src.agent.kg_retrieval_engine import KGRetrievalEngine
from src.runners.sql_mem_migrate import SqlToMemERDMigration, SqlConfig
from src.agent.sys_prompt import REACT_SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()


class ChatAgent:
    def __init__(
        self,
        uuid: str,
        llm_model: str,
        embed_model: str,
        db_type: str,
        db_user: str,
        db_password: str,
        db_host: str,
        db_name: str,
        db_port: str | int,
        db_url: str = None,
        tools=None,
        kg_url: str = None,
        kg_user: str = None,
        kg_password: str = None,
        api_key: Optional[str] = None,
        is_ollama: bool = False,
    ):
        self.llm = self.init_llm(api_key, model_name=llm_model, is_ollama=is_ollama)
        self.db_url = db_url
        self.db_type = db_type
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_name = db_name
        self.kg_url = kg_url
        self.kg_user = kg_user
        self.kg_password = kg_password
        self.kg_engine = KGRetrievalEngine(
            llm=self.llm, url=kg_url, username=kg_user, password=kg_password
        )
        self.sql_engine = SQLRetrievalEngine(
            llm=self.llm,
            embed_model_name=embed_model,
            connection_config={
                "db_type": db_type,
                "db_user": db_user,
                "db_password": db_password,
                "db_host": db_host,
                "db_port": int(db_port),
                "db_name": db_name,
            },
        )

        print("---- Retrievers set successfully ----")

        self.query_engine_tools = (
            [self.kg_engine.graph_tool, self.sql_engine.db_tool]
            if tools is None
            else tools
        )

        self.agent = ReActAgent(llm=self.llm, tools=self.query_engine_tools)
        self.agent.formatter.system_header = REACT_SYSTEM_PROMPT
        chat_store = SimpleChatStore()

        self.chat_memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=chat_store,
            chat_store_key=uuid,
        )

        self.chatCtx = Context[DictState](self.agent)

    @staticmethod
    def init_llm(api_key=None, model_name: Optional[str] = None, is_ollama=False):
        print(f"model name {model_name} | is_ollama {is_ollama}")
        """Return an LLM instance based on the provided model name."""
        try:
            if model_name and is_ollama:
                print("→ USING OLLAMA")
                return Ollama(
                    model=model_name,
                    request_timeout=120.0,
                    context_window=8000,
                )

            else:
                model = model_name or "models/gemini-2.0-flash"
                if model in GEMINI_MODELS:
                    if not api_key:
                        raise ValueError(
                            "GEMINI_API_KEY environment variable is not set."
                        )
                    return Gemini(model=model, api_key=api_key)

                if model in OPENAI_MODELS:
                    if not api_key:
                        raise ValueError(
                            "OPENAI_API_KEY environment variable is not set."
                        )
                    return OpenAI(model=model, api_key=api_key)

                raise ValueError(f"Unsupported model '{model}'.")
        except Exception as error:
            print("An error occurred during model init")
            raise error

    def migration_to_memgraph(self):
        if self.db_type.lower() == "mysql":
            migrator = SqlToMemERDMigration(
                graph_url=self.kg_url,
                graph_password=self.kg_password,
                graph_username=self.kg_user,
                sql_config=SqlConfig(
                    user=self.db_user,
                    password=self.db_password,
                    database=self.db_name,
                    host=self.db_host,
                ),
            )
            try:
                migrator.migrate_erd()
            except Exception as e:
                raise e
            finally:
                del migrator

    async def chat_stream(self, user_input: str) -> str:
        handler = self.agent.run(user_input, ctx=self.chatCtx, memory=self.chat_memory)
        response_text = ""
        async for ev in handler.stream_events(expose_internal=True):
            if isinstance(ev, InternalDispatchEvent):
                print(type(ev), ev)
            if isinstance(ev, ToolCallResult):
                print(
                    f"Call {ev.tool_name} with args {ev.tool_kwargs}\nReturned: {ev.tool_output}"
                )
            elif isinstance(ev, AgentStream):
                delta = ev.delta or ""
                response_text += delta
                print(delta, end="", flush=True)

        await handler
        return response_text
