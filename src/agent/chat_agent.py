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
            api_key=api_key,
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
            token_limit=6000,
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
                return Ollama(
                    model=model_name,
                    base_url="http://host.docker.internal:11434",
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
        messages = self.chat_memory.get_all()
        self.current_token_count = self.chat_memory._token_count_for_messages(messages)
        print("-" * 12)
        print(f"Current tokens used in conversation: {self.current_token_count}")
        print("-" * 12)
        return response_text

    async def chat_stream_generator(self, user_input: str):
        try:
            handler = self.agent.run(user_input, ctx=self.chatCtx, memory=self.chat_memory)
            buffer = ""
            
            async for ev in handler.stream_events(expose_internal=True):
                if isinstance(ev, InternalDispatchEvent):
                    print(type(ev), ev)
                if isinstance(ev, ToolCallResult):
                    tool_call_result = f"Call {ev.tool_name} with args {ev.tool_kwargs}\nReturned: {ev.tool_output}"
                    print(tool_call_result)
                    yield f"data: {tool_call_result}\n\n"
                elif isinstance(ev, AgentStream):
                    delta = ev.delta or ""
                    if not delta.strip():
                        continue
                    
                    buffer += delta
                    print(delta, end="", flush=True)
                    
                    # Split on whitespace boundaries
                    while True:
                        # Check for newline first
                        if '\n' in buffer:
                            lines = buffer.split('\n', 1)
                            if lines[0]:  # Only yield if there's content
                                yield f"data: {lines[0]}\n\n"
                            buffer = lines[1]
                        # Then check for space
                        elif ' ' in buffer:
                            parts = buffer.split(' ', 1)
                            if parts[0]:  # Only yield if there's content
                                yield f"data: {parts[0]}\n\n"
                            buffer = parts[1]
                        else:
                            # No whitespace boundary found, keep buffering
                            break
            
            # Flush any remaining content in buffer at the end
            if buffer:
                yield f"data: {buffer}\n\n"

            messages = self.chat_memory.get_all()
            self.current_token_count = self.chat_memory._token_count_for_messages(messages)
            
            yield f"data: Tokens Used: {self.current_token_count}\n\n\n"

        except Exception as e:
            # Flush any remaining content on error too
            if buffer:
                yield f"data: {buffer}\n\n"
            raise e

    def get_chat_history(self) -> list[dict]:
        messages = self.chat_memory.get_all()
        return [
            msg.model_dump() for msg in messages
        ]  # Convert each ChatMessage to dict

    # def summarize_chat_history(self):
    #     # this method will summarize the chat history and place it into the system prompt
    #     # the chat history list will be cleared

    #     if self.current_token_count < 3000:
    #         return

    #     messages = self.chat_memory.get_all()
    #     messages_text = ""
    #     for message in messages:
    #         messages_text += f"{message.role}: {message.content}\n"

    #     response = self.llm.complete(
    #         f"""
    #         Summarize the following chat history and convert into the following template:
    #         User: ...\nAssistant: Called tool [tool name] with args [tool args] , Observed correct / incorrect response, ...[Called other tool and observed response]... and Answered: ...

    #         If there is any data in Answer, then please preserve it. But you may summarize the text

    #         Chat History:
    #         {messages_text}
    #         """
    #     )

    #     self.agent.formatter.system_header = REACT_SYSTEM_PROMPT.replace(
    #         "{summarized_chat_history}", response.text.strip(), 1
    #     )

    #     self.chat_memory.reset()
