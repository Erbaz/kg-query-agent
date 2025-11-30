from typing import Optional
from llama_index.core.llms import LLM
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine, inspect
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding


def create_sqlalchemy_engine(config: dict):
    """
    config example:
    {
        "db_type": "mysql",
        "db_user": "user,
        "db_password": "root,
        "db_host": "localhost",
        "db_port": 3306,
        "db_name": "mydb",
    }
    """
    dialect = config["db_type"]
    db_host = (
        "localhost"
        if config["db_host"] == "host.docker.internal"
        else config["db_host"]
    )
    if dialect == "postgresql":
        url = (
            f"postgresql+psycopg2://{config['db_user']}:{config['db_password']}"
            f"@{db_host}:{config['db_port']}/{config['db_name']}"
        )
    elif dialect == "mysql":
        url = (
            f"mysql+pymysql://{config['db_user']}:{config['db_password']}"
            f"@{db_host}:{config['db_port']}/{config['db_name']}"
        )
    else:
        raise ValueError(f"Unsupported dialect: {dialect}")

    return create_engine(url)


class SQLRetrievalEngine:

    def __init__(
        self,
        llm: Gemini | OpenAI | Ollama,
        embed_model_name: str,
        connection_config: dict,
        api_key: Optional[str] = None,
        is_ollama=False,
        tables=None,
    ):
        # 1. Build SQLAlchemy engine from config
        engine = create_sqlalchemy_engine(connection_config)

        # 2. Wrap with LlamaIndex SQLDatabase
        sql_database = SQLDatabase(engine=engine)

        if tables is None:
            inspector = inspect(engine)
            tables = inspector.get_table_names()

        # ---- 1. If Ollama backend ----
        if isinstance(llm, Ollama) or is_ollama:
            self.embed_model = OllamaEmbedding(model_name=embed_model_name)

        # ---- 2. GEMINI ----
        if isinstance(llm, Gemini):
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set")
            self.embed_model = GeminiEmbedding(
                model_name=embed_model_name, api_key=api_key
            )

        # ---- 3. OPENAI ----
        if isinstance(llm, OpenAI):
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self.embed_model = OpenAIEmbedding(
                model_name=embed_model_name, api_key=api_key
            )

        # 3. Build retriever
        print("------ NLSQL Retriever -----")
        print(
            f"llm: {type(llm).__name__} || embedder: {type(self.embed_model).__name__}"
        )
        self.retriever = NLSQLRetriever(
            llm=llm,
            embed_model=self.embed_model,
            sql_database=sql_database,
            tables=tables or None,
            return_raw=False,
        )

        # 4. Build query engine
        self.query_engine = RetrieverQueryEngine.from_args(self.retriever, llm=llm)

        self.db_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="sql_db_tool",
            description="Query the SQL database to retrieve data from it.",
        )
