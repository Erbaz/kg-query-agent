import re

from llama_index.core.indices.property_graph import (
    PGRetriever,
    LLMSynonymRetriever,
    TextToCypherRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from dotenv import load_dotenv

load_dotenv()


class KGRetrievalEngine:
    def __init__(
        self,
        llm,
        sub_retrievers=None,
        url="bolt://localhost:7687",
        username="memgraph",
        password="",
    ):

        self.graph_store = MemgraphPropertyGraphStore(
            url=url, username=username, password=password
        )

        self.llm = llm

        def strip_markdown_fences(query: str) -> str:
            # Remove triple backtick code fences and leading/trailing whitespace
            return re.sub(r"^```(?:\w+)?\s*([\s\S]*?)\s*```$", r"\1", query.strip())

        self.sub_retrievers = (
            [
                LLMSynonymRetriever(self.graph_store, llm=self.llm),
                TextToCypherRetriever(
                    self.graph_store,
                    llm=self.llm,
                    cypher_validator=strip_markdown_fences,
                ),
            ]
            if sub_retrievers is None
            else sub_retrievers
        )

        self.retriever = PGRetriever(sub_retrievers=self.sub_retrievers)
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever,
            llm=self.llm,
        )

        self.graph_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="memgraph_graph_tool",
            description="Query the Memgraph knowledge graph to retrieve information about the ontologies of a Relational Database's ERD / Schema.",
        )
