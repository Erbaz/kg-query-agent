from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agent.chat_agent import ChatAgent
from src.cache.chat_cache import chat_cache
import uuid
import json

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class CreatChatRequest(BaseModel):
    model: str
    memgraph_url: str
    memgraph_user: str
    memgraph_password: str
    db_name: str
    db_user: str
    db_password: str
    db_host: str
    db_port: str | int
    db_type: str
    embed_model: str
    is_ollama: bool | None = None
    api_key: str | None = None


@router.post("/chat")
async def create_chat(req: CreatChatRequest):
    try:
        chat_id = str(uuid.uuid4())
        chat_agent = ChatAgent(
            uuid=chat_id,
            llm_model=req.model,
            embed_model=req.embed_model,
            api_key=req.api_key,
            kg_url=req.memgraph_url,
            kg_user=req.memgraph_user,
            kg_password=req.memgraph_password,
            db_name=req.db_name,
            db_type=req.db_type,
            db_host=req.db_host,
            db_port=req.db_port,
            db_password=req.db_password,
            db_user=req.db_user,
            is_ollama=req.is_ollama,
        )

        print("--- start migration ---")
        chat_agent.migration_to_memgraph()

        if not chat_cache.store(chat_id, chat_agent):
            raise HTTPException(status_code=500, detail="Unable to create chat session")

        return {"chat_id": chat_id}
    except Exception as e:
        print("Error: ", e)
        raise HTTPException(
            status_code=500, detail="Unable to create chat session: " + str(e)
        )


@router.post("/chat/{chat_id}", response_model=QueryResponse)
async def query_endpoint(chat_id: str, req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    # Convert string id to UUID for dictionary lookup
    try:
        uuid.UUID(chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat_id format")

    chat_agent = chat_cache.get(chat_id)
    if chat_agent is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Use the agent to get a response
    response = await chat_agent.chat_stream(req.question)

    return QueryResponse(answer=response)


@router.get("/chat/{chat_id}")
async def get_chat_history(chat_id: str):
    chat_agent = chat_cache.get(chat_id)
    if chat_agent is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return chat_agent.get_chat_history()
