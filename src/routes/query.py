from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agent.chat_agent import ChatAgent
import uuid
router = APIRouter()

chat_agents_dict = {}


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str    
    
@router.post("/chat")
async def create_chat():
    uuid = uuid.uuid4()
    chat_agent = ChatAgent(uuid=str(uuid))
    chat_agents_dict[uuid] = chat_agent
    return {"chat_id": str(uuid)}

@router.post("/chat/:id", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="question is required")
    
    # Convert string id to UUID for dictionary lookup
    try:
        chat_uuid = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid chat_id format")
    
    # Retrieve the agent from the dictionary
    if chat_uuid not in chat_agents_dict:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    chat_agent = chat_agents_dict[chat_uuid]
    
    # Use the agent to get a response
    response = await chat_agent.chat_stream(req.question)

    return QueryResponse(answer=response)