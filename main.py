from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from src.routes.query import router as query_router

app = FastAPI(title="kg-query-agent")

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(query_router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "kg-query-agent"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)