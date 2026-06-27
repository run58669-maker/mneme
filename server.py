"""Mneme API server — FastAPI wrapper for the MemoryAgent."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mneme import MemoryAgent

DB = os.environ.get("MNEME_DB", "mneme.db")
_agent: MemoryAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = MemoryAgent(DB)
    yield


app = FastAPI(title="Mneme", version="0.2.0", lifespan=lifespan)


class ChatReq(BaseModel):
    message: str
    temperature: float = 0.7


class ChatResp(BaseModel):
    reply: str


class SleepResp(BaseModel):
    gist_id: int | None
    consolidated: int | None
    gist: str | None


@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    reply = _agent.chat(req.message, temperature=req.temperature)
    return ChatResp(reply=reply)


@app.post("/sleep", response_model=SleepResp)
def sleep():
    result = _agent.sleep()
    return SleepResp(**result)


@app.post("/new_session")
def new_session():
    _agent.new_session()
    return {"ok": True}


@app.get("/stats")
def stats():
    return _agent.stats()


@app.get("/health")
def health():
    return {"status": "ok", "db": DB}
