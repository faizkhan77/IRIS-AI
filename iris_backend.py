# iris_backend.py

import uvicorn
import logging
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware

try:
    from agents.iris import IrisOrchestrator
    from agents import db_ltm
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError as e:
    print(f"Fatal Error: Could not import required agent modules. Details: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

shared_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    async with AsyncSqliteSaver.from_conn_string("iris_api_checkpoint.sqlite") as memory_checkpointer:
        iris_agent = IrisOrchestrator(checkpointer=memory_checkpointer)
        shared_resources["iris_agent"] = iris_agent
        logger.info("IrisOrchestrator agent has been successfully initialized.")
        yield
    logger.info("Application shutting down...")
    shared_resources.clear()

class LoginRequest(BaseModel):
    email: str
    name: str | None = None

class LoginResponse(BaseModel):
    message: str
    user_id: int
    email: str
    name: str

class ChatRequest(BaseModel):
    user_identifier: str
    user_input: str
    thread_id: str

app = FastAPI(
    title="IRIS - AI Financial Analyst API",
    description="An API to interact with the IRIS agent for financial analysis.",
    version="1.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/login", response_model=LoginResponse)
async def handle_login(request: LoginRequest):
    logger.info(f"Login attempt for email: {request.email}")
    try:
        user_data = await asyncio.to_thread(db_ltm.authenticate_and_get_user, email=request.email, name=request.name)
        if user_data:
            logger.info(f"Login successful for user ID: {user_data['id']}")
            return LoginResponse(
                message=f"Welcome, {user_data['name']}!",
                user_id=user_data['id'],
                email=user_data['email'],
                name=user_data['name']
            )
        else:
            raise HTTPException(status_code=401, detail="Authentication failed.")
    except Exception as e:
        logger.error(f"Error during login for {request.email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during login.")

@app.get("/users/{user_id}/sessions")
async def get_chat_history(user_id: int):
    try:
        sessions = await asyncio.to_thread(db_ltm.get_user_sessions, user_id)
        for session in sessions:
            if 'started_at' in session and session['started_at']:
                session['started_at'] = session['started_at'].isoformat()
        return sessions
    except Exception as e:
        logger.error(f"Error fetching history for user_id {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not fetch chat history.")

@app.get("/sessions/{thread_id}/messages")
async def get_session_messages(thread_id: str):
    try:
        messages = await asyncio.to_thread(db_ltm.get_messages_by_thread_id, thread_id)
        for msg in messages:
            if 'timestamp' in msg and msg['timestamp']:
                msg['timestamp'] = msg['timestamp'].isoformat()
        return messages
    except Exception as e:
        logger.error(f"Error fetching messages for thread_id {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not fetch messages for session.")

# --- THIS IS THE DEFINITIVE, CORRECTED STREAMING ENDPOINT ---
@app.post("/chat")
async def handle_chat_streaming(request: ChatRequest):
    """
    This endpoint now correctly handles the agent's full output.
    It runs the agent to completion, captures the final response,
    and then streams that final response back to the client.
    This guarantees only the desired output is sent.
    """
    iris_agent = shared_resources.get("iris_agent")
    if not iris_agent:
        raise HTTPException(status_code=503, detail="IRIS Agent is not available.")

    payload = {
        "user_identifier": request.user_identifier,
        "user_input": request.user_input,
        "thread_id": request.thread_id
    }
    config = {"configurable": {"thread_id": request.thread_id}}

    async def stream_generator():
        final_response_content = None
        try:
            # We will use ainvoke to run the graph to completion.
            # astream_events is for debugging or complex multi-stream UI,
            # but ainvoke is simpler for getting a final result.
            final_state = await iris_agent.ainvoke(payload, config)
            
            # The final response is in the state after the graph has finished.
            final_response_content = final_state.get("final_response")

            if not final_response_content:
                final_response_content = "Sorry, I was unable to generate a response."

            # Now, we stream the final response character by character for a typing effect.
            # This gives the UI the streaming effect it wants, but with the correct, final data.
            for char in final_response_content:
                yield char
                await asyncio.sleep(0.01) # Small delay for a natural typing feel

        except Exception as e:
            logger.error(f"Error during agent invocation for {request.thread_id}: {e}", exc_info=True)
            yield "\n\nAn error occurred while processing your request."

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/")
def read_root():
    return {"status": "IRIS API is running"}

if __name__ == "__main__":
    uvicorn.run("iris_backend:app", host="0.0.0.0", port=8000, reload=True)