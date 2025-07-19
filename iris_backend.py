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

@app.post("/chat")
async def handle_chat_streaming(request: ChatRequest):
    """
    This endpoint runs the agent to completion, then intelligently streams the
    final response. It supports both simple text responses and structured
    GenUI responses with a chart component.
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

    # This is the key difference: we now use Server-Sent Events (SSE)
    # which is the standard for sending mixed data types like this.
    async def sse_stream_generator():
        try:
            # 1. Run the agent to completion. This logic is IDENTICAL to your preferred code.
            final_state = await iris_agent.ainvoke(payload, config)
            final_response = final_state.get("final_response")

            # 2. Check the type of the final response to decide how to stream.
            if isinstance(final_response, dict):
                # CASE A: It's a structured GenUI response (e.g., with a chart)
                
                # First, send the UI component data as a single, complete event.
                ui_components = final_response.get("ui_components", [])
                if ui_components:
                    yield f"event: ui_component\ndata: {json.dumps(ui_components)}\n\n"
                
                # Then, stream the text part character by character.
                text_response = final_response.get("text_response", "...")
                for char in text_response:
                    chunk_data = json.dumps({"chunk": char})
                    yield f"event: text_chunk\ndata: {chunk_data}\n\n"

            elif isinstance(final_response, str):
                # CASE B: It's a simple string response (e.g., "hi", or a formatted answer)
                
                # We just stream the text part, character by character.
                # No ui_component event is sent.
                for char in final_response:
                    chunk_data = json.dumps({"chunk": char})
                    yield f"event: text_chunk\ndata: {chunk_data}\n\n"
            
            else:
                # Fallback for an empty or unexpected response type
                fallback_text = "Sorry, I was unable to generate a response."
                for char in fallback_text:
                    chunk_data = json.dumps({"chunk": char})
                    yield f"event: text_chunk\ndata: {chunk_data}\n\n"

        except Exception as e:
            logger.error(f"Error during agent invocation for {request.thread_id}: {e}", exc_info=True)
            error_message = json.dumps({"error": "An error occurred while processing your request."})
            yield f"event: error\ndata: {error_message}\n\n"
        finally:
            # Always send an 'end' event so the client knows when to stop listening.
            yield "event: end\ndata: {}\n\n"

    # The media type MUST be "text/event-stream" for SSE to work.
    return StreamingResponse(sse_stream_generator(), media_type="text/event-stream")
@app.get("/")
def read_root():
    return {"status": "IRIS API is running"}

if __name__ == "__main__":
    uvicorn.run("iris_backend:app", host="0.0.0.0", port=8000, reload=True)