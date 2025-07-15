# iris_backend.py

import uvicorn
import logging
import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# --- NEW: CORS Middleware ---
from fastapi.middleware.cors import CORSMiddleware

# --- Agent & DB Imports ---
try:
    from agents.iris import IrisOrchestrator
    from agents import db_ltm # <-- Import the db_ltm module
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError as e:
    print(f"Fatal Error: Could not import required agent modules. Details: {e}")
    exit(1)


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State Management ---
shared_resources = {}

# --- FastAPI Lifespan Manager (Unchanged) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    async with AsyncSqliteSaver.from_conn_string("iris_api_checkpoint.sqlite") as memory_checkpointer:
        logger.info("AsyncSqliteSaver checkpointer context entered.")
        iris_agent = IrisOrchestrator(checkpointer=memory_checkpointer)
        shared_resources["iris_agent"] = iris_agent
        logger.info("IrisOrchestrator agent has been successfully initialized.")
        yield
    logger.info("Application shutting down...")
    shared_resources.clear()

# --- API Data Models ---
class LoginRequest(BaseModel):
    email: str
    name: str | None = None # Name is optional

class LoginResponse(BaseModel):
    message: str
    user_id: int
    email: str
    name: str

class ChatRequest(BaseModel):
    user_identifier: str
    user_input: str
    thread_id: str

# No ChatResponse model needed for streaming, as we send raw text chunks.

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="IRIS - AI Financial Analyst API",
    description="An API to interact with the IRIS agent for financial analysis.",
    version="1.1.0", # Version bump!
    lifespan=lifespan
)

# --- NEW: Add CORS Middleware ---
# This allows our Next.js app (e.g., from localhost:3000) to call our API (e.g., on localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # The origin of your Next.js app
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- NEW: Login Endpoint ---
@app.post("/login", response_model=LoginResponse)
async def handle_login(request: LoginRequest):
    """
    Authenticates a user. If the user doesn't exist, they are created.
    This mimics the Streamlit login flow.
    """
    logger.info(f"Login attempt for email: {request.email}")
    try:
        # Use the existing function from your db_ltm module
        user_data = await asyncio.to_thread(
            db_ltm.authenticate_and_get_user,
            email=request.email,
            name=request.name
        )
        if user_data:
            logger.info(f"Login successful for user ID: {user_data['id']}")
            return LoginResponse(
                message=f"Welcome, {user_data['name']}!",
                user_id=user_data['id'],
                email=user_data['email'],
                name=user_data['name']
            )
        else:
            raise HTTPException(status_code=401, detail="Authentication failed. Please check your details.")
    except Exception as e:
        logger.error(f"Error during login for {request.email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during login.")

# --- NEW: Chat History Endpoint ---
@app.get("/users/{user_id}/sessions")
async def get_chat_history(user_id: int):
    """Fetches all chat session metadata for a given user."""
    try:
        sessions = await asyncio.to_thread(db_ltm.get_user_sessions, user_id)
        # We need to format the datetime objects to be JSON serializable
        for session in sessions:
            session['started_at'] = session['started_at'].isoformat()
        return sessions
    except Exception as e:
        logger.error(f"Error fetching history for user_id {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not fetch chat history.")


# --- MODIFIED: Chat Endpoint for Streaming ---
@app.post("/chat")
async def handle_chat_streaming(request: ChatRequest):
    """
    Processes a user's query and STREAMS the response back.
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
        try:
            # astream_events is perfect for this. We listen for chunks from the final agent.
            # Make sure you have langgraph v0.1.0 or higher.
            async for event in iris_agent.app.astream_events(payload, config, version="v1"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if content := chunk.content:
                        # Yield the content chunk directly to the client
                        logger.debug(f"Streaming chunk: {content}")
                        yield content
        except Exception as e:
            logger.error(f"Error during agent stream for {request.thread_id}: {e}", exc_info=True)
            # Yield a final error message if something goes wrong during the stream
            yield f"\n\nAn error occurred: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")


# --- Root Endpoint (Unchanged) ---
@app.get("/")
def read_root():
    return {"status": "IRIS API is running"}


# --- Uvicorn Runner (Unchanged) ---
if __name__ == "__main__":
    uvicorn.run("iris_backend:app", host="0.0.0.0", port=8000, reload=True)