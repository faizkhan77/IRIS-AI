# /iris-agent-backend/main.py

import os
import sys
import json
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse # Keep for streaming the single final answer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Attempt to import necessary components
try:
    from agents.iris import iris_graph_builder, SupervisorDecision # Assuming SupervisorDecision is defined in iris.py
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver # Use Async version for FastAPI
    from langchain_core.messages import BaseMessage # Assuming BaseMessage is used in IrisState
    print("Successfully imported iris_graph_builder and AsyncSqliteSaver.")
except ImportError as e:
    print(f"Error importing from agents.iris or its dependencies: {e}")
    # Dummy fallbacks for server to run if imports fail
    SupervisorDecision = type("SupervisorDecision", (BaseModel,), {}) 
    BaseMessage = type("BaseMessage", (), {"dict": lambda: {}}) 
    class DummyApp:
        async def astream(self, inputs: Dict[str, Any], config: Dict[str, Any], stream_mode: str):
            print("DummyApp astream called") # Add print for dummy
            yield {"finalize_response_and_log_chat": {"final_response": "Dummy fallback: Agent not fully initialized."}}
            await asyncio.sleep(0.1) # Simulate some async work
    class DummyGraphBuilder:
        def compile(self, checkpointer=None): 
            print("DummyGraphBuilder compile called") # Add print for dummy
            return DummyApp()
    iris_graph_builder = DummyGraphBuilder()
    AsyncSqliteSaver = None 
    print("Using dummy fallbacks for server due to import error.")


# --- Global Variables for App and Checkpointer ---
iris_app_instance_global: Optional[Any] = None
checkpointer_instance_global: Optional[AsyncSqliteSaver] = None
# No need for sqlite_cm_global if using AsyncSqliteSaver.from_conn_string() which returns the instance

IRIS_STM_CHECKPOINT_DB = "iris_stm_checkpoint_async.sqlite" # Use a different DB name for async version if needed

# To store the context manager for proper exit handling
sqlite_cm_global: Optional[Any] = None 



# --- Lifespan Manager for FastAPI ---
@asynccontextmanager
async def lifespan(app_fastapi: FastAPI):
    global iris_app_instance_global, checkpointer_instance_global, sqlite_cm_global
    print("FastAPI app startup: Initializing IRIS agent with Async Checkpointer...")
    
    if AsyncSqliteSaver:
        try:
            # Step 1: Get the async context manager from from_conn_string()
            sqlite_cm_global = AsyncSqliteSaver.from_conn_string(IRIS_STM_CHECKPOINT_DB)
            print("Async context manager for SqliteSaver created.")

            # Step 2: Manually and asynchronously "enter" the context to get the actual checkpointer instance
            # This is what the 'async with ... as ...' statement does behind the scenes.
            checkpointer_instance_global = await sqlite_cm_global.__aenter__()
            print(f"AsyncSqliteSaver instance obtained for '{IRIS_STM_CHECKPOINT_DB}'.")

            # Step 3: Now compile the graph with the valid checkpointer instance
            iris_app_instance_global = iris_graph_builder.compile(checkpointer=checkpointer_instance_global)
            print("IRIS Graph compiled successfully with AsyncSqliteSaver.")

        except Exception as e:
            import traceback
            print(f"Lifespan Error: Failed to initialize/compile IRIS graph with AsyncSqliteSaver: {e}")
            traceback.print_exc()
            iris_app_instance_global = DummyGraphBuilder().compile()
            print("Lifespan: Falling back to dummy agent due to checkpointer/compilation error.")
    else:
        print("Lifespan Warning: AsyncSqliteSaver not available. Agent will run without persistence.")
        iris_app_instance_global = iris_graph_builder.compile() 
        print("IRIS Graph compiled without checkpointer (AsyncSqliteSaver was not imported).")
    
    if iris_app_instance_global is None:
        print("Lifespan CRITICAL: iris_app_instance_global is None after setup attempts!")
        iris_app_instance_global = DummyGraphBuilder().compile()

    # The 'yield' passes control back to FastAPI to run the application
    yield 
    
    # After the application is done (on shutdown), this code runs
    print("FastAPI app shutdown: Cleaning up IRIS agent resources...")
    if sqlite_cm_global and hasattr(sqlite_cm_global, '__aexit__'):
        # Step 4: Manually exit the async context to clean up resources (like closing the DB connection)
        print("Exiting AsyncSqliteSaver context...")
        await sqlite_cm_global.__aexit__(None, None, None)
        print("AsyncSqliteSaver context exited.")
    
    checkpointer_instance_global = None
    iris_app_instance_global = None
    sqlite_cm_global = None
    print("IRIS agent resources cleaned up.")

# --- FastAPI App Setup ---
app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], # Adjust to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API ---
class ChatInvokeRequest(BaseModel):
    userInput: str
    userIdentifier: str
    langgraphThreadId: str # This will be used as thread_id for LangGraph

class FinalAnswerResponse(BaseModel):
    """Pydantic model for the single, final answer."""
    final_answer: str


async def stream_iris_final_answer(request: ChatInvokeRequest):
    """
    Streams only the final answer from the IRIS agent.
    Intermediate steps are processed server-side but not sent to client.
    """
    global iris_app_instance_global
    if iris_app_instance_global is None:
        error_message = "IRIS agent is not initialized. Please check server logs."
        print(f"API Error: {error_message}")
        error_payload = {"error": error_message}
        yield (json.dumps(error_payload) + "\n").encode("utf-8")
        return

    initial_state_for_graph = {
        "user_input": request.userInput,
        "user_identifier": request.userIdentifier,
        "langgraph_thread_id": request.langgraphThreadId,
    }
    config_for_graph = {"configurable": {"thread_id": request.langgraphThreadId}}

    print(f"Invoking IRIS agent for thread_id: {request.langgraphThreadId}, user: {request.userIdentifier}")
    print(f"Initial state for graph: {initial_state_for_graph}")

    final_state: Optional[Dict[str, Any]] = None

    try:
        # Let the stream run to completion, capturing the last event
        async for event_chunk in iris_app_instance_global.astream(
            initial_state_for_graph,
            config=config_for_graph,
            stream_mode="values" 
        ):
            print(f"Server received event_chunk: {str(event_chunk)[:500]}...") # Log intermediate steps
            # The last event_chunk will contain the final state of all keys
            final_state = event_chunk

        # After the loop completes, inspect the final_state
        if final_state and isinstance(final_state, dict) and "final_response" in final_state:
            final_response_content = final_state.get("final_response")
            if final_response_content:
                response_payload = FinalAnswerResponse(final_answer=final_response_content)
                json_payload = response_payload.model_dump_json() + "\n"
                print(f"Server yielding final answer: {json_payload.strip()}")
                yield json_payload.encode("utf-8")
                return # Explicitly return after yielding the answer
            
        # If we reach here, something went wrong, and we didn't get a final_response
        error_message = "Could not retrieve a final answer from the agent's final state."
        print(f"API Error: {error_message}. Final state was: {final_state}")
        error_payload = {"error": error_message}
        yield (json.dumps(error_payload) + "\n").encode("utf-8")

    except Exception as e:
        import traceback
        print(f"Error during IRIS agent stream for thread {request.langgraphThreadId}:")
        traceback.print_exc()
        error_output = {"error": "An unexpected error occurred processing your request.", "details": str(e)}
        yield (json.dumps(error_output) + "\n").encode("utf-8")

@app.post("/chat/invoke", response_model=None) # response_model=None as it's a streaming response
async def chat_invoke_endpoint(request: ChatInvokeRequest):
    """
    Endpoint to interact with the IRIS agent.
    It streams back only the single final answer.
    """
    return StreamingResponse(
        stream_iris_final_answer(request),
        media_type="application/x-ndjson" # Still using ndjson for consistency, even for a single message
    )

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server for IRIS Agent with Uvicorn...")
    # Ensure reload is True for development if you make changes to agents
    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True)