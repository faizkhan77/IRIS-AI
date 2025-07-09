# /iris-agent-backend/main.py

import os
import sys
import json
import asyncio
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- FIX 1: IMPROVED IMPORT HANDLING TO PROVIDE CLEARER ERRORS ---
try:
    # This assumes 'iris.py' is in the 'agents' subfolder and defines 'iris_graph_builder'.
    from agents.iris import iris_graph_builder
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    print("Successfully imported 'iris_graph_builder' and 'AsyncSqliteSaver'.")
    AGENT_INITIALIZED_SUCCESSFULLY = True
except ImportError as e:
    print(f"FATAL IMPORT ERROR: {e}")
    print("Could not import 'iris_graph_builder' from 'agents.iris'.")
    print("The server will run with a DUMMY agent. Please ensure 'agents/iris.py' exists and contains 'iris_graph_builder'.")
    
    # Define dummy fallbacks so the server can start and show an error
    class DummyApp:
        async def astream(self, *args, **kwargs):
            yield {"error_state": {"final_response": "CRITICAL ERROR: The main IRIS agent failed to load. Please check the server logs."}}
            await asyncio.sleep(0.1)
    class DummyGraphBuilder:
        def compile(self, checkpointer=None):
            return DummyApp()
    iris_graph_builder = DummyGraphBuilder()
    AsyncSqliteSaver = None
    AGENT_INITIALIZED_SUCCESSFULLY = False
# --- END OF FIX ---


# --- Global Variables for App and Checkpointer ---
iris_app_instance_global = None
checkpointer_instance_global = None
sqlite_cm_global = None 

IRIS_STM_CHECKPOINT_DB = "iris_stm_checkpoint_async.sqlite"

# --- Lifespan Manager for FastAPI ---
@asynccontextmanager
async def lifespan(app_fastapi: FastAPI):
    global iris_app_instance_global, checkpointer_instance_global, sqlite_cm_global
    print("FastAPI app startup: Initializing IRIS agent...")
    
    if AGENT_INITIALIZED_SUCCESSFULLY and AsyncSqliteSaver:
        try:
            sqlite_cm_global = AsyncSqliteSaver.from_conn_string(IRIS_STM_CHECKPOINT_DB)
            checkpointer_instance_global = await sqlite_cm_global.__aenter__()
            print(f"AsyncSqliteSaver instance obtained for '{IRIS_STM_CHECKPOINT_DB}'.")
            iris_app_instance_global = iris_graph_builder.compile(checkpointer=checkpointer_instance_global)
            print("IRIS Graph compiled successfully with AsyncSqliteSaver.")
        except Exception as e:
            print(f"Lifespan Error: Failed to initialize/compile IRIS graph: {e}")
            iris_app_instance_global = DummyGraphBuilder().compile()
            print("Lifespan: Falling back to dummy agent due to compilation error.")
    else:
        print("Lifespan Warning: Agent will run without persistence (dummy or no checkpointer).")
        iris_app_instance_global = iris_graph_builder.compile() 
    
    yield 
    
    print("FastAPI app shutdown: Cleaning up resources...")
    if sqlite_cm_global and hasattr(sqlite_cm_global, '__aexit__'):
        await sqlite_cm_global.__aexit__(None, None, None)
        print("AsyncSqliteSaver context exited.")
    
    checkpointer_instance_global = None
    iris_app_instance_global = None
    sqlite_cm_global = None
    print("IRIS agent resources cleaned up.")

# --- FastAPI App Setup ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API ---
class ChatInvokeRequest(BaseModel):
    userInput: str
    userIdentifier: str
    langgraphThreadId: str

class FinalAnswerResponse(BaseModel):
    final_answer: str

async def stream_iris_final_answer(request: ChatInvokeRequest):
    """
    Streams only the final answer from the IRIS agent.
    """
    global iris_app_instance_global
    if iris_app_instance_global is None:
        error_payload = {"error": "IRIS agent is not initialized."}
        yield (json.dumps(error_payload) + "\n").encode("utf-8")
        return

    # This config is for LangGraph's internal mechanics (like checkpointing)
    config_for_graph = {"configurable": {"thread_id": request.langgraphThreadId}}
    
    print(f"Invoking IRIS agent for thread_id: {request.langgraphThreadId}")

    final_state = None
    try:
        # --- THIS IS THE ONE AND ONLY FIX ---
        # The first argument to astream/ainvoke is the dictionary of inputs for your state.
        # It MUST contain all the keys your first node needs.
        initial_inputs = {
            # This key MUST match your state definition
            "chat_history": [HumanMessage(content=request.userInput)],
            
            # These keys are what your initialize_session_node requires
            "user_identifier": request.userIdentifier,
            "langgraph_thread_id": request.langgraphThreadId,

            # --- ADD THIS LINE ---
            # This key is what your supervisor_decide_node requires
            "user_input": request.userInput
        }

        async for event_chunk in iris_app_instance_global.astream(
            initial_inputs,  # Pass the complete dictionary of inputs here
            config=config_for_graph,
            stream_mode="values"
        ):
            print(f"Server received event_chunk: {str(event_chunk)[:500]}...")
            final_state = event_chunk

        # --- The rest of the function for processing the final answer is correct ---
        final_response_content = None
        if final_state and isinstance(final_state, dict):
            if "finalize_response" in final_state and final_state.get("finalize_response"):
                 final_response_content = final_state.get("finalize_response").get("final_response")
            elif "final_response" in final_state:
                 final_response_content = final_state.get("final_response")

        if final_response_content:
            response_payload = {"final_answer": final_response_content}
            json_payload = json.dumps(response_payload) + "\n"
            print(f"Server yielding final answer: {json_payload.strip()}")
            yield json_payload.encode("utf-8")
        else:
            print(f"API Error: Could not extract a final answer. Final state was: {final_state}")
            error_payload = {"error": "The agent finished, but a final answer could not be constructed."}
            yield (json.dumps(error_payload) + "\n").encode("utf-8")

    except Exception as e:
        import traceback
        print(f"Error during IRIS agent stream for thread {request.langgraphThreadId}:")
        traceback.print_exc()
        error_output = {"error": "An unexpected error occurred processing your request.", "details": str(e)}
        yield (json.dumps(error_output) + "\n").encode("utf-8")
@app.post("/chat/invoke")
async def chat_invoke_endpoint(request: ChatInvokeRequest):
    return StreamingResponse(
        stream_iris_final_answer(request),
        media_type="application/x-ndjson"
    )

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server for IRIS Agent with Uvicorn...")
    # --- FIX 3: ENSURE THE PORT IS 8005 TO MATCH THE FRONTEND'S EXPECTATION ---
    uvicorn.run("main:app", host="127.0.0.1", port=8005, reload=True)