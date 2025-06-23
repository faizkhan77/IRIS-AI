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
from starlette.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.iris import iris_graph_builder, SupervisorDecision
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langchain_core.messages import BaseMessage
    print("Successfully imported iris_graph_builder and AsyncSqliteSaver.")
except ImportError as e:
    print(f"Error importing from agents.iris or its dependencies: {e}")
    print("Using dummy fallbacks.")
    SupervisorDecision = BaseModel
    BaseMessage = type("BaseMessage", (), {"dict": lambda: {}})
    class DummyApp:
        async def astream(self, inputs: Dict[str, Any], config: Dict[str, Any], stream_mode: str):
            yield {"finalize_response_and_log_chat": {"final_response": "Dummy fallback response: Agent not fully loaded."}}
    class DummyGraphBuilder:
        def compile(self, checkpointer=None): return DummyApp()
    iris_graph_builder = DummyGraphBuilder()
    AsyncSqliteSaver = None


# --- Global Variables for App and Checkpointer ---
iris_app_instance_global: Optional[Any] = None
checkpointer_instance_global: Optional[AsyncSqliteSaver] = None
sqlite_cm_global: Optional[Any] = None

IRIS_STM_CHECKPOINT_DB = "iris_stm_checkpoint.sqlite"

# --- Lifespan Manager for FastAPI ---
@asynccontextmanager
async def lifespan(app_fastapi: FastAPI):
    global iris_app_instance_global, checkpointer_instance_global, sqlite_cm_global
    print("FastAPI app startup: Initializing IRIS agent...")
    if AsyncSqliteSaver:
        try:
            sqlite_cm_global = AsyncSqliteSaver.from_conn_string(IRIS_STM_CHECKPOINT_DB)
            actual_checkpointer = await sqlite_cm_global.__aenter__()
            checkpointer_instance_global = actual_checkpointer
            iris_app_instance_global = iris_graph_builder.compile(checkpointer=checkpointer_instance_global)
            print(f"IRIS Graph compiled successfully with AsyncSqliteSaver using '{IRIS_STM_CHECKPOINT_DB}'.")
        except Exception as e:
            print(f"Lifespan: Failed to compile IRIS graph with AsyncSqliteSaver: {e}")
            if isinstance(iris_graph_builder, DummyGraphBuilder): # Check if it's already the dummy
                 iris_app_instance_global = iris_graph_builder.compile()
            else: # If original failed, create a new dummy
                iris_app_instance_global = DummyGraphBuilder().compile()
                print("Lifespan: Falling back to dummy agent due to compilation error.")
    else:
        print("Lifespan: AsyncSqliteSaver not available. Compiling IRIS graph (potentially dummy) without checkpointer.")
        iris_app_instance_global = iris_graph_builder.compile()

    if iris_app_instance_global is None: # Should not happen if dummy fallback works
        print("Lifespan CRITICAL: iris_app_instance_global could not be compiled, even dummy.")
        # As a last resort, ensure a dummy app is assigned if all else fails
        iris_app_instance_global = DummyGraphBuilder().compile()


    yield

    print("FastAPI app shutdown: Cleaning up resources...")
    if sqlite_cm_global and hasattr(sqlite_cm_global, '__aexit__'):
        await sqlite_cm_global.__aexit__(None, None, None)
        print("Lifespan: AsyncSqliteSaver context exited.")
    checkpointer_instance_global = None
    iris_app_instance_global = None
    sqlite_cm_global = None


# --- FastAPI App Setup ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInvokeRequest(BaseModel):
    userInput: str
    userIdentifier: str
    langgraphThreadId: str

def safe_serialize(data_obj):
    if isinstance(data_obj, SupervisorDecision):
        return data_obj.model_dump()
    # For BaseMessage or similar objects from Langchain
    if hasattr(data_obj, 'dict') and callable(getattr(data_obj, 'dict')) and not isinstance(data_obj, type(dict)):
        try:
            # Attempt a generic dict conversion first
            serialized_dict = data_obj.dict()
            # Recursively serialize contents of the dict
            return {k: safe_serialize(v) for k, v in serialized_dict.items()}
        except Exception:
            # Fallback if .dict() fails or specific structure isn't met
            pass # Will be caught by general str(data_obj) later
    if isinstance(data_obj, dict):
        return {k: safe_serialize(v) for k, v in data_obj.items()}
    if isinstance(data_obj, list):
        return [safe_serialize(item) for item in data_obj]
    if hasattr(data_obj, 'model_dump') and callable(getattr(data_obj, 'model_dump')): # For other Pydantic models
        return data_obj.model_dump()
    try:
        # Check if it's directly JSON serializable
        json.dumps(data_obj)
        return data_obj
    except (TypeError, OverflowError):
        # Fallback to string representation if not directly serializable
        return str(data_obj)

async def stream_iris_agent_responses(request: ChatInvokeRequest):
    global iris_app_instance_global
    if iris_app_instance_global is None:
        error_msg = {"node": "system_error", "output": {"error_message": "IRIS agent is not initialized on the server (lifespan)."} }
        yield (json.dumps(error_msg) + "\n").encode("utf-8")
        return

    # Inputs for this specific invocation. The checkpointer handles loading existing state.
    initial_inputs_for_graph = {
        "user_input": request.userInput,
        "user_identifier": request.userIdentifier,
        "langgraph_thread_id": request.langgraphThreadId,
        # chat_history is typically managed by the checkpointer based on thread_id
    }
    config_for_graph = {"configurable": {"thread_id": request.langgraphThreadId}}

    print(f"[STREAM_IRIS] Streaming for thread_id: {request.langgraphThreadId}, user: {request.userIdentifier}")
    print(f"[STREAM_IRIS] Graph type: {type(iris_app_instance_global)}")
    print(f"[STREAM_IRIS] Initial inputs to graph: {initial_inputs_for_graph}")
    print(f"[STREAM_IRIS] Config for graph: {config_for_graph}")

    try:
        # CRITICAL CHANGE: stream_mode="updates"
        # This mode yields dictionaries where keys are node names that just completed
        # and values are their outputs.
        async for event_chunk in iris_app_instance_global.astream(
            initial_inputs_for_graph,
            config=config_for_graph,
            stream_mode="updates"
        ):
            print(f"[STREAM_IRIS_EVENT_CHUNK] Raw event_chunk (updates mode): {event_chunk}")

            if not isinstance(event_chunk, dict):
                print(f"[STREAM_IRIS_SKIP] Skipping non-dict event_chunk: {type(event_chunk)} - {event_chunk}")
                continue
            
            if not event_chunk:
                print(f"[STREAM_IRIS_SKIP] Skipping empty dict event_chunk.")
                continue

            # In stream_mode="updates", event_chunk is typically a dict like:
            # {'node_that_just_ran': <output_of_that_node>}
            # It should contain only the output of the node(s) that completed in this step.
            for node_name, node_output_data in event_chunk.items():
                print(f"[STREAM_IRIS_NODE] Processing update for node: '{node_name}', Output type: {type(node_output_data)}")
                # print(f"[STREAM_IRIS_NODE_DATA] Output data for '{node_name}': {str(node_output_data)[:500]}...")


                serializable_node_output = safe_serialize(node_output_data)
                payload_to_send = {"node": node_name, "output": serializable_node_output}

                try:
                    json_payload = json.dumps(payload_to_send) + "\n"
                    print(f"[STREAM_IRIS_YIELD] Yielding payload for '{node_name}': {json_payload.strip()}")
                    yield json_payload.encode("utf-8")
                except (TypeError, OverflowError) as e:
                    error_detail_str = f"Serialization error for node '{node_name}': {e}. Output snippet: {str(payload_to_send)[:200]}"
                    print(f"[STREAM_IRIS_ERROR] {error_detail_str}")
                    error_payload_dict = {"node": node_name, "output": {"error": "Serialization issue", "details": error_detail_str}}
                    yield (json.dumps(error_payload_dict) + "\n").encode("utf-8")
                
                # No need for asyncio.sleep here unless specifically needed for rate limiting or a very fast graph.

                # Check for the final response condition based on the node_name and its output
                if node_name == "finalize_response_and_log_chat":
                    if isinstance(node_output_data, dict) and "final_response" in node_output_data:
                        print(f"[STREAM_IRIS_FINAL] Final response detected from node '{node_name}' for thread '{request.langgraphThreadId}'. Ending stream.")
                        return # Stop the entire generator function
            
    except Exception as e:
        import traceback
        print(f"[STREAM_IRIS_EXCEPTION] Error during IRIS agent stream for thread {request.langgraphThreadId}:")
        traceback.print_exc()
        error_output_details = {"error_message": str(e), "trace": traceback.format_exc()}
        error_message_payload = {"node": "system_error", "output": error_output_details}
        try:
            yield (json.dumps(error_message_payload) + "\n").encode("utf-8")
        except Exception as e_json:
            print(f"[STREAM_IRIS_EXCEPTION] Could not serialize exception details for client: {e_json}")
            # Send a generic error if the detailed one can't be serialized
            generic_error_payload = {"node": "system_error", "output": {"error_message": "An unhandled exception occurred during streaming, and its details could not be serialized."}}
            yield (json.dumps(generic_error_payload) + "\n").encode("utf-8")


@app.post("/chat/invoke")
async def chat_invoke_endpoint(request: ChatInvokeRequest):
    return StreamingResponse(
        stream_iris_agent_responses(request),
        media_type="application/x-ndjson" # Use ndjson for line-delimited JSON
    )

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server for IRIS Agent with Uvicorn...")
    # Ensure the host is accessible if running frontend on a different machine/container
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)