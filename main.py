# /iris-agent-backend/main.py

import os
import sys
import json
import uuid
import asyncio
from typing import TypedDict, Annotated, List as TypingList, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

# Ensure the current directory is in PYTHONPATH to find iris and sub_agents
# This adds /iris-agent-backend to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Attempt to import from agents/iris.py and its dependencies
try:
    # Assuming iris.py is in an 'agents' subdirectory relative to main.py
    from agents.iris import iris_graph_builder, SupervisorDecision # Your main graph builder
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langchain_core.messages import BaseMessage # Assuming this is available if langgraph is
    print("Successfully imported iris_graph_builder and SqliteSaver from agents.iris.")
except ImportError as e:
    print(f"Error importing from agents.iris or its dependencies: {e}")
    print("Please ensure agents/iris.py, model_config.py, db_ltm.py, and sub_agents are correctly placed and importable.")
    print("Using dummy fallbacks for server to run, but agent functionality will be limited.")
    # Define dummy versions if full iris.py setup is not available
    class SupervisorDecision(BaseModel): route: str = "dummy"; reasoning: str = "dummy" 
    class BaseMessage: # Dummy for BaseMessage if langchain_core.messages fails (less likely)
        def dict(self): 
            return {"type": "dummy", "content": "dummy message content"}
    class DummyApp:
        async def astream(self, inputs: Dict[str, Any], config: Dict[str, Any], stream_mode: str):
            yield {"initialize_session_and_ltm_ids": {"db_user_id": 0, "db_session_id": 0, "chat_history": []}}
            await asyncio.sleep(0.1)
            yield {"supervisor_decide": {"supervisor_decision_output": SupervisorDecision().model_dump()}}
            await asyncio.sleep(0.1)
            yield {"finalize_response_and_log_chat": {"final_response": "This is a dummy response from a partially loaded agent."}}
    class DummyGraphBuilder:
        def compile(self, checkpointer=None):
            print("COMPILED DUMMY APP")
            return DummyApp()
    iris_graph_builder = DummyGraphBuilder()
    SqliteSaver = None # Will prevent checkpointer usage if it failed to import
    # If BaseMessage itself failed to import, this would be an issue for safe_serialize
    # but it's typically a core part of langchain.

# --- FastAPI App Setup ---
app = FastAPI()

# CORS Middleware: Allow requests from your React frontend (default port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LangGraph App Compilation ---
IRIS_STM_CHECKPOINT_DB = "iris_stm_checkpoint.sqlite" 
iris_app_instance = None

if SqliteSaver:
    try:
        global_checkpointer = SqliteSaver.from_conn_string(IRIS_STM_CHECKPOINT_DB)
        iris_app_instance = iris_graph_builder.compile(checkpointer=global_checkpointer)
        print(f"IRIS Graph compiled successfully with SqliteSaver using '{IRIS_STM_CHECKPOINT_DB}'.")
    except Exception as e:
        print(f"Failed to compile IRIS graph with SqliteSaver: {e}")
        print("Falling back to compiling without checkpointer (if dummy graph).")
        if isinstance(iris_graph_builder, DummyGraphBuilder): 
             iris_app_instance = iris_graph_builder.compile()
else: 
    print("SqliteSaver not available. Compiling IRIS graph (potentially dummy) without checkpointer.")
    iris_app_instance = iris_graph_builder.compile() 

if iris_app_instance is None:
    print("CRITICAL: iris_app_instance could not be compiled. API will not function correctly.")
    iris_app_instance = DummyGraphBuilder().compile()


# --- Pydantic Models for API ---
class ChatInvokeRequest(BaseModel):
    userInput: str
    userIdentifier: str         
    langgraphThreadId: str      


# --- Helper for Serialization ---
def safe_serialize(data_obj):
    if isinstance(data_obj, SupervisorDecision):
        return data_obj.model_dump()
    # Check for BaseMessage specifically if it's not a Pydantic model
    # but has a dict() method as defined in the dummy or actual langchain_core.
    if hasattr(data_obj, 'dict') and callable(getattr(data_obj, 'dict')) and not isinstance(data_obj, type(dict)):
         # The `not isinstance(data_obj, type(dict))` is to avoid calling .dict() on plain dicts
         # that might happen to have a key 'dict'. More robustly, check against imported BaseMessage type.
        try:
            # Check if it's an actual BaseMessage or our dummy
            # This check might need to be more specific if other classes also have .dict()
            if "content" in data_obj.dict() and "type" in data_obj.dict(): # Heuristic for BaseMessage-like
                 return data_obj.dict()
        except Exception:
            pass # Fall through if .dict() fails or isn't what we expect

    if isinstance(data_obj, dict):
        return {k: safe_serialize(v) for k, v in data_obj.items()}
    if isinstance(data_obj, list):
        return [safe_serialize(item) for item in data_obj]
    
    try:
        if hasattr(data_obj, 'model_dump'): # For other Pydantic models
            return data_obj.model_dump()
    except Exception:
        pass 
    
    try:
        json.dumps(data_obj) 
        return data_obj
    except (TypeError, OverflowError):
        return str(data_obj)


# --- Streaming Endpoint Logic ---
async def stream_iris_agent_responses(request: ChatInvokeRequest):
    if iris_app_instance is None:
        # Ensure error message follows the same JSON structure
        error_msg = {"node": "system_error", "output": {"error_message": "IRIS agent is not initialized on the server."}}
        yield (json.dumps(error_msg) + "\n").encode("utf-8")
        return

    initial_state_for_graph = {
        "user_input": request.userInput,
        "user_identifier": request.userIdentifier,
        "langgraph_thread_id": request.langgraphThreadId,
        "chat_history": [], # Start with empty chat history for new threads; checkpointer handles loading for existing threads
    }
    config_for_graph = {"configurable": {"thread_id": request.langgraphThreadId}}

    print(f"Streaming for thread_id: {request.langgraphThreadId}, user: {request.userIdentifier}")

    try:
        async for event_value_map in iris_app_instance.astream(
            initial_state_for_graph,
            config=config_for_graph,
            stream_mode="values" 
        ):
            if not event_value_map: # Should not happen with stream_mode="values" unless graph is empty
                continue

            node_name = list(event_value_map.keys())[0]
            node_output_data = event_value_map[node_name]

            serializable_node_output = safe_serialize(node_output_data)
            payload_to_send = {"node": node_name, "output": serializable_node_output}
            
            try:
                json_payload = json.dumps(payload_to_send) + "\n"
                yield json_payload.encode("utf-8")
            except (TypeError, OverflowError) as e:
                print(f"Serialization error for node {node_name}: {e}. Output was: {payload_to_send}")
                error_payload = json.dumps({"node": node_name, "output": {"error": "Serialization issue", "details": str(e)}}) + "\n"
                yield error_payload.encode("utf-8")
            
            await asyncio.sleep(0.01) 

            if node_name == "finalize_response_and_log_chat":
                if isinstance(serializable_node_output, dict) and "final_response" in serializable_node_output:
                    print(f"Final response detected for thread {request.langgraphThreadId}. Ending stream.")
                    break 
    
    except Exception as e:
        print(f"Error during IRIS agent stream for thread {request.langgraphThreadId}: {e}")
        import traceback
        traceback.print_exc()
        # Ensure error message follows the same JSON structure
        error_output = {"error_message": str(e), "trace": traceback.format_exc()}
        error_message_payload = {"node": "error", "output": error_output}
        yield (json.dumps(error_message_payload) + "\n").encode("utf-8") # Corrected utf-f-8 to utf-8


@app.post("/chat/invoke")
async def chat_invoke_endpoint(request: ChatInvokeRequest):
    return StreamingResponse(
        stream_iris_agent_responses(request),
        media_type="application/x-ndjson" 
    )

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server for IRIS Agent...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)