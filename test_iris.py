# test_iris.py
import os
import sys
import uuid
import time
from contextlib import contextmanager

# --- Path Setup ---
# Get the absolute path of the current script (test_iris.py, which is in root)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add ROOT_DIR to the beginning of sys.path
# This ensures that when agents.iris is imported, its own non-relative imports
# (like `import db_ltm` or `import model_config`) will find files in ROOT_DIR.
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# The agents directory itself doesn't need to be added to sys.path separately
# if we are importing like `from agents.iris import ...` because Python will
# find the 'agents' package in ROOT_DIR.
# current_dir = os.path.dirname(os.path.abspath(__file__)) # This is ROOT_DIR
# agents_dir = os.path.join(ROOT_DIR, "agents")
# if agents_dir not in sys.path:
# sys.path.insert(0, agents_dir) # This line might actually be causing confusion or is redundant if ROOT_DIR is already first.
# Let's remove the explicit agents_dir addition for now and rely on ROOT_DIR.

from langgraph.checkpoint.sqlite import SqliteSaver
from colorama import Fore, Style, init as colorama_init

try:
    from agents.iris import iris_graph_builder # This should find agents/iris.py
    # db_ltm will be imported by agents.iris directly from ROOT_DIR
except ImportError as e:
    print(f"{Fore.RED}Error importing IRIS modules. Structure:")
    print(f"ROOT_DIR expected: {ROOT_DIR}")
    print(f"sys.path: {sys.path}")
    print(f"Ensure 'agents' directory with '__init__.py' and 'iris.py' exists in ROOT_DIR.")
    print(f"Ensure 'db_ltm.py' and 'model_config.py' exist in ROOT_DIR.")
    print(f"Details: {e}{Style.RESET_ALL}")
    sys.exit(1)
try:
    import db_ltm # For direct use in test script, should also find from ROOT_DIR
except ImportError as e:
    print(f"{Fore.RED}Error importing db_ltm directly in test_iris.py: {e}{Style.RESET_ALL}")
    sys.exit(1)

# Initialize Colorama
colorama_init(autoreset=True)

# --- Configuration ---
STM_CHECKPOINT_DB = "iris_stm_checkpoint_test.sqlite" # Use a separate DB for testing

# --- Helper for Loading Animation ---
@contextmanager
def loading_animation(text="Processing"):
    chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"] # Braille spinner
    # chars = ["|", "/", "-", "\\"] # Simpler spinner
    idx = 0
    stop_animation = False

    def animate():
        nonlocal idx
        while not stop_animation:
            sys.stdout.write(f"\r{Fore.CYAN}{text} {chars[idx % len(chars)]} {Style.RESET_ALL}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)

    import threading
    spinner_thread = threading.Thread(target=animate)
    spinner_thread.daemon = True # Allow main program to exit even if thread is still running
    spinner_thread.start()
    
    try:
        yield
    finally:
        stop_animation = True
        spinner_thread.join(timeout=0.5) # Give thread a moment to finish
        sys.stdout.write(f"\r{' ' * (len(text) + 5)}\r") # Clear the animation line
        sys.stdout.flush()

# --- Main Test Application ---
def run_iris_test_session():
    print(f"{Fore.MAGENTA}--- IRIS Agent Test Session ---{Style.RESET_ALL}")

    # 1. Get User Information
    print(f"{Fore.YELLOW}Let's get some details to start a session.")
    user_email = input(f"{Fore.GREEN}Enter your email address: {Style.RESET_ALL}").strip()
    user_name = input(f"{Fore.GREEN}Enter your name (optional, press Enter to skip): {Style.RESET_ALL}").strip()

    if not user_email:
        print(f"{Fore.RED}Email address is required to start. Exiting.{Style.RESET_ALL}")
        return

    # Use a unique LangGraph thread ID for this entire test session
    # This allows conversation continuity within this single run of test_iris.py
    session_lg_thread_id = f"test_cli_thread_{str(uuid.uuid4())[:8]}"
    
    print(f"\n{Fore.CYAN}--- Session Details ---")
    print(f"User Email (LTM Identifier): {user_email}")
    if user_name:
        print(f"User Name: {user_name}")
    print(f"LangGraph Thread ID (STM): {session_lg_thread_id}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'exit', 'quit', or 'bye' to end the session.{Style.RESET_ALL}\n")

    # 2. Compile IRIS App with Checkpointer
    # The checkpointer context needs to be active for the duration of the session
    with SqliteSaver.from_conn_string(STM_CHECKPOINT_DB) as checkpointer:
        print(f"{Fore.CYAN}Compiling IRIS graph with SQLite checkpointer ({STM_CHECKPOINT_DB})...{Style.RESET_ALL}")
        iris_app = iris_graph_builder.compile(checkpointer=checkpointer)
        print(f"{Fore.GREEN}IRIS graph compiled successfully.{Style.RESET_ALL}\n")

        # Configuration for LangGraph stream
        config_for_stream = {"configurable": {"thread_id": session_lg_thread_id}}
        
        # Initial input for the graph for this session
        base_inputs = {
            "user_identifier": user_email, # This is the LTM key
            "langgraph_thread_id": session_lg_thread_id,
            # 'db_user_id' and 'db_session_id' will be set by the graph
        }
        if user_name: # Pass name if provided, db_ltm.get_or_create_user can use it
            # This isn't directly in IrisState, but db_ltm.get_or_create_user can accept it.
            # For now, get_or_create_user in db_ltm.py takes email and optional name.
            # IrisState doesn't need 'user_name' directly.
            pass


        # 3. Conversation Loop
        while True:
            user_query = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}").strip()

            if user_query.lower() in ["exit", "quit", "bye"]:
                print(f"{Fore.MAGENTA}Ending session. Goodbye!{Style.RESET_ALL}")
                # Optionally, call db_ltm.end_ltm_session here if you want to mark it in LTM
                # This would require fetching the db_session_id from the last state,
                # or modifying IRIS to return it or for db_ltm to find it via langgraph_thread_id.
                # For simplicity, we'll skip explicit LTM session ending here.
                break

            if not user_query:
                continue

            current_turn_inputs = {**base_inputs, "user_input": user_query}
            final_iris_response = None
            node_flow = []

            with loading_animation("IRIS is thinking..."):
                try:
                    for event_value_map in iris_app.stream(current_turn_inputs, config=config_for_stream, stream_mode="values"):
                        # Identify the node that just ran
                        # The event_value_map keys are the fields updated by the last node(s).
                        # The last key in the dict is often the primary output of the node.
                        # For a more accurate "current node," LangGraph event stream with `stream_mode="events"`
                        # would be better, but "values" is simpler for getting state.
                        
                        # Heuristic: the last key in the event that is not a known state key
                        # or a special key like "__end__" might indicate the node.
                        # This is a bit tricky with stream_mode="values".
                        # Let's just track major state updates.
                        
                        updated_keys = list(event_value_map.keys())
                        # Try to infer node from updated keys, or use a simpler message
                        # For simplicity in this test script, we'll focus on the final output.
                        # More detailed node tracking would involve inspecting the 'op' in 'events' stream_mode.

                        # We can print when key parts of the state are updated.
                        if "supervisor_decision_output" in event_value_map and event_value_map["supervisor_decision_output"]:
                            node_name = "Supervisor Decide"
                            if node_name not in node_flow: node_flow.append(node_name)
                        elif "intermediate_response" in event_value_map and event_value_map["intermediate_response"]:
                            # Could be sub-agent call or direct iris response
                            # Determine based on supervisor_decision.route if available
                            # For now, generic
                            if "Call" not in " ".join(node_flow) and "Prepare Direct" not in " ".join(node_flow):
                                supervisor_decision = event_value_map.get("supervisor_decision_output")
                                if supervisor_decision:
                                    route = supervisor_decision.route
                                    if route in ["fundamentals", "sentiment", "technicals"]:
                                        node_name = f"Call {route.capitalize()} Agent"
                                    else:
                                        node_name = "Prepare Direct IRIS Response"
                                    if node_name not in node_flow: node_flow.append(node_name)

                        if "final_response" in event_value_map and event_value_map["final_response"]:
                            final_iris_response = event_value_map["final_response"]
                            if "Finalize Response" not in node_flow: node_flow.append("Finalize Response")
                    
                    # Print node flow after processing is complete
                    if node_flow:
                        flow_str = " -> ".join([f"{Fore.CYAN}{name}{Style.RESET_ALL}" for name in node_flow])
                        print(f"\r{Fore.BLUE}Processed flow: {flow_str} {Fore.GREEN}✔️{Style.RESET_ALL}")

                except Exception as e:
                    print(f"\r{Fore.RED}Error during IRIS processing: {e}{Style.RESET_ALL}")
                    import traceback
                    traceback.print_exc()
                    final_iris_response = "I encountered an error. Please try again."

            if final_iris_response:
                print(f"{Fore.GREEN}IRIS: {Style.RESET_ALL}{final_iris_response}")
            else:
                print(f"{Fore.RED}IRIS: No response generated.{Style.RESET_ALL}")
            
            node_flow.clear() # Reset for next turn


if __name__ == "__main__":
    # Ensure necessary environment variables are set (e.g., GROQ_API_KEY, DB credentials if not default)
    # The .env file should be loaded by iris.py or db_ltm.py already
    if not os.getenv("GROQ_API_KEY"):
        print(f"{Fore.RED}GROQ_API_KEY not found. Please set it in your .env file.{Style.RESET_ALL}")
        sys.exit(1)
    
    # Check if DB LTM can be connected (optional pre-check)
    try:
        with db_ltm.engine.connect() as connection:
            print(f"{Fore.GREEN}Successfully connected to LTM database ({db_ltm.DB_NAME}).{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Failed to connect to LTM database: {e}")
        print(f"Ensure MySQL is running and configured as per db_ltm.py.{Style.RESET_ALL}")
        sys.exit(1)

    run_iris_test_session()

    # Cleanup the test STM database (optional)
    # if os.path.exists(STM_CHECKPOINT_DB):
    #     os.remove(STM_CHECKPOINT_DB)
    #     print(f"\n{Fore.CYAN}Cleaned up test STM database: {STM_CHECKPOINT_DB}{Style.RESET_ALL}")