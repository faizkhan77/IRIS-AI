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

    session_lg_thread_id = f"test_cli_thread_{str(uuid.uuid4())[:8]}"
    
    print(f"\n{Fore.CYAN}--- Session Details ---")
    print(f"User Email (LTM Identifier): {user_email}")
    if user_name:
        print(f"User Name: {user_name}")
    print(f"LangGraph Thread ID (STM): {session_lg_thread_id}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'exit', 'quit', or 'bye' to end the session.{Style.RESET_ALL}\n")

    # 2. Compile IRIS App with Checkpointer
    with SqliteSaver.from_conn_string(STM_CHECKPOINT_DB) as checkpointer:
        print(f"{Fore.CYAN}Compiling IRIS graph with SQLite checkpointer ({STM_CHECKPOINT_DB})...{Style.RESET_ALL}")
        iris_app = iris_graph_builder.compile(checkpointer=checkpointer)
        print(f"{Fore.GREEN}IRIS graph compiled successfully.{Style.RESET_ALL}\n")

        config_for_stream = {"configurable": {"thread_id": session_lg_thread_id}}
        
        base_inputs = {
            "user_identifier": user_email,
            "langgraph_thread_id": session_lg_thread_id,
        }
        # user_name is handled by db_ltm, not needed in state

        # 3. Conversation Loop
        while True:
            user_query = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}").strip()

            if user_query.lower() in ["exit", "quit", "bye"]:
                print(f"{Fore.MAGENTA}Ending session. Goodbye!{Style.RESET_ALL}")
                break

            if not user_query:
                continue

            current_turn_inputs = {**base_inputs, "user_input": user_query}
            final_iris_response = None
            
            # The console output from iris.py will show the node flow.
            # Here, we just focus on getting the final result.
            print(f"{Fore.BLUE}--- IRIS Processing Start ---{Style.RESET_ALL}")
            with loading_animation("IRIS is thinking..."):
                try:
                    # Stream the events to find the final response
                    for event_value_map in iris_app.stream(current_turn_inputs, config=config_for_stream, stream_mode="values"):
                        if "final_response" in event_value_map and event_value_map["final_response"]:
                            final_iris_response = event_value_map["final_response"]
                
                except Exception as e:
                    print(f"\r{Fore.RED}Error during IRIS processing: {e}{Style.RESET_ALL}")
                    import traceback
                    traceback.print_exc()
                    final_iris_response = "I encountered an error. Please try again."

            print(f"{Fore.BLUE}--- IRIS Processing End ---{Style.RESET_ALL}")

            if final_iris_response:
                print(f"{Fore.GREEN}IRIS: {Style.RESET_ALL}{final_iris_response}")
            else:
                print(f"{Fore.RED}IRIS: No response generated.{Style.RESET_ALL}")


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