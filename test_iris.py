# test_iris.py
import os
import sys
import uuid
import time
import asyncio
import threading
import traceback
from contextlib import contextmanager

# Path Setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from colorama import Fore, Style, init as colorama_init
from agents.iris import IrisOrchestrator
import db_ltm

colorama_init(autoreset=True)
STM_CHECKPOINT_DB = "iris_stm_checkpoint.sqlite"

@contextmanager
def loading_animation(text="IRIS is thinking"):
    # (Loading animation code is unchanged)
    stop_animation = False
    spinner_thread = None
    def animate():
        chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        idx = 0
        while not stop_animation:
            sys.stdout.write(f"\r{Fore.CYAN}{text} {chars[idx % len(chars)]} {Style.RESET_ALL}")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.1)
    try:
        spinner_thread = threading.Thread(target=animate)
        spinner_thread.daemon = True
        spinner_thread.start()
        yield
    finally:
        stop_animation = True
        if spinner_thread: spinner_thread.join(timeout=0.2)
        sys.stdout.write(f"\r{' ' * (len(text) + 5)}\r")
        sys.stdout.flush()

def generate_graph_diagram(orchestrator):
    """Generates a Mermaid syntax diagram of the LangGraph flow."""
    try:
        # Get the graph object from the compiled app
        graph = orchestrator.app.get_graph()
        
        # Generate the Mermaid diagram string
        mermaid_string = graph.draw_mermaid()
        
        # Save to a file
        diagram_path = os.path.join(ROOT_DIR, "iris_graph_diagram.md")
        with open(diagram_path, "w") as f:
            f.write("### IRIS Agent Graph Diagram\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_string)
            f.write("\n```\n")
        
        print("\n" + "="*50)
        print(f"{Fore.MAGENTA}Graph diagram has been generated!{Style.RESET_ALL}")
        print(f"File saved to: {Style.BRIGHT}{diagram_path}{Style.RESET_ALL}")
        print("You can view this file in a Markdown viewer that supports Mermaid (like VS Code with a Mermaid extension, or GitHub).")
        print("="*50 + "\n")

    except Exception as e:
        print(f"{Fore.RED}Could not generate graph diagram: {e}{Style.RESET_ALL}")


async def main_chat_loop():
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    async with AsyncSqliteSaver.from_conn_string(STM_CHECKPOINT_DB) as memory_checkpointer:
        orchestrator = IrisOrchestrator(checkpointer=memory_checkpointer)
        
        # Generate the diagram after the orchestrator is initialized
        generate_graph_diagram(orchestrator)
        
        user_id, thread_id = f"user_cli_{uuid.uuid4().hex[:6]}", f"thread_cli_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_payload = {"user_identifier": user_id, "thread_id": thread_id}
        
        print(f"{Fore.MAGENTA}IRIS Chat Session Initialized{Style.RESET_ALL}")
        print(f"LTM User ID: {Style.BRIGHT}{user_id}{Style.RESET_ALL}")
        print(f"STM Thread ID: {Style.BRIGHT}{thread_id}{Style.RESET_ALL}")
        print("Type 'exit' or 'quit' to end the session.")
        print("="*50 + "\n")

        while True:
            try:
                user_query = input(f"{Fore.GREEN}You: {Style.RESET_ALL}")
                if user_query.lower() in ["exit", "quit"]: break
                if not user_query.strip(): continue
                payload = {**initial_payload, "user_input": user_query}
                with loading_animation():
                    await orchestrator.ainvoke(payload, config)
            except KeyboardInterrupt: break
            except Exception as e:
                print(f"{Fore.RED}\nAn unexpected error occurred: {e}")
                traceback.print_exc()
                break
        print(f"\n{Fore.YELLOW}Ending session. Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main_chat_loop())