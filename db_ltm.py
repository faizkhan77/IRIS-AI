# db_ltm.py
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy import create_engine, text, Enum
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# --- Database Configuration ---
DB_USER = "devuser"
DB_PASSWORD = "iris_agent"
DB_HOST = "45.114.142.157"
DB_PORT = "3306"
DB_NAME = "iris_long_term_memory"

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Helper Functions ---
def _execute_query(query: str, params: Optional[Dict[str, Any]] = None, fetch_one: bool = False, is_insert_with_id_return: bool = False, is_dml: bool = False):
    """
    Executes a SQL query.
    is_insert_with_id_return: Set to True if it's an INSERT and you need the lastrowid.
    is_dml: Set to True for DML statements like UPDATE, DELETE, or INSERT without needing lastrowid (e.g., INSERT ON DUPLICATE KEY UPDATE).
    fetch_one: Set to True if you expect a single row (for SELECT).
    """
    with SessionLocal() as session:
        try:
            result_proxy = session.execute(text(query), params)
            
            if is_insert_with_id_return:
                session.commit()
                return result_proxy.lastrowid
            elif is_dml: # For UPDATE, DELETE, INSERT ON DUPLICATE KEY UPDATE
                session.commit()
                return result_proxy.rowcount # Returns number of affected rows, can be useful
            elif fetch_one: # For SELECT expecting one row
                row = result_proxy.fetchone()
                session.commit() 
                return row
            else: # For SELECT expecting multiple rows
                rows = result_proxy.fetchall()
                session.commit()
                return rows
        except Exception as e:
            session.rollback()
            print(f"Database query error: {e}\nQuery: {query}\nParams: {params}")
            raise

# --- User Management ---
def get_or_create_user(user_identifier: str, name: Optional[str] = None) -> int:
    """
    Retrieves an existing user by email/identifier or creates a new one.
    Returns the user's database ID.
    `user_identifier` is assumed to be the email for now as per schema.
    """
    existing_user = _execute_query(
        "SELECT id FROM users WHERE email = :identifier",
        {"identifier": user_identifier},
        fetch_one=True
    )
    if existing_user:
        return existing_user[0]
    else:
        # Create new user
        user_name = name if name else user_identifier.split('@')[0] if '@' in user_identifier else "New User"
        user_id = _execute_query(
            "INSERT INTO users (email, name) VALUES (:email, :name)",
            {"email": user_identifier, "name": user_name},
            is_insert_with_id_return=True # Corrected parameter name
        )
        print(f"Created new user: {user_identifier} with ID: {user_id}")
        return user_id

# --- Session Management ---
def get_or_create_session(db_user_id: int, langgraph_thread_id: str) -> int:
    """
    Retrieves an existing session by langgraph_thread_id or creates a new one.
    Returns the session's database ID.
    """
    existing_session = _execute_query(
        "SELECT id FROM sessions WHERE langgraph_thread_id = :thread_id AND user_id = :user_id",
        {"thread_id": langgraph_thread_id, "user_id": db_user_id},
        fetch_one=True
    )
    if existing_session:
        # Optionally update ended_at to NULL if session is being "resumed"
        _execute_query(
            "UPDATE sessions SET ended_at = NULL WHERE id = :session_id",
            {"session_id": existing_session[0]},
            is_dml=True # This is an UPDATE
        )
        return existing_session[0]
    else:
        session_id = _execute_query(
            "INSERT INTO sessions (user_id, langgraph_thread_id, started_at) VALUES (:user_id, :thread_id, :started_at)",
            {"user_id": db_user_id, "thread_id": langgraph_thread_id, "started_at": datetime.now()},
            is_insert_with_id_return=True # Corrected parameter name
        )
        print(f"Started new LTM session ID: {session_id} for user ID: {db_user_id}, thread: {langgraph_thread_id}")
        return session_id

def end_ltm_session(db_session_id: int, summary: Optional[str] = None):
    """Marks a session as ended and optionally adds a summary."""
    params = {"session_id": db_session_id, "ended_at": datetime.now()}
    if summary:
        params["summary"] = summary
    
    _execute_query(
        "UPDATE sessions SET ended_at = :ended_at" + (", summary = :summary" if summary else "") + " WHERE id = :session_id",
        params,
        is_dml=True # This is an UPDATE
    )
    print(f"Ended LTM session ID: {db_session_id}")

# --- Preferences & Facts Management ---
def get_user_preferences(db_user_id: int) -> Dict[str, Any]:
    """Retrieves all preferences for a user."""
    rows = _execute_query(
        "SELECT pref_key, pref_val FROM user_preferences WHERE user_id = :user_id",
        {"user_id": db_user_id}
    ) # Default behavior is fetchall for SELECT
    preferences = {}
    for row in rows:
        key, val_str = row
        try:
            # Attempt to parse as JSON if it looks like it, otherwise keep as string
            if val_str and ((val_str.startswith('{') and val_str.endswith('}')) or \
               (val_str.startswith('[') and val_str.endswith(']'))):
                preferences[key] = json.loads(val_str)
            else:
                preferences[key] = val_str
        except (json.JSONDecodeError, TypeError): # TypeError if val_str is None
            preferences[key] = val_str # Keep as raw string if not valid JSON
    return preferences

def set_user_preference(db_user_id: int, pref_key: str, pref_value: Any):
    """Sets or updates a user preference. `pref_value` will be JSON encoded if it's a dict or list."""
    if isinstance(pref_value, (dict, list)):
        val_to_store = json.dumps(pref_value)
    else:
        val_to_store = str(pref_value) # Ensure it's a string

    query = """
    INSERT INTO user_preferences (user_id, pref_key, pref_val, updated_at)
    VALUES (:user_id, :pref_key, :pref_val, :updated_at)
    ON DUPLICATE KEY UPDATE pref_val = VALUES(pref_val), updated_at = VALUES(updated_at);
    """ # Corrected ON DUPLICATE KEY UPDATE syntax slightly to use VALUES()
    _execute_query(
        query,
        {
            "user_id": db_user_id,
            "pref_key": pref_key,
            "pref_val": val_to_store,
            "updated_at": datetime.now()
        },
        is_dml=True # THIS IS THE KEY CHANGE for INSERT ... ON DUPLICATE ...
    )
    print(f"Set preference for user {db_user_id}: {pref_key} = {str(pref_value)[:50]}...")

def add_explicit_facts(db_user_id: int, facts_to_add: List[str]):
    """Adds new facts to a JSON list stored under a specific preference key."""
    if not facts_to_add:
        return

    FACTS_PREF_KEY = "explicitly_remembered_facts"
    current_prefs = get_user_preferences(db_user_id) # This uses SELECT, so it's fine
    existing_facts = current_prefs.get(FACTS_PREF_KEY, [])
    
    if not isinstance(existing_facts, list): 
        print(f"Warning: Corrupted '{FACTS_PREF_KEY}' for user {db_user_id}. Resetting.")
        existing_facts = []

    updated = False
    for fact in facts_to_add:
        if fact not in existing_facts:
            existing_facts.append(fact)
            updated = True
    
    if updated:
        set_user_preference(db_user_id, FACTS_PREF_KEY, existing_facts) # This now uses the corrected _execute_query
        print(f"Added facts for user {db_user_id}: {facts_to_add}")


# --- Chat Log Management ---
def log_chat_message(db_session_id: int, role: str, content: str):
    """Logs a chat message to the LTM chat_logs table."""
    valid_roles = ('user', 'assistant', 'tool', 'system')
    if role.lower() not in valid_roles:
        print(f"Warning: Invalid role '{role}' for chat log. Defaulting to 'assistant'.")
        role_to_log = 'assistant'
    else:
        role_to_log = role.lower()

    _execute_query(
        "INSERT INTO chat_logs (session_id, role, content, ts) VALUES (:session_id, :role, :content, :ts)",
        {
            "session_id": db_session_id,
            "role": role_to_log,
            "content": content,
            "ts": datetime.now()
        },
        is_insert_with_id_return=True # Assuming you might want the chat_log ID, though not strictly needed here
                                    # If not needed, can be is_dml=True
    )

def get_chat_history_for_ltm_session(db_session_id: int, limit: int = 50) -> List[Tuple[str, str, datetime]]:
    """Retrieves chat history for a given LTM session ID."""
    rows = _execute_query(
        "SELECT role, content, ts FROM chat_logs WHERE session_id = :session_id ORDER BY ts DESC LIMIT :limit",
        {"session_id": db_session_id, "limit": limit}
    )
    # Return in chronological order for display (oldest first)
    return [(row[0], row[1], row[2]) for row in reversed(rows)]


if __name__ == '__main__':
    print("Testing DB LTM module...")
    # Example Usage (ensure your DB is set up and running)
    try:
        # Test User
        test_email = f"testuser_{datetime.now().strftime('%H%M%S%f')}@example.com" # Unique email per run
        db_uid = get_or_create_user(test_email, "Test User")
        print(f"DB User ID: {db_uid}")

        # Test Preferences
        set_user_preference(db_uid, "favorite_color", "blue")
        set_user_preference(db_uid, "risk_profile", {"level": "high", "notes": "likes tech"})
        add_explicit_facts(db_uid, ["Loves AI", "Prefers morning meetings"])
        add_explicit_facts(db_uid, ["Loves AI"]) # Test duplicate fact

        prefs = get_user_preferences(db_uid)
        print(f"User Preferences: {json.dumps(prefs, indent=2)}")

        # Test Session & Chat Logs
        test_thread_id = f"test_thread_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        db_sid = get_or_create_session(db_uid, test_thread_id)
        print(f"DB Session ID: {db_sid}")

        log_chat_message(db_sid, "user", "Hello IRIS!")
        log_chat_message(db_sid, "assistant", "Hello! How can I help you today?")
        log_chat_message(db_sid, "user", "What's my favorite color?")
        log_chat_message(db_sid, "assistant", f"Based on my memory, your favorite color is {prefs.get('favorite_color')}.")
        
        history = get_chat_history_for_ltm_session(db_sid)
        print("\nLTM Chat History:")
        for role, content, ts in history:
            print(f"  {ts} [{role.upper()}]: {content}")

        # end_ltm_session(db_sid, "User asked about preferences and tested chat logging.")
        print(f"Test complete for user {test_email} (ID: {db_uid}), session thread {test_thread_id} (LTM Session ID: {db_sid}).")

    except Exception as e:
        print(f"Error during db_ltm test: {e}")
        import traceback
        traceback.print_exc()