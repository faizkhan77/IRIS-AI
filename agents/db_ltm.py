# agents/db_ltm.py

import mysql.connector
from mysql.connector import Error, pooling
import os
from dotenv import load_dotenv
from typing import Tuple, Optional

load_dotenv()

# --- Database Connection Pool ---
try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="iris_pool",
        pool_size=5,
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "root"),
        database="iris_long_term_memory"
    )
    print("MySQL Connection Pool for LTM created successfully.")
except Error as e:
    print(f"Error while connecting to MySQL using Connection pool: {e}")
    db_pool = None

def get_db_connection():
    """Gets a connection from the pool."""
    if not db_pool:
        raise ConnectionError("Database connection pool is not initialized.")
    return db_pool.get_connection()


# --- THIS IS THE NEW, CORRECT LOGIN FUNCTION FOR THE UI ---
def authenticate_and_get_user(email: str, name: str) -> dict:
    """
    Handles user login and creation based on email and name.
    Matches your exact schema and login requirements.
    Returns a dictionary of the user's data or None if auth fails.
    """
    cnx = get_db_connection()
    cursor = cnx.cursor(dictionary=True)
    try:
        # Check if user exists by email
        cursor.execute("SELECT id, email, name FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        if user:
            # User exists, verify name if provided
            if name and user.get('name') and user['name'].lower() != name.lower():
                # Name mismatch, authentication fails
                return None 
            # If name matches, or stored name is null, or provided name is empty, login is successful
            return user
        else:
            # User does not exist, create a new one
            cursor.execute(
                "INSERT INTO users (email, name) VALUES (%s, %s)",
                (email, name if name else None)
            )
            cnx.commit()
            new_user_id = cursor.lastrowid
            # Fetch and return the new user's data
            cursor.execute("SELECT id, email, name FROM users WHERE id = %s", (new_user_id,))
            return cursor.fetchone()
    finally:
        cursor.close()
        cnx.close()


# --- THIS IS THE SIMPLIFIED FUNCTION FOR THE IRIS BACKEND ---
def get_or_create_user(user_identifier: str) -> int:
    """
    Gets user ID by email. If not found, creates a new user with only the email.
    This is used by the agent backend, which only knows the user_identifier (email).
    """
    cnx = get_db_connection()
    cursor = cnx.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (user_identifier,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            # Create a new user with email only, name can be null
            cursor.execute("INSERT INTO users (email) VALUES (%s)", (user_identifier,))
            cnx.commit()
            return cursor.lastrowid
    finally:
        cursor.close()
        cnx.close()


# --- ALL OTHER FUNCTIONS MATCHING YOUR SCHEMA ---

def get_or_create_session(user_id: int, langgraph_thread_id: str) -> int:
    """Gets session ID, creates one if not exists. Matches your `sessions` table."""
    cnx = get_db_connection()
    cursor = cnx.cursor()
    try:
        cursor.execute("SELECT id FROM sessions WHERE langgraph_thread_id = %s", (langgraph_thread_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            cursor.execute("INSERT INTO sessions (user_id, langgraph_thread_id) VALUES (%s, %s)", (user_id, langgraph_thread_id))
            cnx.commit()
            return cursor.lastrowid
    finally:
        cursor.close()
        cnx.close()

def get_user_sessions(user_id: int) -> list:
    """Retrieves all chat sessions for a given user, ordered by most recent."""
    cnx = get_db_connection()
    cursor = cnx.cursor(dictionary=True)
    try:
        query = """
            SELECT id, langgraph_thread_id as thread_id, started_at,summary
            FROM sessions
            WHERE user_id = %s
            ORDER BY started_at DESC
        """
        cursor.execute(query, (user_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        cnx.close()

def get_session_chat_logs(session_id: int) -> list:
    """Retrieves all messages for a specific session, ordered chronologically."""
    cnx = get_db_connection()
    cursor = cnx.cursor(dictionary=True)
    try:
        query = """
            SELECT role, content
            FROM chat_logs
            WHERE session_id = %s
            ORDER BY ts ASC
        """
        cursor.execute(query, (session_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        cnx.close()

def log_chat_message(session_id: int, role: str, content: str):
    """Logs a single message to the chat_logs table."""
    cnx = get_db_connection()
    cursor = cnx.cursor()
    try:
        query = "INSERT INTO chat_logs (session_id, role, content) VALUES (%s, %s, %s)"
        cursor.execute(query, (session_id, role, content))
        cnx.commit()
    finally:
        cursor.close()
        cnx.close()

def set_user_preference(user_id: int, key: str, value: str):
    """Sets or updates a user preference."""
    cnx = get_db_connection()
    cursor = cnx.cursor()
    try:
        query = """
            INSERT INTO user_preferences (user_id, pref_key, pref_val)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE pref_val = VALUES(pref_val)
        """
        cursor.execute(query, (user_id, key, value))
        cnx.commit()
    finally:
        cursor.close()
        cnx.close()

def get_user_preferences(user_id: int) -> dict:
    """Retrieves all preferences for a user."""
    cnx = get_db_connection()
    cursor = cnx.cursor(dictionary=True)
    try:
        query = "SELECT pref_key, pref_val FROM user_preferences WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        results = cursor.fetchall()
        return {row['pref_key']: row['pref_val'] for row in results}
    finally:
        cursor.close()
        cnx.close()

def get_messages_by_thread_id(thread_id: str) -> list:
    """
    Retrieves all messages for a specific session using the langgraph_thread_id.
    This is the function our FastAPI endpoint will use.
    """
    cnx = get_db_connection()
    cursor = cnx.cursor(dictionary=True)
    try:
        # This query efficiently joins the sessions and chat_logs tables
        # to find messages based on the thread_id.
        query = """
            SELECT cl.role, cl.content, cl.ts as timestamp
            FROM chat_logs cl
            JOIN sessions s ON cl.session_id = s.id
            WHERE s.langgraph_thread_id = %s
            ORDER BY cl.ts ASC
        """
        cursor.execute(query, (thread_id,))
        return cursor.fetchall()
    finally:
        cursor.close()
        cnx.close()


# --- NEW FUNCTION FOR SESSION SUMMARIZATION ---
def update_session_summary(session_id: int, summary: str):
    """Updates the summary for a given session."""
    cnx = get_db_connection()
    cursor = cnx.cursor()
    try:
        query = "UPDATE sessions SET summary = %s WHERE id = %s"
        cursor.execute(query, (summary, session_id))
        cnx.commit()
    finally:
        cursor.close()
        cnx.close()

def get_session_and_user_ids_by_thread(thread_id: str) -> Optional[Tuple[int, int]]:
    """Fetches user_id and session_id based on the langgraph_thread_id."""
    cnx = get_db_connection()
    cursor = cnx.cursor()
    try:
        # We join to get the user_id from the parent table
        query = """
            SELECT s.id, s.user_id
            FROM sessions s
            WHERE s.langgraph_thread_id = %s
        """
        cursor.execute(query, (thread_id,))
        result = cursor.fetchone()
        if result:
            session_id, user_id = result
            return session_id, user_id
        return None
    finally:
        cursor.close()
        cnx.close()

