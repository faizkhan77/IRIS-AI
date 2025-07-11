# agents/db_ltm.py
import mysql.connector
from mysql.connector import Error
import os

# --- Database Connection Pool ---
try:
    db_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="iris_pool",
        pool_size=5,
        host="localhost",
        user="root",
        password="root", # Or your actual password
        database="iris_long_term_memory"
    )
    print("MySQL Connection Pool created successfully")
except Error as e:
    print(f"Error while connecting to MySQL using Connection pool: {e}")
    db_pool = None

def get_db_connection():
    if not db_pool:
        raise ConnectionError("Database connection pool is not initialized.")
    return db_pool.get_connection()

# --- User and Session Management ---
def get_or_create_user(user_identifier: str) -> int:
    """Gets user ID by email/identifier, creates one if not exists."""
    cnx = get_db_connection()
    cursor = cnx.cursor()
    try:
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (user_identifier,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            # Create new user
            cursor.execute("INSERT INTO users (email) VALUES (%s)", (user_identifier,))
            cnx.commit()
            return cursor.lastrowid
    finally:
        cursor.close()
        cnx.close()

def get_or_create_session(db_user_id: int, langgraph_thread_id: str) -> int:
    """Gets session ID, creates one if not exists."""
    cnx = get_db_connection()
    cursor = cnx.cursor()
    try:
        cursor.execute("SELECT id FROM sessions WHERE langgraph_thread_id = %s", (langgraph_thread_id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            cursor.execute("INSERT INTO sessions (user_id, langgraph_thread_id) VALUES (%s, %s)", (db_user_id, langgraph_thread_id))
            cnx.commit()
            return cursor.lastrowid
    finally:
        cursor.close()
        cnx.close()

# --- Chat Logging ---
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

# --- Long-Term Memory (Preferences) ---
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