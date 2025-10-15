# Open-LLM-webchat - Clear long-term memory by user
import os
import sqlite3
import settings
import gradio as gr

from colorama import Fore

def clear_long_term_memory(username: str) -> gr.update:
    """
    Clear the long-term memory of a specific user from the database.

    This function connects to the SQLite database defined in settings, checks
    if the `user_memories` table exists, verifies if the specified user has
    records, and deletes all long-term memory entries associated with that user.
    It prints informative messages for each step and handles database errors.

    Args:
        username (str): The name of the user whose long-term memory will be cleared.

    Returns:
        gr.update: A Gradio UI update object to hide the modal memory manegement.
    """
    
    db_path = settings.DB_FILE

    # Check if the file exists
    if not os.path.exists(db_path):
        print(f"{Fore.LIGHTYELLOW_EX}Database not found: {db_path}{Fore.RESET}")
        gr.Warning(f"Database not found: {db_path}", duration=3)
        return gr.update(visible=False)

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Checks if the user_memories table exists
        cur.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='user_memories';
        """)
        if cur.fetchone() is None:
            print(f"{Fore.LIGHTYELLOW_EX}Table 'user_memories' does not exist.{Fore.RESET}")
            conn.close()
            gr.Warning("Table 'user_memories' does not exist.", duration=3)
            return gr.update(visible=False)

        # Checks if user exists in table
        cur.execute("SELECT COUNT(*) FROM user_memories WHERE user_id = ?", (username,))
        count = cur.fetchone()[0]

        if count == 0:
            print(f"{Fore.LIGHTYELLOW_EX}No records found for user '{username}'.{Fore.RESET}")
            conn.close()
            gr.Warning(f"No records found for user '{username}'", duration=3)
            return gr.update(visible=False)

        # Delete the records
        cur.execute("DELETE FROM user_memories WHERE user_id = ?", (username,))
        conn.commit()
        conn.close()
        
        gr.Success(f"Long-term memory erased for the user '{username}'.", duration=3)

        print(f"{Fore.LIGHTGREEN_EX}Long-term memory erased for the user '{username}'.{Fore.RESET}")

    except sqlite3.Error as e:
        print(f"{Fore.RED}Error clearing memory: {e}{Fore.RESET}")
        gr.Error("Error clearing memory.", duration=2)

    return gr.update(visible=False)
