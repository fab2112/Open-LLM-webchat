# Open-LLM-webchat - Process user sessions from data base
import os
import uuid
import json
import sqlite3
import settings

import gradio as gr
from typing import Tuple
from colorama import Fore
from datetime import datetime
from gradio import ChatMessage


def ensure_tmp_directory(db_file: str):
    """
    Ensure that the directory for a given database file exists.

    This function checks if the directory containing the specified SQLite database file
    exists, and if not, it creates the directory. Success and error messages are printed
    to the console for feedback.

    Args:
        db_file (str): Path to the SQLite database file for which the directory should exist.
    """
    db_dir = os.path.dirname(db_file)
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
            print(f"{Fore.GREEN}Created directory: {db_dir}{Fore.RESET}")
        except Exception as e:
            print(f"{Fore.RED}Failed to create directory {db_dir}: {e}{Fore.RESET}")


def get_session_ids_from_db(username: str) -> gr.update:
    """
    Retrieve all session IDs for a given user from the database.

    This function connects to the SQLite database defined in `settings.DB_FILE`,
    ensures the database directory exists, checks if the `agent_sessions` table
    exists, and fetches all session IDs associated with the specified username.
    If the table does not exist or an error occurs, an empty list is returned.

    Args:
        username (str): The user ID for which to fetch session IDs.

    Returns:
        gr.update: A Gradio update object containing the list of session IDs (reversed order).
    """
    session_ids = []
    try:
        # Ensure the tmp directory exists
        ensure_tmp_directory(settings.DB_FILE)

        # Connect to the database
        conn = sqlite3.connect(settings.DB_FILE)
        cursor = conn.cursor()

        # Check if the agent_sessions table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='agent_sessions';
        """)
        table_exists = cursor.fetchone()

        if table_exists:
            # Fetch all session IDs
            cursor.execute(
                """
                SELECT session_id FROM agent_sessions
                WHERE user_id = ?
            """,
                (username,),
            )
            session_ids = [row[0] for row in cursor.fetchall()]
        else:
            print(
                f"{Fore.YELLOW}Warning: Table 'agent_sessions' does not exist in {settings.DB_FILE}{Fore.RESET}"
            )

        conn.close()

    except sqlite3.Error as e:
        print(f"{Fore.RED}Database error: {e}{Fore.RESET}")
    except Exception as e:
        print(f"{Fore.RED}Unexpected error while fetching sessions: {e}{Fore.RESET}")

    return gr.update(choices=session_ids[::-1])


def load_sessions_history(
    session_id: str,
    username: str,
    chat_event_metadata_radio: str,
    latex_mode_radio: str
) -> Tuple[list, gr.update]:
    """
    Load the conversation history for a specific session from the database.

    This function retrieves the stored chat runs from the `agent_sessions` table
    for the given `session_id` and `username`. It reconstructs the chat history
    as a list of `ChatMessage` objects. If `chat_event_metadata_radio` is "ON",
    tool call metadata (tool name, arguments, and execution time) is included.

    Execution times for tool calls are calculated using Unix timestamps from
    tool-related events. Duplicate messages (same role and content) are removed
    while preserving order.

    Args:
        session_id (str): The session identifier whose history should be loaded.
        username (str): The user ID associated with the session.
        chat_event_metadata_radio (str): "ON" to include tool metadata, otherwise ignored.
        latex_mode_radio (str): "ON" to eneble standard LaTeX and "OFF" for default.

    Returns:
        Tuple[List[ChatMessage], gr.update]:
            - List of ChatMessage objects representing the session history.
            - Gradio update object to adjust the chatbot container height.
            Returns empty history with a default height if the session does not exist
            or an error occurs.
    """
    
    if latex_mode_radio == "ON":
        # Standard LaTeX
        latex_var = [
            {"left": "$$", "right": "$$", "display": True},     
            {"left": "$", "right": "$", "display": False},      
            {"left": r"\[", "right": r"\]", "display": True},   
            {"left": r"\(", "right": r"\)", "display": False},  
        ]
    else:
        # Default Chatbot LaTeX
        latex_var = [
            {"left": "$$", "right": "$$", "display": True},        
        ]
        
    try:
        ensure_tmp_directory(settings.DB_FILE)
        conn = sqlite3.connect(settings.DB_FILE)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT runs FROM agent_sessions 
            WHERE session_id = ? AND user_id = ?
        """,
            (session_id, username),
        )
        result = cursor.fetchone()
        conn.close()

        if result is None or not result[0]:
            return ([], gr.update(min_height=350, max_height=350))

        memory = json.loads(json.loads(result[0]))
        runs = memory
        chat_history = []

        for run in runs:
            messages = run.get("messages", [])
            events = run.get("events", [])

            # Create a mapping tool_call_id -> (start_timestamp, end_timestamp)
            tool_times = {}
            for event in events:
                tool_call_id = event.get("tool", {}).get("tool_call_id")
                if not tool_call_id:
                    continue
                if tool_call_id not in tool_times:
                    tool_times[tool_call_id] = {"start": None, "end": None}

                if event["event"] == "ToolCallStarted":
                    tool_times[tool_call_id]["start"] = event["created_at"]
                elif event["event"] == "ToolCallCompleted":
                    tool_times[tool_call_id]["end"] = event["created_at"]

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                tool_name = msg.get("tool_name")
                tool_args = msg.get("tool_args")
                tool_call_id = msg.get("tool_call_id")

                # Show tool metadata only if enabled
                if chat_event_metadata_radio == "ON" and role == "tool":
                    exec_time = "N/A"
                    if tool_call_id in tool_times:
                        start = tool_times[tool_call_id]["start"]
                        end = tool_times[tool_call_id]["end"]
                        if start is not None and end is not None:
                            exec_time = f"{end - start:.3f}s"

                    # ToolCallStarted
                    custom_content_started = (
                        f"\n**Tool:** `{tool_name}`\n**Arguments:** `{tool_args}`\n"
                    )
                    chat_history.append(
                        ChatMessage(
                            role="assistant",
                            content=custom_content_started,
                            metadata={"title": "🛠️ ToolCallStarted"},
                        )
                    )

                    # ToolCallCompleted
                    if isinstance(content, list):
                        content = "\n".join(str(item).strip() for item in content if item)
                    elif not isinstance(content, str):
                        content = str(content)
                    result_text = content.strip() if content else ""
                    custom_content_completed = (
                        f"**Results:**\n{result_text}\n**Execution time:** {exec_time}"
                    )
                    chat_history.append(
                        ChatMessage(
                            role="assistant",
                            content=custom_content_completed,
                            metadata={"title": "🛠️ ToolCallCompleted"},
                        )
                    )

                elif role in ["user", "assistant"]:
                    chat_history.append(
                        ChatMessage(
                            role=role, content=content if content is not None else ""
                        )
                    )
                    
                # Trigger LaTeX on message by renew session history
                if latex_mode_radio == "ON":
                     for msg in reversed(chat_history):
                        if msg.role == "assistant":
                            msg.content += " "
                            break
                        
        # Remove duplicates while maintaining order
        seen = set()
        unique_history = []
        for msg in chat_history:
            identifier = (msg.role, msg.content)
            if identifier not in seen:
                seen.add(identifier)
                unique_history.append(msg)

        return (unique_history, gr.update(min_height=650, max_height=650, latex_delimiters=latex_var))

    except sqlite3.Error as e:
        print(f"{Fore.RED}Database error loading session history: {e}{Fore.RESET}")
        return ([], gr.update(min_height=350, max_height=350))
    except Exception as e:
        print(f"{Fore.RED}Unexpected error loading session history: {e}{Fore.RESET}")
        return ([], gr.update(min_height=350, max_height=350))


def delete_session_from_db(
    session_id: str, username: str
) -> Tuple[gr.update, list, gr.update]:
    """
    Delete a specific chat session from the database and return updated session IDs.

    This function removes the session with the given `session_id` for the specified
    `username` from the `agent_sessions` table. It ensures the database directory exists,
    handles potential errors, and prints informative messages. After deletion, it
    returns updated session IDs and UI updates for Gradio components.

    Args:
        session_id (str): The session ID to delete.
        username (str): The user ID associated with the session.

    Returns:
        Tuple: A tuple containing:
            - Updated list of session IDs for radio sessions.
            - An empty list to clear the chat history.
            - Gradio update object to adjust the chat container height.
    """
    try:
        if not session_id:
            print(f"{Fore.YELLOW}No session selected for deletion{Fore.RESET}")
            gr.Warning("No session selected for deletion.", duration=2)
            return (
                get_session_ids_from_db(username),          # Update session_radio
                [],                                         # Clean chatbot
                gr.update(min_height=350, max_height=350),  # Adjust chatbot layout
            )

        ensure_tmp_directory(settings.DB_FILE)
        conn = sqlite3.connect(settings.DB_FILE)
        cursor = conn.cursor()

        # Delete the session
        cursor.execute(
            """
            DELETE FROM agent_sessions
            WHERE session_id = ? AND user_id = ?
        """,
            (session_id, str(username)),
        )
        conn.commit()
        conn.close()

        print(f"{Fore.GREEN}Deleted session: {session_id}{Fore.RESET}")       
        gr.Success("Deleted session.", duration=2)

        return (
            get_session_ids_from_db(username),              # Update session_radio
            [],                                             # Clean chatbot
            gr.update(min_height=350, max_height=350),      # Adjust chatbot layout
        )

    except sqlite3.Error as e:
        print(f"{Fore.RED}Database error deleting session: {e}{Fore.RESET}")
        gr.Error("Database error deleting session.", duration=2)
        return (
            get_session_ids_from_db(username),              
            [],                                             
            gr.update(min_height=350, max_height=350),      
        )
    except Exception as e:
        print(f"{Fore.RED}Unexpected error deleting session: {e}{Fore.RESET}")
        gr.Error("Unexpected error deleting session.", duration=2)
        return (
            get_session_ids_from_db(username),              
            [],                                             
            gr.update(min_height=350, max_height=350),      
        )


def delete_all_sessions_from_db(username: str) -> Tuple[gr.update, list, gr.update]:
    """
    Delete all chat sessions for a specific user from the database.

    This function removes all sessions associated with the given `username`
    from the `agent_sessions` table. It ensures the database directory exists,
    handles errors, prints informative messages, and returns updated session
    information for UI components.

    Args:
        username (str): The user ID whose sessions will be deleted.

    Returns:
        Tuple: A tuple containing:
            - Gradio update object for the session radio (empty).
            - An empty list to clear the chatbot history.
            - Gradio update object to hide the modal.
    """
    try:
        ensure_tmp_directory(settings.DB_FILE)
        conn = sqlite3.connect(settings.DB_FILE)
        cursor = conn.cursor()

        # Delete all sessions for the given user_id
        cursor.execute(
            """
            DELETE FROM agent_sessions
            WHERE user_id = ?
        """,
            (username,),
        )
        conn.commit()
        conn.close()

        print(f"{Fore.GREEN}Deleted ALL sessions{Fore.RESET}")
        gr.Success("Deleted ALL sessions.", duration=2)
        
        return (
            gr.update(choices=[], value=None),
            [],
            gr.update(visible=False),
        )

    except sqlite3.Error as e:
        print(f"{Fore.RED}Database error deleting all sessions: {e}{Fore.RESET}")
        gr.Error("Database error deleting all sessions.", duration=2)
        
        return (
            get_session_ids_from_db(settings.DB_FILE),
            [],
            gr.update(visible=False),
        )
    except Exception as e:
        print(f"{Fore.RED}Unexpected error deleting all sessions: {e}{Fore.RESET}")
        gr.Error("Unexpected error deleting all sessions.", duration=2)
        
        return (
            get_session_ids_from_db(settings.DB_FILE),
            [],
            gr.update(visible=False),
        )


def get_unique_session_id() -> str:
    """
    Generate a unique session identifier.

    This function creates a unique ID by combining the current timestamp
    (formatted as "DD-MM-YYYY_HH:MM:SS") with a shortened UUID. The resulting
    ID can be used to uniquely identify user sessions in the system.

    Returns:
        str: A unique session identifier string.
    """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    parts = str(uuid.uuid4()).split("-")
    short_hash = "-".join(parts[:4])
    unique_id = f"{timestamp}___{short_hash}"
    return unique_id


def clear_and_start_new_session() -> Tuple[str, str, None, gr.update, gr.update, None]:
    """
    Reset the chat interface and initialize a new session.

    This function clears previous messages and session data, generates a
    new unique session ID, and returns UI update objects to reset the chat
    container height and interface elements.

    Returns:
        Tuple: A tuple containing:
            - Empty string for clear chatbot.
            - Empty string for messages.
            - None for session radio.
            - New unique session ID.
            - Gradio update object to reset chatbot height.
            - None for upload file gradio component.
    """
    return (
        "",
        "",
        None,
        get_unique_session_id(),
        gr.update(min_height=350, max_height=350),
        None,
    )


def init_user_and_sessions(request: gr.Request) -> Tuple[str, list, gr.update]:
    """
    Initialize the authenticated user and load their sessions.

    This function retrieves the username from the Gradio request object,
    fetches the session IDs associated with the user from the database,
    and generates a markdown HTML snippet displaying the user's name.

    Args:
        request (gr.Request): The Gradio request object containing user information.

    Returns:
        Tuple:
            - username (str): The authenticated user's name.
            - session_ids (list): List of session IDs for the user.
            - Gradio update object: Markdown update displaying the user's name.
    """
    username = request.username
    session_ids = get_session_ids_from_db(username)

    markdown_html = f"<h3><b>{username}</b></h3>"

    return (username, session_ids, gr.update(value=markdown_html))
