# Open-LLM-webchat - Process runtime chat messages
import asyncio
import settings
import traceback
import gradio as gr

from colorama import Fore
from gradio import ChatMessage
from typing import List, Tuple
from agno.agent import RunEvent
from utils.agent import get_agent
from agno.tools.mcp import MultiMCPTools
from agno.tools.reasoning import ReasoningTools
from utils.process_session import get_session_ids_from_db
from typing import AsyncGenerator, Tuple, List, Dict, Any
from utils.process_mcp import load_mcps_stdioserverparameters

# Globals variables
running_agent = None
current_run_id = None

# Load MCPs server parameters 
stdio_server_params = load_mcps_stdioserverparameters()

async def get_response(
    user_msg: str,
    chat_history: List[ChatMessage],
    selected_model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens_number: int,
    session_radio_id: str,
    unique_session_id: str,
    chat_event_metadata_radio: bool,
    chat_stream_radio: bool,
    file_uploaded: str | None,
    knowledge_base_selector: str,
    short_term_memory_selector: str,
    long_term_memory_selector: str,
    short_memory_history_runs: int,
    username: str,
    chat_event_reasoning: str,
    mcp_tools_radio: str
) -> AsyncGenerator[Tuple[Any, Any, Any, Any, Any], None]:
    """
    Handle user input and generate responses from the AI agent asynchronously.

    This function manages the chat flow, including session handling, memory configuration,
    streaming or non-streaming responses, and optional integration with reasoning,
    and MCP tools. It updates the chat interface dynamically as messages
    and events occur.

    Args:
        user_msg (str): The message entered by the user.
        chat_history (List[ChatMessage]): The conversation history.
        selected_model (str): The model identifier to be used by the agent.
        temperature (float): Sampling temperature controlling response randomness.
        top_p (float): Nucleus sampling parameter (probability threshold).
        top_k (int): Maximum number of tokens considered during sampling.
        max_tokens_number (int): Maximum number of tokens generated (input + output).
        session_radio_id (str): The current chat session id identifier.
        unique_session_id (str): The unique session id generated.
        chat_event_metadata_radio (bool): Enables or disables tool metadata display in chat.
        chat_stream_radio (bool): Enables or disables streaming mode for responses.
        file_uploaded (str | None): Path to a user-uploaded knowledge base file (if any).
        knowledge_base_selector (str): Enables ("ON") or disables ("OFF") knowledge base.
        short_term_memory_selector (str): Enables ("ON") or disables ("OFF") short-term memory.
        long_term_memory_selector (str): Enables ("ON") or disables ("OFF") long-term memory.
        short_memory_history_runs (int): Number of recent interactions stored in short-term memory.
        username (str): The username logged.
        chat_event_reasoning (str): Enables reasoning mode when set to "Agent" or "Tool".
        mcp_tools_radio (str): Enables ("ON") or disables ("OFF") MCP tool integration.

    Yields:
        Tuple[Any, Any, Any, Any, Any]: UI updates for Gradio components including:
            - Updated text input field,
            - Updated chat history,
            - Session list,
            - Button stop stream visibility,
            - Chatbot adjustments.

    """
    
    # Set username
    if not username:
        username = "user_default"

    # Set global variables
    global running_agent, current_run_id
    
    # Set first messages
    chat_history.append(ChatMessage(role="user", content=user_msg)) 
    chat_history.append(ChatMessage(role="assistant", content="### ..."))
    
    # First yield to show user message and empty response from assistant
    yield (
        gr.update(placeholder="...", value="", interactive=False),
        chat_history,
        get_session_ids_from_db(username),
        gr.update(visible=False),
        gr.update(min_height=650, max_height=650),
    )

    # Use the selected session ID if available, otherwise use the unique session ID
    selected_session = session_radio_id if session_radio_id else str(unique_session_id)
    knowledge_base_path = file_uploaded if file_uploaded else None
    
    # Set agent
    agent = await get_agent(
        knowledge_base_selector=knowledge_base_selector,
        knowledge_base_path=knowledge_base_path,
        selected_model=selected_model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens_number=max_tokens_number,
        short_term_memory_selector=short_term_memory_selector,
        long_term_memory_selector=long_term_memory_selector,
        short_memory_history_runs=short_memory_history_runs,
        chat_event_reasoning=chat_event_reasoning
    )
    
    # Set global variable
    running_agent = agent
    
    # Set tools
    agent.tools = []
    
    # Set Reasoning tool
    if chat_event_reasoning and chat_event_reasoning[0] == "Tool" and mcp_tools_radio == "ON":
        agent.tools.append(ReasoningTools(add_instructions=True))

    try:
        stream_response = ""
        
        # Asynchronous integration of mcps in chat
        async with MultiMCPTools(server_params_list=stdio_server_params, timeout_seconds=20) as mcp_tools:
            
            if mcp_tools_radio == "ON":
                agent.tools.append(mcp_tools)
            
            # Stream chat
            if chat_stream_radio == "ON":
                
                # Load stream chunks responses for agent
                async for chunk in agent.arun(
                    user_msg,
                    user_id=username,
                    session_id=selected_session,
                    stream=True,
                    stream_intermediate_steps=True,
                ):
                    current_run_id = chunk.run_id
                    
                    await asyncio.sleep(settings.STREAM_DELAY)

                    # Tool start event 
                    if chunk.event == RunEvent.tool_call_started:
                        stream_response += "\n"
                        
                        if chat_event_metadata_radio == "ON":
                            
                            if chat_history[-1].content == "### ...":
                                chat_history[-1] = ChatMessage(role="assistant", content="")
                                
                            stream_response += "\n"
                            
                            custom_content = (
                                f"\n**Tool:** `{chunk.tool.tool_name}`\n"
                                f"**Arguments:** `{chunk.tool.tool_args}`\n"
                            )
                            
                            chat_history.append(ChatMessage(
                                role="assistant",
                                content=custom_content,
                                metadata={"title": "🛠️ ToolCallStarted"}
                            ))
                            
                            yield (
                                gr.update(placeholder="...", value="", interactive=False),
                                chat_history,
                                get_session_ids_from_db(username),
                                gr.update(visible=True),
                                gr.update(min_height=650, max_height=650),
                            )
                    
                    # Tool completed event 
                    elif chunk.event == RunEvent.tool_call_completed:
                        if chat_event_metadata_radio == "ON":
                            
                            if chat_history[-1].content == "### ...":
                                chat_history[-1] = ChatMessage(role="assistant", content="")

                            custom_content = (
                                f"\n**Results:** \n{chunk.tool.result}\n"
                                f"**Execution time:** {chunk.tool.metrics.duration:.4f}s" if chunk.tool.metrics else "**Execution time:** N/A\n"
                            )
                            
                            chat_history.append(ChatMessage(
                                role="assistant",
                                content=custom_content,
                                metadata={"title": "🛠️ ToolCallCompleted"}
                            ))
                            
                            chat_history.append(ChatMessage(role="assistant", content=""))
                            
                            stream_response = ""
                            
                            yield (
                                gr.update(placeholder="...", value="", interactive=False),
                                chat_history,
                                get_session_ids_from_db(username),
                                gr.update(visible=True),
                                gr.update(min_height=650, max_height=650),
                            )
                    
                    # Assistant menssages event
                    if chunk.event == RunEvent.run_content:
                        if isinstance(chunk.content, str):
                            
                            stream_response += chunk.content
                            
                            chat_history[-1] = ChatMessage(role="assistant", content=stream_response)
                            
                            yield (
                                gr.update(placeholder="...", value="", interactive=False),
                                chat_history,
                                get_session_ids_from_db(username),
                                gr.update(visible=True),
                                gr.update(min_height=650, max_height=650),
                            )
            
            # Not Stream chat
            elif chat_stream_radio == "OFF":
            
                # Load response for agent
                agent_response = await agent.arun(
                    user_msg,
                    user_id=username,
                    session_id=selected_session,
                    stream=False,
                )

                if chat_history[-1].content == "### ...":
                    chat_history[-1] = ChatMessage(role="assistant", content="")

                # Process assistant messages and tools events
                if chat_event_metadata_radio == "ON":
                    
                    # Find the index of the last message with role='user'
                    last_user_index = max(
                        (i for i, message in enumerate(agent_response.messages) if message.role == "user"),
                        default=-1
                    )

                    # Iterate only through messages after the last 'user'
                    for message in agent_response.messages[last_user_index + 1:]:

                        if message.role == "tool":
                            tool_name = message.tool_name
                            tool_args = message.tool_args
                            exec_time = "N/A"

                            custom_content_started = (
                                f"\n**Tool:** `{tool_name}`\n"
                                f"**Arguments:** `{tool_args}`\n"
                            )
                            
                            chat_history.append(ChatMessage(
                                role="assistant",
                                content=custom_content_started,
                                metadata={"title": "🛠️ ToolCallStarted"}
                            ))

                            result_text = message.content.strip() if message.content else ""
                            custom_content_completed = (
                                f"**Results:**\n{result_text}\n"
                                f"**Execution time:** {exec_time}"
                            )
                            
                            chat_history.append(ChatMessage(
                                role="assistant",
                                content=custom_content_completed,
                                metadata={"title": "🛠️ ToolCallCompleted"}
                            ))

                        elif message.role == "assistant":
                            chat_history.append(ChatMessage(
                                role="assistant",
                                content="" if message.content is None else message.content
                            ))
                
                # Process messages from assistant only 
                else:
                    
                    # Find the index of the last message with role='user'
                    last_user_index = max(
                        (i for i, message in enumerate(agent_response.messages) if message.role == "user"),
                        default=-1
                    )

                    # Iterate only through messages after the last 'user'
                    for message in agent_response.messages[last_user_index + 1:]:

                        if message.role == "assistant":
                            chat_history.append(ChatMessage(
                                role="assistant",
                                content="" if message.content is None else message.content
                            ))
                                              
            # Update sessions after completion
            yield (
                gr.update(placeholder="Ask something...", value="", interactive=True),
                chat_history,
                get_session_ids_from_db(username),
                gr.update(visible=False),
                gr.update(min_height=650, max_height=650),
            )
            
    except Exception as e:
        nome_da_excecao = type(e).__name__        
        track_line = f" Line-{traceback.extract_tb(e.__traceback__)[0].lineno}"     
        menssagem = "Exception: "
        print(f"{Fore.LIGHTRED_EX}{menssagem}{nome_da_excecao}:{track_line}{Fore.RESET}")
        #
        error_msg = "An error occurred while processing your request."
        chat_history[-1] = ChatMessage(role="assistant", content=error_msg)
        yield (
            gr.update(placeholder="Ask something...", value="", interactive=True),
            chat_history,
            get_session_ids_from_db(username),
            gr.update(visible=False),
            gr.update(min_height=650, max_height=650),
        )
        #traceback.print_exc()  
        
    finally:
        running_agent = None
        

def stop_agent_running_stream():
    """Stop the current run streaming generation"""
    global running_agent
    if running_agent is not None:
        running_agent.cancel_run(run_id=current_run_id)
        running_agent = None
        return gr.Info("Stop stream running.")