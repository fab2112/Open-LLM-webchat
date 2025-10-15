# Open-LLM-webchat - Agent settings
import settings

from agno.agent import Agent
from utils.models import get_model
from agno.db.sqlite import SqliteDb
from utils.process_knowledge import load_base_knowledge_to_agent


async def get_agent(
    knowledge_base_selector: str,
    knowledge_base_path: str,
    selected_model: str,
    temperature: int,
    top_p: int,
    top_k: int,
    max_tokens_number: int,
    short_term_memory_selector: str,
    long_term_memory_selector: str,
    short_memory_history_runs: int,
    chat_event_reasoning: str | None,
):
    """
    Initialize and return a configured Agent instance.

    This function loads the knowledge base, sets up short-term and long-term memory options,
    initializes the selected model, and enables reasoning if applicable.

    Args:
        knowledge_base_selector (str): Enables ("ON") or disables ("OFF") knowledge base.
        knowledge_base_path (str): Path to the knowledge base files.
        selected_model (str): Name of the model to use.
        temperature (int): Sampling temperature controlling randomness.
        top_p (int): Nucleus sampling parameter (probability threshold).
        top_k (int): Maximum number of tokens considered during sampling.
        max_tokens_number (int): Maximum number of tokens generated (input + output).
        short_term_memory_selector (str): Enables ("ON") or disables ("OFF") short-term memory.
        long_term_memory_selector (str): Enables ("ON") or disables ("OFF") long-term memory.
        short_memory_history_runs (int): Number of past interactions stored in short-term memory.
        chat_event_reasoning (str | None): Enables reasoning mode if set to "Agent".

    Returns:
        Agent: A fully configured Agent instance.
    """

    # Enable | Disable short memory
    if short_term_memory_selector == "ON":
        add_history_to_context = True
    else:
        add_history_to_context = False

    # Enable | Disable long memory
    if long_term_memory_selector == "ON":
        enable_user_memories = True
    else:
        enable_user_memories = False

    # Set agent reasoning
    if chat_event_reasoning and chat_event_reasoning[0] == "Agent":
        reasonig_var = True
    else:
        reasonig_var = False

    pdf_knowledge_base, search_knowledge = await load_base_knowledge_to_agent(
        knowledge_base_selector=knowledge_base_selector
    )

    model = get_model(selected_model, temperature, top_p, top_k, max_tokens_number)

    # Configure SqliteDb with specific tables
    db = SqliteDb(
        db_file=settings.DB_FILE,
        session_table="agent_sessions",  
        memory_table="user_memories",  
    )

    agent = Agent(
        model=model,
        # Main Data Base
        db=db,
        knowledge=pdf_knowledge_base,
        search_knowledge=search_knowledge,
        # Short Memory
        add_history_to_context=add_history_to_context,
        num_history_runs=short_memory_history_runs,
        # Long Memory
        enable_user_memories=enable_user_memories,
        add_datetime_to_context=True,
        markdown=True,
        debug_mode=settings.DEBUG_MODE,
        instructions=[
            """
            Use the search_knowledge_base tool to search knowledge in base only when requested.
            Use the tools available for current or factual issues.
            """
        ],
        store_events=True,
        reasoning=reasonig_var,
    )

    return agent
