# Open-LLM-webchat - Models
import os
import settings

from dotenv import load_dotenv
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.models.google import Gemini
from agno.models.nvidia import Nvidia
from agno.models.openai import OpenAIChat
from agno.knowledge.embedder.google import GeminiEmbedder


# Load .env
load_dotenv()


def get_model(
    llm_type: str, llm_temp: float, llm_top_p: float, llm_top_k: float, max_tokens: int
) -> object:
    """
    Initialize and return a language model instance based on the selected provider.

    This function detects the model provider (Google, Groq, Cerebras, or Ollama)
    from the given `llm_type` and returns the corresponding configured model object.

    Args:
        llm_type (str): Model type identifier.
        llm_temp (float): Sampling temperature controlling randomness.
        llm_top_p (float): Nucleus sampling parameter (probability threshold).
        llm_top_k (float): Maximum number of tokens considered during sampling.
        max_tokens (int): Maximum number of tokens generated (input + output).

    Returns:
        object: A configured model instance for the selected LLM provider.
    """

    # OpenAI
    if llm_type.startswith("OpenAI"):
        model = llm_type.split("_", 1)[1]
        return OpenAIChat(
            id=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=llm_temp,
            top_p=llm_top_p,
            max_retries=3,
            max_completion_tokens=max_tokens,
        )

    # Google
    if llm_type.startswith("Google"):
        model = llm_type.split("_", 1)[1]
        return Gemini(
            id=model,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=llm_temp,
            top_p=llm_top_p,
            top_k=llm_top_k,
            max_output_tokens=max_tokens,
        )

    # Groq
    elif llm_type.startswith("Groq"):
        model = llm_type.split("_", 1)[1]
        return Groq(
            id=model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=llm_temp,
            top_p=llm_top_p,
            max_retries=3,
            max_tokens=max_tokens,
        )

    # Nvidia
    elif llm_type.startswith("Nvidia"):
        model = llm_type.split("_", 1)[1]
        return Nvidia(
            id=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=llm_temp,
            top_p=llm_top_p,
            max_retries=3,
            max_completion_tokens=max_tokens,
        )

    # Ollama
    elif llm_type.startswith("Ollama"):
        model = llm_type.split("_", 1)[1]
        return Ollama(id=model, host=settings.OLLAMA_URL)


def get_embedding_model() -> object:
    """
    Initialize and return the embedding model instance.

    This function creates a Gemini embedding model with predefined ID and dimensions,
    using the API key from the environment variables.

    Returns:
        GeminiEmbedder: Configured embedding model instance.
    """

    embedding_model = GeminiEmbedder(
        id="gemini-embedding-001",
        dimensions=768,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    return embedding_model
