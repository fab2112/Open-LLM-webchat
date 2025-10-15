# Open-LLM-webchat - Process the knowledge base (RAG - PDF)
import gradio as gr

from agno.db.sqlite import SqliteDb
from agno.vectordb.chroma import ChromaDb
from utils.models import get_embedding_model
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.chunking.recursive import RecursiveChunking


# Set knowledge metadata data base
contents_db = SqliteDb(
    db_file="tmp/knowledge_contents.db", knowledge_table="knowledge_contents"
)

# Set vector data base
vector_db = ChromaDb(
    path="tmp/chromadb",
    persistent_client=True,
    collection="agno_docs",
    embedder=get_embedding_model(),
)

# Set pdf reader
pdf_reader = PDFReader(
    name="Custom PDF Reader",
    chunking_strategy=RecursiveChunking(chunk_size=1000, overlap=100),
)

async def upload_file_to_base_knowledge(knowledge_base_selector: str, knowledge_base_path: str):
    """
    Asynchronously upload a file to the knowledge base if enabled.

    If the knowledge base selector is set to "ON" and a valid file path is provided,
    this function adds the file content to the knowledge base using the configured
    PDF reader and databases.

    Args:
        knowledge_base_selector (str): Enables ("ON") or disables ("OFF") the knowledge base upload.
        knowledge_base_path (str): Path to the file to be added and processed in knowledge base.

    Returns:
        None: This function does not return a value.
    """
    if knowledge_base_path and knowledge_base_selector == "ON":
        knowledge = Knowledge(vector_db=vector_db, contents_db=contents_db)

        await knowledge.add_content_async(
            path=knowledge_base_path,
            reader=pdf_reader,
            skip_if_exists=True,
        )
        return gr.Success("File uploaded to knowledge base successfully.", duration=3)

    return gr.Info("File not uploaded, please enable knowledge base and upload the file..", duration=3)


async def load_base_knowledge_to_agent(knowledge_base_selector: str):
    """
    Load the knowledge base for the agent if enabled.

    If the knowledge base selector is set to "ON", this function initializes
    the knowledge base using the configured databases and enables knowledge search.

    Args:
        knowledge_base_selector (str): Enables ("ON") or disables ("OFF") loading of the knowledge base.

    Returns:
        tuple: A tuple containing:
            - knowledge (Knowledge | None): The loaded knowledge base instance or None.
            - search_knowledge (bool): True if knowledge search is enabled, otherwise False.
    """
    if knowledge_base_selector == "ON":
        knowledge = Knowledge(vector_db=vector_db, contents_db=contents_db)
        search_knowledge = True

    else:
        knowledge = None
        search_knowledge = False

    return knowledge, search_knowledge
