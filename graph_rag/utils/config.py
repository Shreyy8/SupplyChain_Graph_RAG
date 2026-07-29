# graph_rag/utils/config.py
#
# PURPOSE:
#   This file is the single source of truth for all configuration in the
#   FINSIGHT_GraphRAG pipeline. Every module that needs a setting — an API
#   key, a file path, a numeric threshold — imports it from here.
#
#   Nothing is hardcoded anywhere else in the project. If you need to change
#   a setting, you change it in .env and this file picks it up automatically.
#
# HOW IT WORKS:
#   1. python-dotenv reads the .env file from the project root and loads
#      every key-value pair into the process environment variables.
#   2. pydantic-settings reads those environment variables and maps them
#      onto the fields of the Settings class below.
#   3. pydantic validates that every required field is present and that
#      the types are correct (e.g. CHUNK_SIZE must be an integer, not a string).
#   4. If any required field is missing, pydantic raises a clear error
#      before the pipeline starts — much better than a cryptic KeyError later.
#
# DO YOU RUN THIS FILE?
#   No. You never run this file directly.
#   Other modules import the `settings` object at the bottom of this file:
#       from graph_rag.utils.config import settings
#   Then access values like:  settings.openai_api_key
#                             settings.neo4j_uri
#                             settings.chunk_size


from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from graph_rag.utils import logger


class Settings(BaseSettings):
    """
    Holds all configuration values for the FINSIGHT_GraphRAG pipeline.

    pydantic-settings automatically reads each field's value from the
    corresponding environment variable. The field name maps to the env
    var name case-insensitively — so `openai_api_key` reads OPENAI_API_KEY.

    Field() lets us set a default value and a description for each setting.
    If no default is provided, the field is required — the app will refuse
    to start if that environment variable is missing from .env.
    """

    # -------------------------------------------------------------------------
    # OPENAI
    # -------------------------------------------------------------------------

    # Your OpenAI API key — required, no default
    # Used for GPT-4o (entity extraction + answer synthesis) and embeddings
    openai_api_key: str = Field(..., description="OpenAI API key")

    # Which GPT model to use for entity extraction and answer synthesis
    # Default: gpt-4o — best accuracy for entity/relationship extraction
    llm_model: str = Field(default="gpt-4o", description="OpenAI chat model name")

    # Which embedding model to use for converting text chunks into vectors
    # Default: text-embedding-3-small — fast and cost-effective for demos
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name"
    )

    # -------------------------------------------------------------------------
    # NEO4J AURADB
    # -------------------------------------------------------------------------

    # The AuraDB connection URI — looks like neo4j+s://xxxxxxxx.databases.neo4j.io
    # Required — no default because this is unique to your AuraDB instance
    neo4j_uri: str = Field(..., description="Neo4j AuraDB connection URI")

    # AuraDB username — always 'neo4j' for AuraDB, but we read it from .env
    # so it can be overridden if needed
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")

    # AuraDB password — required, no default
    neo4j_password: str = Field(..., description="Neo4j password")

    # -------------------------------------------------------------------------
    # DATA PATHS
    # -------------------------------------------------------------------------

    # Folder containing the five synthetic PDF documents
    # Path is relative to the project root folder
    pdf_dir: Path = Field(
        default=Path("./data/pdfs"),
        description="Directory containing input PDF files"
    )

    # Folder where ChromaDB will store its vector index files on disk
    # ChromaDB creates this folder automatically if it does not exist
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        description="Directory where ChromaDB persists its index files"
    )

    # Name of the ChromaDB collection that stores our document chunk embeddings
    # Think of a collection like a table — all five PDFs get chunked and stored here
    chroma_collection_name: str = Field(
        default="finsight_chunks",
        description="ChromaDB collection name for document chunks"
    )

    # -------------------------------------------------------------------------
    # CHUNKING CONFIGURATION
    # -------------------------------------------------------------------------

    # Number of characters per text chunk
    # 800 characters is roughly 120-150 words — about half a paragraph
    chunk_size: int = Field(
        default=800,
        description="Number of characters per text chunk"
    )

    # Number of characters of overlap between consecutive chunks
    # Overlap prevents sentences at chunk boundaries from being cut off
    chunk_overlap: int = Field(
        default=150,
        description="Number of overlapping characters between consecutive chunks"
    )

    # -------------------------------------------------------------------------
    # RETRIEVAL CONFIGURATION
    # -------------------------------------------------------------------------

    # How many text chunks ChromaDB returns per query
    # These chunks are combined with graph traversal results for the final answer
    top_k_vector: int = Field(
        default=5,
        description="Number of chunks ChromaDB returns per query"
    )

    # How many relationship hops the graph query agent traverses in Neo4j
    # 3 hops covers: entity -> relation -> relation -> relation
    # This is sufficient to cover the ASML -> TSMC -> Apple chain in our data
    top_k_graph_hops: int = Field(
        default=3,
        description="Number of hops for Neo4j graph traversal"
    )

    # -------------------------------------------------------------------------
    # PYDANTIC SETTINGS CONFIGURATION
    # -------------------------------------------------------------------------

    model_config = SettingsConfigDict(
        # Tell pydantic-settings to read from a .env file
        env_file=".env",

        # If the .env file does not exist, do not raise an error —
        # environment variables may be set directly in the shell instead
        env_file_encoding="utf-8",

        # Ignore any extra environment variables that are not defined
        # as fields above — avoids errors from unrelated env vars
        extra="ignore",
    )


# -----------------------------------------------------------------------------
# SINGLE GLOBAL INSTANCE
# -----------------------------------------------------------------------------
# We create one Settings instance here at module load time.
# Every other module imports this single `settings` object.
#
# This means .env is read exactly once when the application starts,
# not on every request. If a required value is missing, the error
# is raised immediately at startup — not buried in a request handler.
#
# Usage in other modules:
#   from graph_rag.utils.config import settings
#   print(settings.neo4j_uri)
#   print(settings.chunk_size)

settings = Settings()