from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge import AgentKnowledge
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.vectordb.qdrant import Qdrant
import requests
from lxml import etree
import os

COLLECTION_NAME = "handbook"


async def load_knowledge():
    """
    Load knowledge base from a sitemap URL

    Args:
        sitemap_url: URL of the sitemap XML
        db_url: Database URL for PostgreSQL

    Returns:
        WebsiteKnowledgeBase: The constructed knowledge base
    """
    # Fetch sitemap XML
    response = requests.get('https://handbook.exelab.asia/sitemap.xml')
    xml_content = response.content

    # Parse XML to extract URLs
    tree = etree.fromstring(xml_content)
    raw_urls = tree.xpath(
        '//ns:loc/text()', namespaces={'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'})

    # Process URLs for knowledge base
    urlsForKnowledgeBase = [u if u.endswith(
        '/') else u + '/' for u in raw_urls if u.rstrip('/') != 'https://handbook.exelab.asia']

    print(urlsForKnowledgeBase)

    # Create the knowledge base with the processed URLs
    knowledge_base = WebsiteKnowledgeBase(
        urls=urlsForKnowledgeBase,
        # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
        vector_db=Qdrant(
            collection=COLLECTION_NAME,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
        ),
    )

    # Properly await the asynchronous call
    await knowledge_base.aload(upsert=True, recreate=False)

rag_agent = Agent(
    name="RAG Agent",
    agent_id="rag-agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    knowledge=AgentKnowledge(vector_db=Qdrant(
        collection=COLLECTION_NAME,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),),
    # Add a tool to search the knowledge base which enables agentic RAG.
    # This is enabled by default when `knowledge` is provided to the Agent.
    search_knowledge=True,
    # Add a tool to read chat history.
    read_chat_history=True,
    # Store the agent sessions in the `ai.rag_agent_sessions` table
    # storage=PostgresStorage(table_name="rag_agent_sessions",
    #                         db_url="postgresql://handbook_qc6v_user:ZhEw01OiYZHcTZAM1LPBHTeTwf45E9Al@dpg-d0iov3qdbo4c738nvh70-a.oregon-postgres.render.com/handbook_qc6v"),
    instructions=[
        "Always search your knowledge base first and use it if available.",
        "Share the page number or source URL of the information you used in your response.",
        "If health benefits are mentioned, include them in the response.",
        "Important: Use tables where possible.",
    ],
    markdown=True,
)
