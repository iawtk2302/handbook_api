from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import uvicorn
import logging
import os

# Import the RAG agent from the existing file
from rag_agent import rag_agent, load_knowledge

# Setup logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Agent API",
    description="API for interacting with a RAG (Retrieval Augmented Generation) agent",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Request model for querying the RAG agent"""
    query: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "RAG Agent API is running. Use /query endpoint to interact with the agent."}


async def chat_response_streamer(message: str) -> AsyncGenerator:
    """
    Stream agent responses chunk by chunk.

    Args:
        message: User message to process

    Yields:
        Text chunks from the agent response
    """
    run_response = await rag_agent.arun(message, stream=True)
    async for chunk in run_response:
        # chunk.content contains the text response from the Agent
        yield f"data: {chunk.content}\n\n"


@app.post("/query")
async def query_agent(request: QueryRequest):
    """
    Sends a query to the RAG agent and returns the response.

    Args:
        request: QueryRequest containing the query and optional parameters

    Returns:
        Either a streaming response or the complete agent response
    """
    logger.debug(f"QueryRequest: {request}")

    try:
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                chat_response_streamer(request.query),
                media_type="text/event-stream",
            )
        else:
            # Return complete response
            response = await rag_agent.arun(
                request.query,
                stream=False
            )

            # Extract sources if available
            sources = []
            if hasattr(response, 'metadata') and response.metadata and 'sources' in response.metadata:
                sources = response.metadata['sources']

            # Return the content and sources
            return {
                "answer": response.content,
                "sources": sources
            }
    except Exception as e:
        logger.error(f"Error querying RAG agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying RAG agent: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/reload_knowledge")
async def reload_knowledge():
    """Endpoint to reload the knowledge base"""
    try:
        await load_knowledge()
        return {"message": "Knowledge base reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reloading knowledge base: {str(e)}"
        )

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Get port from environment variable with a fallback to 8000
    port = int(os.environ.get("PORT", 8000))

    # Run the API server using Uvicorn
    # Bind to 0.0.0.0 for Render deployment
    logger.info(f"Starting API server on port {port}...")
    uvicorn.run("rag_agent_api:app", host="0.0.0.0", port=port, reload=True)
