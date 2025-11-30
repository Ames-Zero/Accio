"""
FastAPI Application - Main Entry Point
"""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yaml
import logging

from app.models import QueryRequest, QueryResponse, CompaniesResponse, Company
from app.rag_engine import RAGEngine
from app.config import API_HOST, API_PORT

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Knowledge Q&A API",
    description="Retrieve and answer questions from PDF knowledge base",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine (singleton)
rag_engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    rag_engine = RAGEngine()
    logger.info("FastAPI application started")


@app.on_event("shutdown")
async def shutdown_event():
    """Close RAG engine on shutdown"""
    global rag_engine
    if rag_engine:
        rag_engine.close()
    logger.info("FastAPI application shutdown")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "PDF Knowledge Q&A API is running",
        "endpoints": ["/query", "/companies", "/docs"]
    }


@app.get("/companies", response_model=CompaniesResponse)
async def get_companies():
    """
    Get list of available companies
    
    Returns:
        CompaniesResponse with list of companies
    """
    try:
        # Load companies from YAML
        companies_file = Path(__file__).parent.parent.parent / "data" / "companies.yaml"
        
        with open(companies_file, 'r') as f:
            data = yaml.safe_load(f)
        
        companies = [
            Company(**company) for company in data['companies']
        ]
        
        return CompaniesResponse(companies=companies)
    
    except Exception as e:
        logger.error(f"Failed to load companies: {e}")
        raise HTTPException(status_code=500, detail="Failed to load companies")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question using RAG
    
    Args:
        request: QueryRequest with question and company_id
    
    Returns:
        QueryResponse with answer and citations
    """
    try:
        logger.info(f"Query received: '{request.question}' for {request.company_id}")
        
        # Validate company_id
        companies_response = await get_companies()
        valid_company_ids = [c.id for c in companies_response.companies]
        
        if request.company_id not in valid_company_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid company_id. Valid options: {valid_company_ids}"
            )
        
        # Execute query
        answer, citations = rag_engine.query(request.question, request.company_id)
        
        logger.info(f"Query completed: {len(citations)} citations")
        
        return QueryResponse(answer=answer, citations=citations)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
