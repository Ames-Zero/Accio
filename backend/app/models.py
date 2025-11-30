"""
Pydantic Models for API
"""

from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request model for /query endpoint"""
    question: str
    company_id: str


class Citation(BaseModel):
    """Citation model with source information"""
    chunk_id: str
    text: str
    page_num: int
    doc_id: str
    doc_name: str
    image_path: str
    bbox: List[float]
    score: float


class QueryResponse(BaseModel):
    """Response model for /query endpoint"""
    answer: str
    citations: List[Citation]


class Company(BaseModel):
    """Company model"""
    id: str
    name: str
    description: str


class CompaniesResponse(BaseModel):
    """Response model for /companies endpoint"""
    companies: List[Company]
