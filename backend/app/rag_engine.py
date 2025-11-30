"""
RAG Engine - Hybrid Retrieval and Answer Generation
"""

import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from app.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    GEMINI_API_KEY,
    GEMINI_PRO_MODEL,
    TOP_K_CHUNKS,
    VECTOR_SIMILARITY_THRESHOLD
)
from app.models import Citation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini (only for answer generation)
genai.configure(api_key=GEMINI_API_KEY)

# Load local embedding model (same as used in graph building)
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')  # 768 dims
logger.info("Loaded local embedding model: all-mpnet-base-v2")


class RAGEngine:
    """Retrieval-Augmented Generation Engine"""
    
    def __init__(self):
        """Initialize Neo4j connection and Gemini models"""
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.gemini_model = genai.GenerativeModel(GEMINI_PRO_MODEL)
        logger.info("RAG Engine initialized")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for search query using LOCAL model (no API call!)
        
        Args:
            query: User question
        
        Returns:
            768-dimensional embedding vector
        """
        try:
            # Local embedding generation - instant, no API call!
            embedding = EMBEDDING_MODEL.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * 768
    
    def vector_search(
        self, 
        query_embedding: List[float], 
        company_id: str,
        top_k: int = TOP_K_CHUNKS
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in Neo4j
        
        Args:
            query_embedding: Query embedding vector
            company_id: Company to filter by
            top_k: Number of results to return
        
        Returns:
            List of matching chunks with metadata
        """
        query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
        YIELD node AS chunk, score
        WHERE chunk.company_id = $company_id AND score >= $threshold
        MATCH (chunk)-[:FROM_PAGE]->(page:Page)-[:IN_DOCUMENT]->(doc:Document)
        RETURN 
            chunk.chunk_id AS chunk_id,
            chunk.text AS text,
            chunk.page_num AS page_num,
            chunk.bbox AS bbox,
            doc.doc_id AS doc_id,
            doc.name AS doc_name,
            page.image_path AS image_path,
            score
        ORDER BY score DESC
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                query_embedding=query_embedding,
                company_id=company_id,
                top_k=top_k,
                threshold=VECTOR_SIMILARITY_THRESHOLD
            )
            
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "page_num": record["page_num"],
                    "doc_id": record["doc_id"],
                    "doc_name": record["doc_name"],
                    "image_path": record["image_path"],
                    "bbox": record["bbox"],
                    "score": record["score"]
                })
            
            return chunks
    
    def fulltext_search(
        self,
        query: str,
        company_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform full-text search in Neo4j
        
        Args:
            query: Search query
            company_id: Company to filter by
            top_k: Number of results to return
        
        Returns:
            List of matching chunks
        """
        cypher_query = """
        CALL db.index.fulltext.queryNodes('chunk_text_index', $query)
        YIELD node AS chunk, score
        WHERE chunk.company_id = $company_id
        MATCH (chunk)-[:FROM_PAGE]->(page:Page)-[:IN_DOCUMENT]->(doc:Document)
        RETURN 
            chunk.chunk_id AS chunk_id,
            chunk.text AS text,
            chunk.page_num AS page_num,
            chunk.bbox AS bbox,
            doc.doc_id AS doc_id,
            doc.name AS doc_name,
            page.image_path AS image_path,
            score
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher_query,
                    query=query,
                    company_id=company_id,
                    top_k=top_k
                )
                
                chunks = []
                for record in result:
                    chunks.append({
                        "chunk_id": record["chunk_id"],
                        "text": record["text"],
                        "page_num": record["page_num"],
                        "doc_id": record["doc_id"],
                        "doc_name": record["doc_name"],
                        "image_path": record["image_path"],
                        "bbox": record["bbox"],
                        "score": record["score"]
                    })
                
                return chunks
        except Exception as e:
            logger.warning(f"Full-text search failed: {e}")
            return []
    
    def graph_traversal_search(
        self,
        query: str,
        company_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search using entity relations (graph traversal)
        
        Args:
            query: Search query
            company_id: Company to filter by
            top_k: Number of results to return
        
        Returns:
            List of relevant chunks via entity relations
        """
        # Extract potential entity names from query (simple word extraction)
        words = query.split()
        entity_patterns = [w for w in words if len(w) > 3]  # Simple heuristic
        
        if not entity_patterns:
            return []
        
        cypher_query = """
        UNWIND $patterns AS pattern
        MATCH (e:Entity)
        WHERE e.company_id = $company_id 
            AND toLower(e.name) CONTAINS toLower(pattern)
        MATCH (e)-[:RELATES_TO*1..2]-(related:Entity)
        MATCH (related)-[:MENTIONED_IN]->(chunk:Chunk)
        MATCH (chunk)-[:FROM_PAGE]->(page:Page)-[:IN_DOCUMENT]->(doc:Document)
        RETURN DISTINCT
            chunk.chunk_id AS chunk_id,
            chunk.text AS text,
            chunk.page_num AS page_num,
            chunk.bbox AS bbox,
            doc.doc_id AS doc_id,
            doc.name AS doc_name,
            page.image_path AS image_path,
            1.0 AS score
        LIMIT $top_k
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    cypher_query,
                    patterns=entity_patterns,
                    company_id=company_id,
                    top_k=top_k
                )
                
                chunks = []
                for record in result:
                    chunks.append({
                        "chunk_id": record["chunk_id"],
                        "text": record["text"],
                        "page_num": record["page_num"],
                        "doc_id": record["doc_id"],
                        "doc_name": record["doc_name"],
                        "image_path": record["image_path"],
                        "bbox": record["bbox"],
                        "score": record["score"]
                    })
                
                return chunks
        except Exception as e:
            logger.warning(f"Graph traversal search failed: {e}")
            return []
    
    def hybrid_retrieval(
        self,
        query: str,
        company_id: str
    ) -> List[Dict[str, Any]]:
        """
        Combine vector search, full-text search, and graph traversal
        
        Args:
            query: User question
            company_id: Company to filter by
        
        Returns:
            Top-k most relevant chunks (deduplicated)
        """
        logger.info(f"Hybrid retrieval for query: '{query}' in {company_id}")
        
        # 1. Generate query embedding
        query_embedding = self.generate_query_embedding(query)
        
        # 2. Vector similarity search
        vector_results = self.vector_search(query_embedding, company_id)
        logger.info(f"Vector search: {len(vector_results)} results")
        
        # 3. Full-text search
        fulltext_results = self.fulltext_search(query, company_id)
        logger.info(f"Full-text search: {len(fulltext_results)} results")
        
        # 4. Graph traversal
        graph_results = self.graph_traversal_search(query, company_id)
        logger.info(f"Graph traversal: {len(graph_results)} results")
        
        # 5. Merge and deduplicate by chunk_id
        all_results = {}
        
        # Prioritize vector search results (highest weight)
        for chunk in vector_results:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in all_results:
                all_results[chunk_id] = chunk
                all_results[chunk_id]["combined_score"] = chunk["score"] * 1.0
        
        # Add full-text results (medium weight)
        for chunk in fulltext_results:
            chunk_id = chunk["chunk_id"]
            if chunk_id in all_results:
                all_results[chunk_id]["combined_score"] += chunk["score"] * 0.5
            else:
                all_results[chunk_id] = chunk
                all_results[chunk_id]["combined_score"] = chunk["score"] * 0.5
        
        # Add graph results (lower weight)
        for chunk in graph_results:
            chunk_id = chunk["chunk_id"]
            if chunk_id in all_results:
                all_results[chunk_id]["combined_score"] += 0.3
            else:
                all_results[chunk_id] = chunk
                all_results[chunk_id]["combined_score"] = 0.3
        
        # Sort by combined score and return top-k
        ranked_results = sorted(
            all_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:TOP_K_CHUNKS]
        
        logger.info(f"Hybrid retrieval: {len(ranked_results)} final results")
        return ranked_results
    
    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate grounded answer using Gemini with retrieved chunks
        
        Args:
            question: User question
            chunks: Retrieved context chunks
        
        Returns:
            Generated answer with citation markers
        """
        if not chunks:
            return "I cannot find any relevant information in the available documents to answer this question."
        
        # Build evidence context with citation markers
        evidence_parts = []
        for i, chunk in enumerate(chunks, start=1):
            evidence_parts.append(f"[{i}] (from {chunk['doc_name']}, page {chunk['page_num']})")
            evidence_parts.append(chunk['text'])
            evidence_parts.append("")  # Blank line
        
        evidence_text = "\n".join(evidence_parts)
        
        # Construct grounding prompt
        prompt = f"""Answer the question using ONLY the provided evidence below. Cite sources using [1], [2], etc.
If the evidence is insufficient to answer the question, say "I cannot determine from available documents."

QUESTION: {question}

EVIDENCE:
{evidence_text}

ANSWER (with citations):"""

        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip()
            logger.info(f"Generated answer: {answer[:100]}...")
            return answer
        
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I encountered an error while generating the answer. Please try again."
    
    def query(self, question: str, company_id: str) -> tuple[str, List[Citation]]:
        """
        Main query function: retrieve + generate answer
        
        Args:
            question: User question
            company_id: Company to search within
        
        Returns:
            Tuple of (answer, citations)
        """
        # Retrieve relevant chunks
        chunks = self.hybrid_retrieval(question, company_id)
        
        # Generate answer
        answer = self.generate_answer(question, chunks)
        
        # Build citations
        citations = []
        for chunk in chunks:
            citation = Citation(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                page_num=chunk["page_num"],
                doc_id=chunk["doc_id"],
                doc_name=chunk["doc_name"],
                image_path=chunk["image_path"],
                bbox=chunk["bbox"],
                score=chunk.get("combined_score", chunk.get("score", 0.0))
            )
            citations.append(citation)
        
        return answer, citations
