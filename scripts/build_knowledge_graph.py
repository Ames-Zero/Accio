"""
Phase 2: Knowledge Graph Builder
Populates Neo4j with extracted content, generates embeddings, and extracts triplets using Gemini.
"""

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
from neo4j import GraphDatabase
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini configuration (only for triplet extraction)
genai.configure(api_key=GEMINI_API_KEY)
TRIPLET_MODEL = genai.GenerativeModel('gemini-2.5-flash')

# Local embedding model (no API calls!)
EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')  # 768 dims, matches Neo4j index

# Chunking parameters
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens

# Rate limiting (15 RPM for free tier)
API_DELAY = 4  # seconds between API calls


class KnowledgeGraphBuilder:
    """Build Neo4j knowledge graph from extracted PDFs"""
    
    def __init__(self):
        """Initialize Neo4j connection and Gemini API"""
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
        
        # Verify Gemini API key
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        logger.info("Gemini API configured")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        logger.info("Closed Neo4j connection")
    
    def setup_schema(self):
        """Create Neo4j schema with constraints and indexes"""
        logger.info("Setting up Neo4j schema...")
        
        with self.driver.session() as session:
            # Drop existing constraints/indexes if they exist
            try:
                session.run("DROP CONSTRAINT company_id IF EXISTS")
                session.run("DROP CONSTRAINT document_id IF EXISTS")
                session.run("DROP CONSTRAINT chunk_id IF EXISTS")
                session.run("DROP CONSTRAINT entity_name IF EXISTS")
                session.run("DROP INDEX chunk_embeddings IF EXISTS")
            except:
                pass
            
            # Create constraints for uniqueness
            constraints = [
                "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint already exists or failed: {e}")
            
            # Create vector index for embeddings (768 dimensions for Gemini text-embedding-004)
            vector_index_query = """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
              }
            }
            """
            try:
                session.run(vector_index_query)
                logger.info("Created vector index for chunk embeddings")
            except Exception as e:
                logger.warning(f"Vector index creation failed: {e}")
            
            # Create full-text index for text search
            try:
                session.run("""
                CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.text]
                """)
                logger.info("Created full-text index")
            except Exception as e:
                logger.warning(f"Full-text index creation failed: {e}")
        
        logger.info("Schema setup complete")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
        
        Returns:
            List of text chunks
        """
        # Simple word-based chunking (approximation of tokens)
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            i += CHUNK_SIZE - CHUNK_OVERLAP
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using local sentence-transformers model (NO API CALL!)
        
        Args:
            text: Input text
        
        Returns:
            768-dimensional embedding vector
        """
        try:
            # Local inference - instant, no API calls!
            embedding = EMBEDDING_MODEL.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 768
    
    def extract_triplets(self, text: str, layout_type: str = "paragraph") -> List[Dict[str, str]]:
        """
        Extract subject-predicate-object triplets using Gemini
        
        Args:
            text: Input text chunk
            layout_type: Layout type context (title, paragraph, list, etc.)
        
        Returns:
            List of triplets with evidence
        """
        # Skip very short text (likely headers, page numbers, etc.)
        if len(text.strip()) < 50:
            return []
        
        prompt = f"""You are extracting structured knowledge from document text.

Extract subject-predicate-object triplets (knowledge facts) from the following text.
Each triplet should represent a clear factual relationship.

Examples:
- "Apple Inc. was founded in 1976" → {{"subject": "Apple Inc.", "predicate": "founded_in", "object": "1976", "evidence": "Apple Inc. was founded in 1976"}}
- "The payment terms are Net 30 days" → {{"subject": "payment terms", "predicate": "are", "object": "Net 30 days", "evidence": "The payment terms are Net 30 days"}}

Rules:
1. Only extract facts explicitly stated in the text
2. Do NOT infer or make assumptions
3. If no clear facts exist, return an empty array: []
4. Return valid JSON only, no markdown

TEXT:
{text}

Return JSON array: [{{"subject": "...", "predicate": "...", "object": "...", "evidence": "..."}}]"""

        try:
            # Rate limiting
            time.sleep(API_DELAY)
            
            response = TRIPLET_MODEL.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Clean markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines)
            
            response_text = response_text.strip()
            
            # Try to parse JSON
            triplets = json.loads(response_text)
            
            if isinstance(triplets, list):
                return triplets
            else:
                logger.warning(f"Unexpected response format: {type(triplets)}")
                return []
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Response: {response_text[:200]}")
            return []
        except Exception as e:
            logger.error(f"Triplet extraction failed: {e}")
            return []
    
    def load_companies(self):
        """Load company metadata from companies.yaml"""
        companies_file = DATA_DIR / "companies.yaml"
        
        with open(companies_file, 'r') as f:
            data = yaml.safe_load(f)
        
        return data['companies']
    
    def create_company_nodes(self):
        """Create Company nodes in Neo4j"""
        companies = self.load_companies()
        
        with self.driver.session() as session:
            for company in companies:
                session.run("""
                MERGE (c:Company {id: $id})
                SET c.name = $name, c.description = $description
                """, 
                id=company['id'],
                name=company['name'],
                description=company['description'])
                
                logger.info(f"Created Company node: {company['name']}")
    
    def process_document(self, json_path: Path, company_id: str):
        """
        Process a single document JSON file and populate Neo4j
        
        Args:
            json_path: Path to processed JSON file
            company_id: Company identifier
        """
        logger.info(f"Processing document: {json_path.name}")
        
        # Load extracted data
        with open(json_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        doc_id = doc_data['doc_id']
        
        with self.driver.session() as session:
            # Create Document node
            session.run("""
            MERGE (d:Document {doc_id: $doc_id})
            SET d.name = $filename, d.company_id = $company_id, d.total_pages = $total_pages
            WITH d
            MATCH (c:Company {id: $company_id})
            MERGE (d)-[:BELONGS_TO]->(c)
            """,
            doc_id=doc_id,
            filename=doc_data['filename'],
            company_id=company_id,
            total_pages=doc_data['total_pages'])
            
            logger.info(f"Created Document node: {doc_id}")
            
            # Process each page
            for page_data in doc_data['pages']:
                self.process_page(session, page_data, doc_id, company_id)
    
    def process_page(self, session, page_data: Dict, doc_id: str, company_id: str):
        """
        Process a single page and create chunks, entities, relations
        
        Args:
            session: Neo4j session
            page_data: Page data dictionary
            doc_id: Document identifier
            company_id: Company identifier
        """
        page_num = page_data['page_num']
        image_path = page_data['image_path']
        
        # Create Page node
        session.run("""
        MERGE (p:Page {page_num: $page_num, doc_id: $doc_id})
        SET p.image_path = $image_path
        WITH p
        MATCH (d:Document {doc_id: $doc_id})
        MERGE (p)-[:IN_DOCUMENT]->(d)
        """,
        page_num=page_num,
        doc_id=doc_id,
        image_path=image_path)
        
        logger.info(f"Processing page {page_num} of {doc_id}")
        
        # Process layout blocks (text chunks)
        for block_idx, block in enumerate(page_data['layout_blocks']):
            if not block['text'].strip():
                continue
            
            # Chunk the text
            chunks = self.chunk_text(block['text'])
            
            for chunk_idx, chunk_text in enumerate(chunks):
                self.process_chunk(
                    session,
                    chunk_text,
                    f"{doc_id}_p{page_num}_b{block_idx}_c{chunk_idx}",
                    page_num,
                    doc_id,
                    company_id,
                    block['type'],
                    block['bbox'],
                    block['confidence']
                )
        
        # Process tables
        for table in page_data['tables']:
            self.process_table(session, table, page_num, doc_id, company_id)
        
        # Process figures
        for figure in page_data['figures']:
            self.process_figure(session, figure, page_num, doc_id)
    
    def process_chunk(
        self,
        session,
        text: str,
        chunk_id: str,
        page_num: int,
        doc_id: str,
        company_id: str,
        layout_type: str,
        bbox: List[float],
        confidence: float
    ):
        """
        Process a text chunk: create node, generate embedding, extract triplets
        
        Args:
            session: Neo4j session
            text: Chunk text
            chunk_id: Unique chunk identifier
            page_num: Page number
            doc_id: Document identifier
            company_id: Company identifier
            layout_type: Layout type (title, paragraph, etc.)
            bbox: Bounding box coordinates
            confidence: Confidence score
        """
        logger.info(f"Processing chunk: {chunk_id}")
        
        # Generate embedding
        embedding = self.generate_embedding(text)
        
        # Create Chunk node with embedding
        session.run("""
        MERGE (ch:Chunk {chunk_id: $chunk_id})
        SET ch.text = $text,
            ch.embedding = $embedding,
            ch.page_num = $page_num,
            ch.doc_id = $doc_id,
            ch.company_id = $company_id,
            ch.layout_type = $layout_type,
            ch.bbox = $bbox,
            ch.confidence = $confidence
        WITH ch
        MATCH (p:Page {page_num: $page_num, doc_id: $doc_id})
        MERGE (ch)-[:FROM_PAGE]->(p)
        """,
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        page_num=page_num,
        doc_id=doc_id,
        company_id=company_id,
        layout_type=layout_type,
        bbox=bbox,
        confidence=confidence)
        
        # Extract triplets
        triplets = self.extract_triplets(text, layout_type)
        
        if len(triplets) > 0:
            logger.info(f"✓ Extracted {len(triplets)} triplets from chunk {chunk_id}")
        else:
            logger.debug(f"No triplets extracted from chunk {chunk_id} (text length: {len(text)})")
        
        # Create entities and relations
        for triplet in triplets:
            self.create_triplet(
                session,
                triplet,
                chunk_id,
                doc_id,
                company_id
            )
    
    def create_triplet(
        self,
        session,
        triplet: Dict[str, str],
        chunk_id: str,
        doc_id: str,
        company_id: str
    ):
        """
        Create entity nodes and relation from a triplet
        
        Args:
            session: Neo4j session
            triplet: Dictionary with subject, predicate, object, evidence
            chunk_id: Source chunk identifier
            doc_id: Document identifier
            company_id: Company identifier
        """
        subject = triplet.get('subject', '').strip()
        predicate = triplet.get('predicate', '').strip()
        obj = triplet.get('object', '').strip()
        evidence = triplet.get('evidence', '')
        
        if not (subject and predicate and obj):
            return
        
        # Create entity nodes
        session.run("""
        MERGE (s:Entity {name: $subject, company_id: $company_id})
        SET s.canonical_name = $subject
        MERGE (o:Entity {name: $object, company_id: $company_id})
        SET o.canonical_name = $object
        
        WITH s, o
        MATCH (ch:Chunk {chunk_id: $chunk_id})
        MERGE (s)-[:MENTIONED_IN]->(ch)
        MERGE (o)-[:MENTIONED_IN]->(ch)
        
        WITH s, o
        MERGE (s)-[r:RELATES_TO {relation: $predicate, company_id: $company_id}]->(o)
        SET r.evidence = $evidence,
            r.confidence = 0.9,
            r.source_type = 'text',
            r.doc_id = $doc_id
        """,
        subject=subject,
        object=obj,
        predicate=predicate,
        evidence=evidence,
        chunk_id=chunk_id,
        doc_id=doc_id,
        company_id=company_id)
    
    def process_table(
        self,
        session,
        table: Dict,
        page_num: int,
        doc_id: str,
        company_id: str
    ):
        """
        Process a table: create Table node and extract row-wise triplets
        
        Args:
            session: Neo4j session
            table: Table data dictionary
            page_num: Page number
            doc_id: Document identifier
            company_id: Company identifier
        """
        table_id = table['table_id']
        
        # Create Table node
        session.run("""
        MERGE (t:Table {table_id: $table_id})
        SET t.page_num = $page_num,
            t.doc_id = $doc_id,
            t.headers = $headers,
            t.bbox = $bbox
        WITH t
        MATCH (p:Page {page_num: $page_num, doc_id: $doc_id})
        MERGE (t)-[:ON_PAGE]->(p)
        """,
        table_id=table_id,
        page_num=page_num,
        doc_id=doc_id,
        headers=table['headers'],
        bbox=table['bbox'])
        
        # Extract facts from table rows
        headers = table['headers']
        for row_idx, row in enumerate(table['rows']):
            # Create row text for triplet extraction
            row_text = " | ".join([f"{h}: {v}" for h, v in zip(headers, row) if h and v])
            
            if row_text.strip():
                # Generate chunk ID for table row
                chunk_id = f"{doc_id}_p{page_num}_{table_id}_row{row_idx}"
                
                # Process as chunk
                self.process_chunk(
                    session,
                    row_text,
                    chunk_id,
                    page_num,
                    doc_id,
                    company_id,
                    "table_row",
                    table['bbox'],
                    0.95
                )
    
    def process_figure(self, session, figure: Dict, page_num: int, doc_id: str):
        """
        Process a figure: create Figure node
        
        Args:
            session: Neo4j session
            figure: Figure data dictionary
            page_num: Page number
            doc_id: Document identifier
        """
        session.run("""
        MERGE (f:Figure {figure_id: $figure_id})
        SET f.caption = $caption,
            f.page_num = $page_num,
            f.doc_id = $doc_id,
            f.bbox = $bbox
        WITH f
        MATCH (p:Page {page_num: $page_num, doc_id: $doc_id})
        MERGE (f)-[:ON_PAGE]->(p)
        """,
        figure_id=figure['figure_id'],
        caption=figure['caption'],
        page_num=page_num,
        doc_id=doc_id,
        bbox=figure['bbox'])
    
    def build_graph(self):
        """Main function to build complete knowledge graph"""
        logger.info("="*60)
        logger.info("BUILDING KNOWLEDGE GRAPH - Phase 2")
        logger.info("="*60)
        
        # Setup schema
        self.setup_schema()
        
        # Create company nodes
        self.create_company_nodes()
        
        # Process each company's documents
        companies = self.load_companies()
        
        for company in companies:
            company_id = company['id']
            processed_dir = PROCESSED_DIR / company_id
            
            if not processed_dir.exists():
                logger.warning(f"No processed files found for {company_id}")
                continue
            
            json_files = sorted(list(processed_dir.glob("*.json")))
            logger.info(f"Found {len(json_files)} documents for {company_id}")
            
            # Process documents with progress bar
            with tqdm(json_files, desc=f"Processing {company_id}", unit="doc") as pbar:
                for json_file in pbar:
                    try:
                        pbar.set_postfix({"current": json_file.stem})
                        self.process_document(json_file, company_id)
                    except Exception as e:
                        logger.error(f"Failed to process {json_file.name}: {e}", exc_info=True)
        
        logger.info("="*60)
        logger.info("KNOWLEDGE GRAPH BUILD COMPLETE")
        logger.info("="*60)


def main():
    """Main execution function"""
    builder = KnowledgeGraphBuilder()
    
    try:
        builder.build_graph()
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
    finally:
        builder.close()


if __name__ == "__main__":
    main()
