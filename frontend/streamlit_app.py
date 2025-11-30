"""
Streamlit Chat Interface for PDF Knowledge Q&A
"""

import streamlit as st
import requests
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List, Dict, Any
import io

# Backend API configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="PDF/OCR Processing for Workflow digitisation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root for image paths - go up from frontend/ to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_companies() -> List[Dict[str, str]]:
    """Fetch available companies from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/companies")
        response.raise_for_status()
        data = response.json()
        return data['companies']
    except Exception as e:
        st.error(f"Failed to load companies: {e}")
        return []


def query_api(question: str, company_id: str) -> Dict[str, Any]:
    """Send query to backend API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "company_id": company_id}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Query failed: {e}")
        return {"answer": "Error processing query", "citations": []}


def highlight_bbox(image_path: str, bbox: List[float]) -> Image.Image:
    """
    Draw red bounding box on screenshot
    
    Args:
        image_path: Path to image file (relative to project root)
        bbox: [x0, y0, x1, y1] coordinates in PDF points (72 DPI)
    
    Returns:
        PIL Image with highlighted bbox
    """
    try:
        # Construct absolute path - handle both relative and absolute paths
        if image_path.startswith('/'):
            abs_path = Path(image_path)
        else:
            # Resolve the path relative to PROJECT_ROOT and make it absolute
            abs_path = (PROJECT_ROOT / image_path).resolve()
        
        # Verify file exists
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found at: {abs_path}")
        
        # Open image
        img = Image.open(abs_path)
        
        # Draw rectangle
        draw = ImageDraw.Draw(img)
        
        # Ensure bbox has valid coordinates
        if bbox and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
            # Scale bbox from PDF coordinates (72 DPI) to screenshot coordinates (300 DPI)
            scale_factor = 300 / 72  # 4.166667
            scaled_bbox = [coord * scale_factor for coord in bbox]
            
            # Draw red rectangle with thicker line
            draw.rectangle(scaled_bbox, outline="red", width=5)
        
        return img
    
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        # Return blank image as fallback
        return Image.new('RGB', (800, 600), color='white')


def display_citations(citations: List[Dict[str, Any]]):
    """
    Display citations with screenshots and text
    
    Args:
        citations: List of citation dictionaries from API
    """
    if not citations:
        st.info("No citations available")
        return
    
    st.markdown("### 📎 Sources")
    
    for i, citation in enumerate(citations, start=1):
        with st.expander(f"Citation [{i}] - {citation['doc_name']}, Page {citation['page_num']}", expanded=(i == 1)):
            # Two column layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Document Screenshot:**")
                try:
                    # Highlight and display image
                    highlighted_img = highlight_bbox(citation['image_path'], citation['bbox'])
                    st.image(highlighted_img, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {e}")
            
            with col2:
                st.markdown("**Citation Details:**")
                st.markdown(f"**Document:** {citation['doc_name']}")
                st.markdown(f"**Page:** {citation['page_num']}")
                st.markdown(f"**Relevance Score:** {citation['score']:.3f}")
                st.markdown("**Text Excerpt:**")
                st.text_area(
                    label="excerpt",
                    value=citation['text'][:500] + ("..." if len(citation['text']) > 500 else ""),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )


def main():
    """Main Streamlit application"""
    
    # App title
    st.title("📚 PDF/OCR Processor")
    st.markdown("Ask questions about company documents and get answers with citations")
    
    # Sidebar - Company selection
    st.sidebar.header("🏢 Select Company")
    
    companies = get_companies()
    
    if not companies:
        st.sidebar.error("No companies available. Check backend connection.")
        return
    
    # Company selector
    company_options = {c['name']: c for c in companies}
    selected_company_name = st.sidebar.selectbox(
        "Company",
        options=list(company_options.keys()),
        index=0
    )
    
    selected_company = company_options[selected_company_name]
    
    # Display company info
    st.sidebar.markdown(f"**{selected_company['name']}**")
    st.sidebar.caption(selected_company['description'])
    
    st.sidebar.markdown("---")
    
    # Example questions
    st.sidebar.markdown("### 💡 Example Questions")
    example_questions = [
        "What happened to NSF's Total Liabilities in FY 2011?",
        "What is the NNI's approach to addressing ethical questions?",
        "Where must accessible parking spaces be located?",
        "What are nanotechnology's potential environmental impacts?",
        "Summarize NSF's financial position"
    ]
    
    for eq in example_questions:
        if st.sidebar.button(eq, key=f"example_{eq}"):
            st.session_state.example_question = eq
    
    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations for assistant messages
            if message["role"] == "assistant" and "citations" in message:
                display_citations(message["citations"])
    
    # Handle example question click
    if 'example_question' in st.session_state:
        user_question = st.session_state.example_question
        del st.session_state.example_question
    else:
        # Chat input
        user_question = st.chat_input("Ask a question about the documents...")
    
    if user_question:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Query API
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = query_api(user_question, selected_company['id'])
            
            answer = result.get("answer", "No answer generated")
            citations = result.get("citations", [])
            
            # Display answer
            st.markdown(answer)
            
            # Display citations
            if citations:
                display_citations(citations)
            
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })
    
    # Sidebar - Chat controls
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("PDF Knowledge Assistant v1.0")
    st.sidebar.caption("Powered by Gemini, Neo4j, and Docling")


if __name__ == "__main__":
    main()
