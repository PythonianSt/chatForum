<img width="3070" height="1579" alt="image" src="https://github.com/user-attachments/assets/d9681c89-b976-4230-b73a-a15339d29fab" /><img width="2303" height="1170" alt="image" src="https://github.com/user-attachments/assets/89519b84-9d79-48da-8c0e-b4f7ac78aff4" />
0. AI Ethics
  The webpage https://www.agnoshealth.com/forums is permitted for code testing. 
1. Model Pipeline Construction Overview
- Agnos Health Forum Chatbot
  A Retrieval-Augmented Generation (RAG) chatbot that answers health questions using data from Agnos Health forums.
- Project Overview
  This system transforms Thai health forum discussions into an intelligent Q&A assistant using semantic search and natural language processing.
- Model Pipeline Architecture
- End-to-End Data Flow

Forum URLs
↓
Web Scraping (aiohttp + BeautifulSoup)
↓
Content Processing & Cleaning
↓
TF-IDF Vectorization (sklearn)
↓
Semantic Search Index (FAISS)
↓
Query Processing & Retrieval
↓
Answer Generation & Source Attribution
↓
Streamlit Chat Interface

2. Pipeline Construction Steps:
1) Data Acquisition & Preparation**
# URL Collection
- Input: threads.txt with forum URLs
- Format: One URL per line with Thai characters
# Concurrent Web Scraping
- Technology: aiohttp for async requests
- Parser: BeautifulSoup for HTML extraction
- Output: scraped_threads.json
# Text Preprocessing
- Thai character preservation
- Whitespace normalization
- Special character removal
2) Vector Database Construction
# TF-IDF Vectorization
- Features: 5000 most important terms
- N-grams: Unigrams + Bigrams for Thai phrases
- Output: Sparse document-term matrix
# Index Building
- Algorithm: Cosine similarity search
- Storage: FAISS for efficient retrieval
- Metadata: Thread titles, URLs, content snippets
3) Query Processing & Retrieval
# Real-time Search
- Query preprocessing (same as documents)
- TF-IDF transformation
- Top-k similarity ranking
- Confidence scoring (0-1 scale)
# Answer Generation
- Most relevant content selection
- Source attribution with URLs
- Medical disclaimer inclusion
# Streamlit UI
- Real-time chat interface
- Example questions sidebar
- Source citation display
- Confidence score visualization

3. Technical Implementation
Core Technologies
Web Scraping: aiohttp, BeautifulSoup4
NLP: scikit-learn TF-IDF, cosine similarity
Vector Search: FAISS
UI: Streamlit
Language: Thai text processing

4. Performance Metrics
Indexing Time: 2-5 minutes (first run)
Query Response: < 2 seconds
Accuracy: Based on cosine similarity (0-1 scale)
Scalability: 1000+ threads tested

5. Future Enhancements
LLM integration for answer refinement
User feedback collection
Advanced Thai NLP (WordVec, BERT)
Multi-language support
Real-time forum updates
