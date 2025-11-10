"""
Simplified RAG System for Agnos Health Forum
Uses TF-IDF instead of sentence transformers to avoid dependency issues
"""

import pandas as pd
import numpy as np
import faiss
import pickle
import json
from typing import List, Dict
import re
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ThreadScraper:
    def __init__(self, max_concurrent=3):
        self.max_concurrent = max_concurrent
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def scrape_thread(self, url: str) -> Dict:
        """Scrape individual thread content"""
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract thread title
                    title = soup.find('h1') or soup.find('title')
                    title_text = title.get_text().strip() if title else "No title"
                    
                    # Try to find main content area
                    content_selectors = [
                        'article', 'main', '.content', '.post-content', 
                        '.thread-content', '.forum-content', '.entry-content'
                    ]
                    
                    content_parts = []
                    for selector in content_selectors:
                        elements = soup.select(selector)
                        for elem in elements:
                            # Remove script and style elements
                            for script in elem(["script", "style"]):
                                script.decompose()
                            text = elem.get_text().strip()
                            if text and len(text) > 50:
                                content_parts.append(text)
                    
                    # Fallback: get all paragraphs
                    if not content_parts:
                        paragraphs = soup.find_all('p')
                        content_parts = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30]
                    
                    content = ' '.join(content_parts) if content_parts else title_text
                    
                    return {
                        'url': url,
                        'title': title_text,
                        'content': content,
                        'content_length': len(content),
                    }
                else:
                    return {'url': url, 'error': f'HTTP {response.status}', 'content': ''}
                    
        except Exception as e:
            return {'url': url, 'error': str(e), 'content': ''}
    
    async def scrape_threads(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple threads concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_scrape(url):
            async with semaphore:
                return await self.scrape_thread(url)
        
        tasks = [bounded_scrape(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and empty content
        valid_results = []
        for result in results:
            if isinstance(result, dict) and result.get('content') and len(result.get('content', '')) > 50:
                valid_results.append(result)
        
        print(f"Successfully scraped {len(valid_results)} out of {len(urls)} threads")
        return valid_results

class AgnosRAG:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.metadata = []
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Thai characters and basic punctuation
        text = re.sub(r'[^\w\s\u0E00-\u0E7F.,!?\-]', ' ', text)
        return text.strip().lower()
    
    def build_index(self, documents: List[Dict]):
        """Build TF-IDF index from documents"""
        self.documents = []
        self.metadata = []
        
        texts = []
        for doc in documents:
            # Combine title and content for better retrieval
            combined_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            cleaned_text = self.preprocess_text(combined_text)
            
            if len(cleaned_text) > 50:  # Minimum text length
                texts.append(cleaned_text)
                self.documents.append(cleaned_text)
                self.metadata.append({
                    'url': doc.get('url', ''),
                    'title': doc.get('title', ''),
                    'original_content': doc.get('content', '')[:1000]  # Keep longer snippet
                })
        
        print(f"Building TF-IDF index with {len(texts)} documents...")
        
        if not texts:
            raise ValueError("No valid documents to index")
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=1,
            max_df=0.8,
            ngram_range=(1, 2)  # Use unigrams and bigrams
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print("Index built successfully!")
    
    def save_index(self, index_path: str):
        """Save TF-IDF index and metadata"""
        with open(f"{index_path}_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        with open(f"{index_path}_matrix.pkl", 'wb') as f:
            pickle.dump(self.tfidf_matrix, f)
        
        with open(f"{index_path}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
    
    def load_index(self, index_path: str):
        """Load TF-IDF index and metadata"""
        with open(f"{index_path}_vectorizer.pkl", 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(f"{index_path}_matrix.pkl", 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        
        with open(f"{index_path}_metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents using TF-IDF"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k results
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include results with some similarity
                results.append({
                    'score': float(similarities[idx]),
                    'content': self.documents[idx],
                    'metadata': self.metadata[idx]
                })
        
        return results

class AgnosHealthRAG:
    def __init__(self, rag_system: AgnosRAG):
        self.rag = rag_system
        
    def answer_question(self, question: str, k: int = 3) -> Dict:
        """Answer question using RAG"""
        # Retrieve relevant documents
        results = self.rag.search(question, k=k)
        
        if not results:
            return {
                'answer': 'ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในฟอรัม Agnos Health\n\nกรุณาลองใช้คำถามอื่นหรือถามคำถามให้ละเอียดมากขึ้น',
                'sources': [],
                'confidence': 0.0
            }
        
        # Use the most relevant result as answer
        best_result = results[0]
        
        # Format answer nicely
        answer_parts = []
        answer_parts.append(f"จากข้อมูลในฟอรัม Agnos Health:\n\n")
        
        # Use the content (limited to reasonable length)
        content = best_result['content']
        if len(content) > 800:
            content = content[:800] + "..."
        answer_parts.append(content)
        
        answer_parts.append(f"\n\nนี่เป็นข้อมูลจากกระทู้ในฟอรัม กรุณาปรึกษาแพทย์สำหรับการวินิจฉัยและการรักษาที่ถูกต้อง")
        
        answer = "".join(answer_parts)
        
        return {
            'answer': answer,
            'sources': [
                {
                    'title': result['metadata']['title'],
                    'url': result['metadata']['url'],
                    'score': result['score'],
                    'content_snippet': result['metadata']['original_content'][:300] + "..."
                }
                for result in results
            ],
            'confidence': float(best_result['score'])
        }

# Utility functions
async def scrape_thread_contents(thread_urls_file: str, output_file: str):
    """Scrape all thread contents from URLs file"""
    # Read URLs from file
    with open(thread_urls_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Scraping {len(urls)} threads...")
    
    async with ThreadScraper(max_concurrent=3) as scraper:
        results = await scraper.scrape_threads(urls)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Scraped {len(results)} threads successfully")
    return results

def build_rag_system(thread_urls_file: str, scraped_data_file: str, index_path: str):
    """Build the complete RAG system"""
    import asyncio
    
    # Scrape data if not exists
    try:
        with open(scraped_data_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"Loaded {len(documents)} existing documents")
    except:
        print("Scraping thread contents...")
        documents = asyncio.run(scrape_thread_contents(thread_urls_file, scraped_data_file))
    
    if not documents:
        raise ValueError("No documents could be scraped. Please check the URLs in threads.txt")
    
    # Build RAG system
    rag_system = AgnosRAG()
    rag_system.build_index(documents)
    rag_system.save_index(index_path)
    
    # Create QA system
    qa_system = AgnosHealthRAG(rag_system)
    return qa_system

def load_rag_system(index_path: str):
    """Load existing RAG system"""
    rag_system = AgnosRAG()
    rag_system.load_index(index_path)
    qa_system = AgnosHealthRAG(rag_system)
    return qa_system

# For testing
if __name__ == "__main__":
    # Test the system
    try:
        qa_system = build_rag_system("threads.txt", "scraped_threads.json", "agnos_health_index")
        
        # Test questions
        test_questions = [
            "กระเพาะปัสสาวะอักเสบ",
            "น้ำในหูไม่เท่ากัน",
            "ปวดท้องประจำเดือน",
            "สุขภาพจิต"
        ]
        
        for question in test_questions:
            print(f"\nQ: {question}")
            result = qa_system.answer_question(question)
            print(f"A: {result['answer'][:200]}...")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Sources: {len(result['sources'])}")
            
    except Exception as e:
        print(f"Error: {e}")