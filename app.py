"""
Streamlit Chatbot for Agnos Health RAG System
Simplified version without sentence-transformers
"""

import streamlit as st
import json
import os
from rag_system import build_rag_system, load_rag_system
import time

# Page configuration
st.set_page_config(
    page_title="Agnos Health Forum Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D5D5D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E0E0E0;
        white-space: pre-wrap;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #F5F5F5;
        border-left: 4px solid #4CAF50;
    }
    .source-item {
        background-color: #FFF3E0;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #FF9800;
    }
    .confidence-high { color: #4CAF50; }
    .confidence-medium { color: #FF9800; }
    .confidence-low { color: #F44336; }
    .stButton button {
        width: 100%;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_system():
    """Initialize the RAG system"""
    try:
        with st.spinner("กำลังโหลดระบบ Agnos Health Assistant..."):
            # Configuration
            THREAD_URLS_FILE = "threads.txt"
            SCRAPED_DATA_FILE = "scraped_threads.json"
            INDEX_PATH = "agnos_health_index"
            
            # Check if index exists, otherwise build it
            if (os.path.exists(f"{INDEX_PATH}_vectorizer.pkl") and 
                os.path.exists(f"{INDEX_PATH}_matrix.pkl") and 
                os.path.exists(f"{INDEX_PATH}_metadata.pkl")):
                
                st.session_state.rag_system = load_rag_system(INDEX_PATH)
                st.success("✅ ระบบโหลดข้อมูลฟอรัมสำเร็จ!")
                
            else:
                if os.path.exists(THREAD_URLS_FILE):
                    st.info("🔄 กำลังสร้างระบบค้นหาจากข้อมูลฟอรัม... (อาจใช้เวลาสักครู่)")
                    st.session_state.rag_system = build_rag_system(THREAD_URLS_FILE, SCRAPED_DATA_FILE, INDEX_PATH)
                    st.success("✅ สร้างระบบค้นหาจากข้อมูลฟอรัมสำเร็จ!")
                else:
                    st.error("❌ ไม่พบไฟล์ threads.txt กรุณาเพิ่ม URL กระทู้ในไฟล์ก่อน")
                    return False
            
            st.session_state.initialized = True
            return True
            
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดระบบ: {str(e)}")
        return False

def get_confidence_color(confidence):
    """Get color based on confidence score"""
    if confidence > 0.3:
        return "confidence-high"
    elif confidence > 0.1:
        return "confidence-medium"
    else:
        return "confidence-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Agnos Health Forum Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ถามคำถามเกี่ยวกับสุขภาพจากข้อมูลในฟอรัม Agnos Health</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("เกี่ยวกับระบบ")
        st.info("""
        🤖 **Agnos Health Assistant** 
        
        ระบบนี้ใช้ข้อมูลจากฟอรัม Agnos Health 
        เพื่อตอบคำถามเกี่ยวกับสุขภาพโดยอ้างอิงจาก
        การสนทนาและคำแนะนำในฟอรัม
        
        **คุณสามารถถามเกี่ยวกับ:**
        - อาการเจ็บป่วย
        - วิธีการรักษา
        - คำแนะนำสุขภาพ
        - ประสบการณ์จากผู้ใช้
        
        ⚠️ **หมายเหตุ:** 
        นี่เป็นข้อมูลจากฟอรัมเท่านั้น 
        กรุณาปรึกษาแพทย์สำหรับการวินิจฉัยที่ถูกต้อง
        """)
        
        st.header("สถิติ")
        if st.session_state.rag_system:
            st.write(f"📚 จำนวนข้อมูล: {len(st.session_state.rag_system.rag.metadata)} กระทู้")
        st.write(f"💬 จำนวนข้อความ: {len(st.session_state.messages)}")
        
        if st.button("ล้างประวัติการสนทนา"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize system if not done
    if not st.session_state.initialized:
        if initialize_system():
            st.session_state.initialized = True
        else:
            st.stop()
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 สนทนา")
        
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>คุณ:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Bot message with confidence
                confidence_color = get_confidence_color(message.get("confidence", 0))
                confidence_text = f"<span class='{confidence_color}'>ความมั่นใจ: {message.get('confidence', 0):.2f}</span>"
                
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>ผู้ช่วย:</strong> {message["content"]}<br>
                    <small>{confidence_text}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 แหล่งข้อมูลอ้างอิง"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="source-item">
                                <strong>#{i+1}: {source['title']}</strong><br>
                                <small>URL: {source['url']}</small><br>
                                <small>คะแนนความเกี่ยวข้อง: {source['score']:.3f}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💡 คำถามตัวอย่าง")
        example_questions = [
            "กระเพาะปัสสาวะอักเสบรักษาอย่างไร",
            "อาการน้ำในหูไม่เท่ากันเป็นอย่างไร",
            "ปวดท้องประจำเดือนควรทำอย่างไร",
            "โรคซึมเศร้ามีอาการอย่างไร",
            "วิธีการดูแลสุขภาพจิต",
            "อาหารสำหรับผู้ป่วยโรคกระเพาะ"
        ]
        
        for question in example_questions:
            if st.button(question, key=question):
                # Add to chat
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get response
                with st.spinner("กำลังค้นหาข้อมูล..."):
                    response = st.session_state.rag_system.answer_question(question)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response['answer'],
                    "confidence": response['confidence'],
                    "sources": response['sources']
                })
                st.rerun()
    
    # Chat input
    st.markdown("---")
    user_input = st.chat_input("พิมพ์คำถามเกี่ยวกับสุขภาพที่นี่...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("🔍 กำลังค้นหาข้อมูลจากฟอรัม..."):
            response = st.session_state.rag_system.answer_question(user_input)
        
        # Add bot response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response['answer'],
            "confidence": response['confidence'],
            "sources": response['sources']
        })
        
        st.rerun()

if __name__ == "__main__":
    main()