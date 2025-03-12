import streamlit as st
import os

# 1. PAGE CONFIG - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Capuchin Bird Audio Recognizer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. CUSTOM CSS TO STYLE THE SIDEBAR AND NAVIGATION
custom_css = """
<style>
/* Make the sidebar navigation more attractive */
div[data-testid="stSidebar"] > div:first-child {
    background-color: #1E293B;
}

div[data-testid="stSidebar"] {
    color: white;
}

/* Style the navigation links */
button[kind="secondary"] {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 10px 15px !important;
    margin-bottom: 8px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: background-color 0.3s !important;
}

button[kind="secondary"]:hover {
    background-color: rgba(255, 255, 255, 0.2) !important;
}

/* Hide 'app' text and replace with 'Home' */
section[data-testid="stSidebarUserContent"] div.element-container div.stMarkdown p {
    font-size: 0 !important; 
}

section[data-testid="stSidebarUserContent"] div.element-container div.stMarkdown p:after {
    content: 'Home' !important;
    font-size: 16px !important;
    color: white !important;
    font-weight: bold !important;
}

/* Style the sidebar container */
div[data-testid="stSidebar"] > div:first-child > div:first-child {
    padding-top: 0 !important;
}

/* Style the Pages section header */
p:has(span:contains("Pages")) {
    display: none !important;
}

/* Make the navigation area stand out */
section[data-testid="stSidebarUserContent"] > div:nth-child(2) {
    background-color: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    padding: 10px;
    margin-top: 20px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Add sidebar image and description
with st.sidebar:
    st.markdown("<h3 style='text-align: center; color: white; margin-top: 0;'>Capuchin Bird Project</h3>", unsafe_allow_html=True)
    
    # Display sidebar image
    sidebar_image = "images/capuchin_bird_small.jpeg"
    if os.path.exists(sidebar_image):
        st.image(sidebar_image, width=150)
    else:
        st.warning("Sidebar image not found")
    
    st.markdown("""
    <div style='text-align: center; margin-top: 10px; margin-bottom: 20px; color: rgba(255,255,255,0.8);'>
        <p>A research tool for ornithologists studying Capuchin birds in their natural habitat.</p>
        <p>Use our AI-powered analysis to detect and recognize bird calls.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-top: 0; margin-bottom: 20px; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)

# 3. YOUR HOMEPAGE CONTENT
st.markdown(
    "<h1 style='font-size: 2.5rem; font-weight: 700; color: #1976D2;'>Capuchin Bird Audio Recognizer</h1>",
    unsafe_allow_html=True
)
st.markdown("### Advanced audio analysis tool for researchers and conservationists")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    The **Capuchin Bird Audio Recognizer** helps researchers and conservationists analyze 
    audio recordings of Capuchin birds. This tool uses advanced signal processing 
    and machine learning to identify and count Capuchin calls in audio files, 
    providing valuable data for behavioral studies and conservation efforts.
    """)

with col2:
    # Create images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")
        
    image_path_large = "images/capuchin_bird_large.jpeg"
    if os.path.exists(image_path_large):
        st.image(image_path_large, width=200)
    else:
        st.warning(f"Image not found: {image_path_large}")
        st.markdown("🐦 Capuchin Bird Image")

st.markdown("## Features")
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("""
    <div style='background-color: #CCCCFF; border-radius: 10px; padding: 20px; height: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        <h4 style='color: #1565C0; font-size: 1.3rem;'>🎵 Upload & Process Audio</h4>
        <p style='color: #424242; font-size: 1.05rem;'>
            Upload audio recordings of Capuchin birds and visualize waveforms and spectrograms.
            Process files to identify and count distinct calls.
        </p>
    </div>
    """, unsafe_allow_html=True)
with colB:
    st.markdown("""
    <div style='background-color: #CCCCFF; border-radius: 10px; padding: 20px; height: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        <h4 style='color: #1565C0; font-size: 1.3rem;'>📊 Analyze Results</h4>
        <p style='color: #424242; font-size: 1.05rem;'>
            View detailed analysis of your audio files including call frequency, 
            patterns, and statistics. Export results for further research.
        </p>
    </div>
    """, unsafe_allow_html=True)
with colC:
    st.markdown("""
    <div style='background-color: #CCCCFF; border-radius: 10px; padding: 20px; height: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        <h4 style='color: #1565C0; font-size: 1.3rem;'>💬 RAG-Powered Chatbot</h4>
        <p style='color: #424242; font-size: 1.05rem;'>
            Ask questions about your audio analysis or Capuchin bird behavior
            with our intelligent assistant powered by RAG technology.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## Getting Started")
st.markdown("""
1. Go to **Upload Audio** from the sidebar  
2. Upload your audio file  
3. Process the file to analyze and count calls  
4. View detailed results and insights  
5. Use the chatbot to ask questions about your data  
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Developed for ecological research and conservation purposes.</p>
    <p>© 2025 Capuchin Bird Audio Recognizer | MIT License</p>
</div>
""", unsafe_allow_html=True)

