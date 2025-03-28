import streamlit as st
import os

# Set page configuration and title
st.set_page_config(page_title="About", page_icon="ℹ️")

# Header section with logo and title
col1, col2 = st.columns([1, 3])

with col1:
    # Logo
    banner_path = os.path.join("images", "capuchin_bird_small.jpeg")
    if os.path.exists(banner_path):
        st.image(banner_path, width=150)
    else:
        st.write("🐦")

with col2:
    st.title("About the Project")
    st.markdown("*An advanced audio analysis tool for Capuchin bird research*")

# Add separator
st.markdown("<hr style='margin: 1em 0; opacity: 0.3;'>", unsafe_allow_html=True)

# Project overview
st.subheader("Project Overview")
st.markdown("""
The Capuchin Bird Audio Recognizer is designed to help researchers identify and count
distinct bird calls in audio recordings. Using advanced signal processing and machine
learning techniques, this tool provides valuable data for behavioral studies and 
conservation efforts.
""")

# Main sections in two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Key Features")
    st.markdown("""
    - Audio upload and waveform visualization
    - Automated bird call detection
    - Interactive results dashboard
    - Timestamp analysis of detected calls
    - Data export capabilities
    """)

with col2:
    st.subheader("Technologies")
    st.markdown("""
    - Streamlit
    - TensorFlow
    - Librosa
    - Matplotlib
    - NumPy/Pandas
    """)

# Team and links in expanders
with st.expander("Team"):
    st.markdown("""
    ### Group B10

    This project was developed by:
    
    - **Hariharan C K**
    - **Hima Harikumar**
    - **Sandra M P**
    - **Siva S**
    
    *Guided by:* **Nisha P K**
    """)

with st.expander("Links & Resources"):
    st.markdown("""
    - [GitHub Repository](https://github.com/Russow10/capuchin-bird-audio-recognizer)
    - [Research Documentation](https://example.com/docs)
    """)

# Footer
st.markdown("<hr style='margin: 2em 0 1em 0; opacity: 0.3;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
© 2025 Capuchin Bird Audio Recognizer | MIT License
</div>
""", unsafe_allow_html=True) 