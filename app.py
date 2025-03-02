import streamlit as st

st.set_page_config(
    page_title="Audio Processing App",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("Audio Processing App")
    st.sidebar.success("Select a page from the sidebar.")
    
    st.markdown(
        """
        ## Welcome to the Audio Processing Application
        
        This application allows you to:
        
        - **Upload Audio**: Upload and process audio files
        - **View Results**: See detailed analysis of your audio files
        - **Chat Assistant**: Ask questions about your audio or get help
        
        To get started, select "Upload Audio" from the sidebar.
        """
    )
    
    # Initialize session state for storing audio data between pages
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

if __name__ == "__main__":
    main()