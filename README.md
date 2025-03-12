# Capuchin Bird Audio Recognizer

An advanced audio analysis tool designed for researchers and conservationists to identify and analyze Capuchin bird calls in audio recordings.

## Features

- **Audio Upload & Processing**: Upload audio recordings and visualize waveforms for initial analysis
- **AI-Powered Detection**: Automatically detect and count Capuchin bird calls using machine learning
- **Detailed Analysis**: View comprehensive analysis including spectrograms, MFCCs, energy levels, and more
- **Timestamp Analysis**: Get precise timestamps of where bird calls occur in recordings
- **Interactive Results**: Visual timeline and exportable CSV data of detected calls
- **RAG-Powered Chatbot**: Ask questions about your analysis or Capuchin birds with our AI assistant

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/capuchin-bird-audio-recognizer.git
cd capuchin-bird-audio-recognizer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in the `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

Run the application with:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

### How to Use:

1. Navigate through the application using the sidebar
2. Upload audio files on the "Upload Audio" page
3. Analyze the audio using the provided tools
4. View detailed results on the "Audio Results" page
5. Use the chatbot to ask questions about your analysis

## Project Structure

- `app.py`: Main application entry point with home page
- `pages/`:
  - `Upload_Audio.py`: Audio file upload and processing functionality
  - `Audio_Results.py`: Detailed analysis results and visualizations
  - `Chat_Assistant.py`: RAG-powered chatbot for audio analysis questions
  - `About.py`: Project information and resources
- `images/`: Image assets for the application
- `weights/`: Pre-trained model weights for bird call detection
- `docs/`: Documentation files
- `chroma_db/` and `temp_chroma_db/`: Vector databases for the RAG chatbot

## Technologies Used

- **Streamlit**: Web application framework
- **TensorFlow**: Deep learning for audio classification
- **Librosa**: Audio processing and feature extraction
- **LangChain**: Framework for the RAG-powered chatbot
- **Chromadb**: Vector database for storing embeddings
- **Matplotlib & Pandas**: Data visualization and manipulation

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 