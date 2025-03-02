import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import os
import time
import random
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB0

seed_value = 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Set page configuration and title
st.set_page_config(page_title="Upload Audio", page_icon="🎤")
st.title("Upload Audio")
st.sidebar.header("Audio Upload Options")

# Audio file uploader
st.sidebar.subheader("Upload Your Audio File")
uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

# Global constants for capuchin detection
TARGET_SR = 16000
THRESHOLD_STAGE1 = 0.5
THRESHOLD_STAGE2 = 0.6
WINDOW_DURATION_OUTER = 6.0
STEP_DURATION_INNER = 0.3
OVERLAP_INNER = 0.0

# Function to extract mel spectrogram
def extract_mel_spectrogram(audio_path=None,target_sr=TARGET_SR, y=None, sr=None, n_mels=128, n_fft=2048, hop_length=512, target_time_frames=None):
    if y is None or sr is None:
        try:
            y, sr = librosa.load(audio_path, sr=target_sr, res_type='kaiser_best')
        except Exception as e:
            st.error(f"Error loading audio file: {audio_path}, {e}")
            return None
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if target_time_frames is not None:
        mel_spec_db = librosa.util.fix_length(mel_spec_db, size=target_time_frames, axis=1)
    return mel_spec_db

# Updated two-stage sliding window function for capuchin call detection
def count_capuchin_calls_two_stage_sliding_window(long_audio_path, model=None,
                                                  threshold_stage1=THRESHOLD_STAGE1, threshold_stage2=THRESHOLD_STAGE2,
                                                  window_duration_outer=WINDOW_DURATION_OUTER,
                                                  step_duration_inner=STEP_DURATION_INNER,
                                                  overlap_inner=OVERLAP_INNER):
    """
    Counts capuchin calls using a two-stage sliding window approach (LATEST VERSION).
    """
    if model is None:
        print("Model is not provided. Please load your trained model.")
        return None

    try:
        y_long, sr_long = librosa.load(long_audio_path, sr=TARGET_SR, res_type='kaiser_best')
    except Exception as e:
        print(f"Error loading long audio file: {long_audio_path}, {e}")
        return None
        
    log_msgs = []
    def log(msg):
        log_msgs.append(msg)
    window_samples_outer = int(window_duration_outer * sr_long)
    step_samples_inner = int(step_duration_inner * sr_long)
    hop_samples_inner = int(step_samples_inner * (1 - overlap_inner))
    total_duration_seconds = librosa.get_duration(y=y_long, sr=sr_long)

    capuchin_call_count = 0
    outer_window_start_time = 0.0

    print(f"Processing audio file (Two-Stage Sliding Window - LATEST): {os.path.basename(long_audio_path)}")
    print(f"Total duration: {total_duration_seconds:.2f} seconds")

    outer_start_sample = 0
    while outer_window_start_time < total_duration_seconds:
        outer_end_sample = outer_start_sample + window_samples_outer
        if outer_end_sample > len(y_long):
            outer_end_sample = len(y_long)
        outer_window_audio = y_long[outer_start_sample:outer_end_sample]
        outer_window_duration = len(outer_window_audio) / sr_long
        outer_window_end_time = outer_window_start_time + outer_window_duration

        print(f"\nOuter Window: Start Time: {outer_window_start_time:.2f}s, End Time: {outer_window_end_time:.2f}s, Duration: {outer_window_duration:.2f}s")

        # Stage 1: Quick Check - Classify Entire 6-second Segment (Option A)
        mel_spec_outer_window = extract_mel_spectrogram(audio_path=None, y=outer_window_audio, sr=sr_long, target_time_frames=157)
        stage1_predicted_class = 0  # Default to no call
        if mel_spec_outer_window is not None:
            mel_spec_outer_window_rgb = np.stack([mel_spec_outer_window] * 3, axis=-1)
            mel_spec_outer_window_reshaped = mel_spec_outer_window_rgb[np.newaxis, ...]
            stage1_prediction = model.predict(mel_spec_outer_window_reshaped, verbose=0)
            stage1_prediction_prob = stage1_prediction[0][0]
            stage1_predicted_class = int(stage1_prediction_prob > threshold_stage1)

        if stage1_predicted_class == 0:  # No Call Indicated by Stage 1
            print("  Stage 1: No Call Indicated - Skipping Stage 2")
        else:  # Call Indicated by Stage 1 - Proceed to Stage 2
            print("  Stage 1: Call Indicated - Proceeding to Stage 2 Inner Loop")
            has_call_in_window = False
            inner_start_sample = 0
            # Stage 2: Detailed Analysis (Inner Loop - 0.2-second Chunks)
            while inner_start_sample < len(outer_window_audio):
                inner_end_sample = inner_start_sample + step_samples_inner
                if inner_end_sample > len(outer_window_audio):
                    inner_end_sample = len(outer_window_audio)
                inner_chunk_audio = outer_window_audio[inner_start_sample:inner_end_sample]
                if len(inner_chunk_audio) < step_samples_inner / 2:
                    inner_start_sample += hop_samples_inner
                    continue
                mel_spec_chunk = extract_mel_spectrogram(audio_path=None, y=inner_chunk_audio, sr=sr_long)
                if mel_spec_chunk is not None:
                    mel_spec_chunk_rgb = np.stack([mel_spec_chunk] * 3, axis=-1)
                    mel_spec_chunk_reshaped = mel_spec_chunk_rgb[np.newaxis, ...]
                    prediction = model.predict(mel_spec_chunk_reshaped, verbose=0)
                    prediction_prob_inner = prediction[0][0]
                    predicted_class_inner = int(prediction_prob_inner > threshold_stage2)
                    if predicted_class_inner == 1:
                        has_call_in_window = True
                inner_start_sample += hop_samples_inner

            if has_call_in_window:
                capuchin_call_count += 1
                print("  Call Event Detected in Outer Window - Incrementing Call Count")
            else:
                print("  Stage 2: No Call Event Detected in Outer Window (Despite Stage 1 Indication)")
        outer_window_start_time += window_duration_outer
        outer_start_sample = int(outer_window_start_time * sr_long)

    print(f"\nTotal Capuchin calls detected (Two-Stage Sliding Window - LATEST) in {os.path.basename(long_audio_path)}: {capuchin_call_count}\n")
    return capuchin_call_count

# Load the capuchin model (cached to load only once)
@st.cache_resource
def load_capuchin_model():
    MODEL_WEIGHTS_FILE = r'weights\capuchin_bird_classifier.weights.h5'
    input_shape = (128, 157, 3)
    input_tensor = Input(shape=input_shape)
    base_model = EfficientNetB0(include_top=False, weights=None, input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output_tensor = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.load_weights(MODEL_WEIGHTS_FILE)
    return model

# Main section: Instructions for the user
st.markdown("""
### Upload your audio file for processing  
Supported formats: WAV, MP3, OGG, FLAC
""")

if uploaded_file is not None:
    # Save the uploaded file in session state and play the audio
    st.session_state.audio_file = uploaded_file
    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    # Create a temporary file to process the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_audio_path = tmp_file.name
    st.session_state.temp_audio_path = temp_audio_path

    # Two buttons: one for general audio analysis and one for capuchin call detection
    col1, col2 = st.columns(2)
    analyze_button = col1.button("Analyze Audio")
    count_button = col2.button("Count Capuchin Calls")
    
    # Audio analysis button callback
    if analyze_button:
        start_time = time.time()
        with st.spinner("Analyzing audio..."):
            try:
                y, sr = librosa.load(temp_audio_path)
                duration = librosa.get_duration(y=y, sr=sr)
                analysis_results = {
                    'filename': uploaded_file.name,
                    'duration': duration,
                    'sample_rate': sr,
                    'waveform': y,
                }
                # Compute spectrogram
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                analysis_results['spectrogram'] = D
                # Estimate tempo
                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    analysis_results['tempo'] = float(tempo)
                except Exception as e:
                    analysis_results['tempo'] = 0.0
                    st.warning(f"Could not extract tempo: {str(e)}")
                # Harmonic and percussive components
                harmonic, percussive = librosa.effects.hpss(y)
                analysis_results['harmonic'] = harmonic
                analysis_results['percussive'] = percussive
                # RMS energy and ZCR
                rms = librosa.feature.rms(y=y)[0]
                analysis_results['rms'] = rms
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                analysis_results['zcr'] = zcr
                # MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                analysis_results['mfccs'] = mfccs
                
                st.session_state.analysis_results = analysis_results
                elapsed = time.time() - start_time
                st.success(f"Audio processed successfully in {elapsed:.2f} seconds!")
                st.info("Detailed results (including capuchin call detection) will be available on the Results page.")
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    
    # Capuchin call detection button callback
    if count_button:
        if st.session_state.get("analysis_results") is None:
            st.error("Please analyze the audio first by clicking 'Analyze Audio'.")
        else:
            start_time = time.time()
            with st.spinner("Counting capuchin calls..."):
                model = load_capuchin_model()
                call_count = count_capuchin_calls_two_stage_sliding_window(temp_audio_path, model)
            if call_count is not None:
                st.session_state.analysis_results['capuchin_calls'] = call_count
                elapsed = time.time() - start_time
                st.success(f"Capuchin call detection executed in {elapsed:.2f} seconds!")
                st.info("Capuchin call results and logs have been saved. Please visit the Results page to view detailed output.")
            else:
                st.error("An error occurred during capuchin call detection.")
else:
    st.info("Please upload an audio file to begin processing.")
