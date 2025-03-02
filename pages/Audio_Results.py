# pages/2_Audio_Results.py
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.set_page_config(page_title="Audio Analysis Results", page_icon="📊")
st.title("Audio Analysis Results")
st.sidebar.header("Results Options")

if st.session_state.get("analysis_results") is None:
    st.warning("No audio analysis results found. Please upload and process an audio file first.")
else:
    results = st.session_state.analysis_results

    # Create two tabs: one for general analysis and one for capuchin call detection results.
    tab1, tab2 = st.tabs(["General Analysis", "Capuchin Call Detection"])

    with tab1:
        st.subheader("Audio Information")
        st.write(f"**Filename:** {results['filename']}")
        st.write(f"**Duration:** {results['duration']:.2f} seconds")
        st.write(f"**Sample Rate:** {results['sample_rate']} Hz")
        if 'tempo' in results and results['tempo'] > 0:
            st.write(f"**Estimated Tempo:** {results['tempo']:.2f} BPM")
        else:
            st.write("**Estimated Tempo:** Not available")

        st.subheader("Audio Visualization")
        visualization_type = st.sidebar.selectbox(
            "Select Visualization",
            ["Spectrogram", "MFCC", "Energy", "Zero Crossing Rate"],
            key="viz_select_results"
        )

        if visualization_type == "Spectrogram":
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(
                results['spectrogram'],
                x_axis='time',
                y_axis='log',
                sr=results['sample_rate'],
                ax=ax
            )
            ax.set_title("Spectrogram")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig)
            st.write("The spectrogram shows frequency content over time.")
        elif visualization_type == "MFCC":
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(
                results['mfccs'],
                x_axis='time',
                sr=results['sample_rate'],
                ax=ax
            )
            ax.set_title("MFCC")
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)
            st.write("MFCC features capture the spectral properties used for audio analysis.")
        elif visualization_type == "Energy":
            fig, ax = plt.subplots(figsize=(10, 4))
            times = librosa.times_like(results['rms'], sr=results['sample_rate'])
            ax.plot(times, results['rms'])
            ax.set_title("RMS Energy")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Energy")
            st.pyplot(fig)
            st.write("RMS Energy indicates how loud the audio signal is over time.")
        elif visualization_type == "Zero Crossing Rate":
            fig, ax = plt.subplots(figsize=(10, 4))
            times = librosa.times_like(results['zcr'], sr=results['sample_rate'])
            ax.plot(times, results['zcr'])
            ax.set_title("Zero Crossing Rate")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ZCR")
            st.pyplot(fig)
            st.write("Zero Crossing Rate provides a measure of the noisiness of the audio signal.")

        st.subheader("Audio Segments Analysis")
        segment_length = st.slider("Segment Length (seconds)", min_value=1, max_value=10, value=3, key="seg_slider")
        samples_per_segment = segment_length * results['sample_rate']
        num_segments = int(len(results['waveform']) / samples_per_segment)

        if num_segments > 0:
            segment_data = []
            for i in range(num_segments):
                start_sample = int(i * samples_per_segment)
                end_sample = int(min(start_sample + samples_per_segment, len(results['waveform'])))
                segment = results['waveform'][start_sample:end_sample]
                try:
                    segment_rms = float(np.mean(librosa.feature.rms(y=segment)[0]))
                    segment_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=segment)[0]))
                except Exception:
                    segment_rms = 0.0
                    segment_zcr = 0.0
                segment_data.append({
                    "Segment": i + 1,
                    "Time (s)": f"{i * segment_length:.1f} - {min((i + 1) * segment_length, results['duration']):.1f}",
                    "Energy": f"{segment_rms:.4f}",
                    "ZCR": f"{segment_zcr:.4f}"
                })
            st.dataframe(pd.DataFrame(segment_data))
        else:
            st.info("Audio is too short to segment with the current settings.")

    with tab2:
        st.subheader("Capuchin Call Detection Results")
        if 'capuchin_calls' in results:
            st.write(f"**Capuchin Call Count:** {results['capuchin_calls']}")
        else:
            st.write("**Capuchin Call Count:** Not available")
        if 'detection_log' in results:
            st.text_area("Detection Log", results['detection_log'], height=300)
        else:
            st.info("No detection log available.")

