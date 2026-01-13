
#USING STREAMLIT BOTH UPLOAD FILE AND LIVE AUDIO

import streamlit as st
import requests
import json
import io
import numpy as np
from scipy.io import wavfile
import soundfile as sf
import sounddevice as sd
import pyaudio
import wave

import os
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
WHISPER_MODEL = "distil-whisper/distil-large-v3"

def transcribe(audio_buffer):
    try:
        # Prepare the request for DeepInfra
        url = f"https://api.deepinfra.com/v1/inference/{WHISPER_MODEL}"
        headers = {"Authorization": f"bearer {DEEPINFRA_API_KEY}"}
        files = {'audio': ('audio.wav', audio_buffer, 'audio/wav')}

        # Send the request
        response = requests.post(url, headers=headers, files=files)
        if response.status_code == 200:
            result = json.loads(response.text)
            return result.get("text", "No transcription result found.")
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def record_audio_using_sounddevice(sample_rate=16000):
    """Record audio for a fixed duration (5 seconds) using sounddevice and return as a BytesIO buffer."""
    try:
        # Check if audio input devices are available
        devices = sd.query_devices()
        input_devices = [device for device in devices if device['max_input_channels'] > 0]

        if not input_devices:
            st.error("No input devices found. Please ensure a microphone is connected.")
            return None

        # Recording duration
        duration = 5  # Fixed recording duration in seconds
        st.info("Recording for 5 seconds...")

        # Start recording
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        st.success("Recording complete!")
        
        # Convert to WAV format
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio)
        buffer.seek(0)  # Reset buffer pointer
        return buffer
    except Exception as e:
        st.error(f"Error during recording: {str(e)}")
        return None

def record_audio_using_pyaudio(sample_rate=16000):
    """Record audio for a fixed duration (5 seconds) using pyaudio and return as a BytesIO buffer."""
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        duration = 5  # Duration in seconds

        # Open stream for recording
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

        st.info(f"Recording for {duration} seconds...")

        frames = []
        for _ in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)

        st.success("Recording complete!")
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save to buffer in WAV format
        buffer = io.BytesIO()
        wf = wave.open(buffer, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error during recording: {str(e)}")
        return None

# Streamlit app setup
st.title("Whisper Large V3 Turbo: Transcribe Audio")
st.write("Transcribe long-form microphone or audio inputs. Powered by DeepInfra.")

# Option for file upload or live recording
option = st.radio("Choose an option:", ["Upload an audio file", "Record live audio"])

if option == "Upload an audio file":
    # File upload with a unique key
    uploaded_file = st.file_uploader(
        "Upload an audio file", 
        type=["wav", "mp3", "ogg"], 
        key="file_upload_option"
    )
    if uploaded_file is not None:
        try:
            audio_data, rate = sf.read(uploaded_file)
            buffer = io.BytesIO()
            wavfile.write(buffer, rate, audio_data)
            buffer.seek(0)
            transcription = transcribe(buffer)
            if transcription:
                st.subheader("Transcription Result:")
                st.write(transcription)
        except Exception as e:
            st.error(f"Error processing the uploaded file: {str(e)}")

elif option == "Record live audio":
    # Live recording
    if st.button("Start Recording", key="start_recording_button"):
        # Use either sounddevice or pyaudio for recording based on availability
        audio_buffer = record_audio_using_sounddevice() or record_audio_using_pyaudio()
        if audio_buffer:
            transcription = transcribe(audio_buffer)
            if transcription:
                st.subheader("Transcription Result:")
                st.write(transcription)