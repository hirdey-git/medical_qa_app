import streamlit as st
import openai
import numpy as np
import av
import os
import tempfile
import wave
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_webrtc import RTCConfiguration
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
# Set your OpenAI API key here or use an environment variable
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Prompt template
BASE_PROMPT = """
You are a medically accurate AI assistant.

Use only the following public, legally safe medical sources:
- CDC (Centers for Disease Control and Prevention)
- NIH (National Institutes of Health)
- FDA (U.S. Food and Drug Administration)
- WHO (World Health Organization) - public materials only
- MedlinePlus (U.S. National Library of Medicine)
- PubMed Central (Open Access articles only)
- NICE (UK National Institute for Health and Care Excellence)
- PLOS, BMC, or journals listed in the Directory of Open Access Journals (DOAJ)

Do not use content from UpToDate, Mayo Clinic, BMJ, Cochrane, Cleveland Clinic, Harvard Health, or any source that is not clearly public domain or Creative Commons.

Never guess. If the information is not covered in the listed sources, respond:
"I don't have enough verified, legally usable information to answer that."

Always respond with a respectful and professional tone. This is not medical advice; recommend users consult a licensed healthcare provider.

Question: {question}

Answer:
"""

def get_medical_answer(question):
    prompt = BASE_PROMPT.format(question=question)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=5000,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def transcribe_audio_file(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

# Streamlit UI
st.set_page_config(page_title="Medical QA Voice Assistant", layout="centered")
st.title("👩‍⚕️ Medical QA Assistant (Real-Time Voice Enabled)")

st.markdown("""
This assistant uses **only legally permitted medical sources** such as CDC, NIH, MedlinePlus, and PubMed Central (Open Access). It avoids using any proprietary clinical content.
""")

st.markdown("### 🎤 Record Your Voice")

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_config,
    media_stream_constraints={"audio": True, "video": False},
    audio_receiver_size=1024,
    async_processing=True,
)

if ctx.audio_receiver:
    try:
        audio_frames = ctx.audio_receiver.get_frames(timeout=5)
        if not audio_frames:
            st.warning("No audio frames received.")
        else:
            frame = audio_frames[0]
            sample_rate = frame.sample_rate
            samples = np.concatenate([f.to_ndarray() for f in audio_frames]).astype(np.int16)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(samples.tobytes())

                st.success("Voice recorded. Transcribing...")
                transcription = transcribe_audio_file(f.name)
                st.info(f"Transcription: {transcription}")

                if st.button("Get Answer"):
                    with st.spinner("Generating medically verified response..."):
                        try:
                            answer = get_medical_answer(transcription)
                            st.success("Response:")
                            st.markdown(answer)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    except Exception as e:
        st.error(f"Recording failed: {str(e)}")
else:
    st.info("Click the microphone above to record your voice.")
