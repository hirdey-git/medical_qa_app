import streamlit as st
import openai
import os
import tempfile
import base64
from io import BytesIO
from pydub import AudioSegment
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
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


def transcribe_audio(audio_file):
    audio = AudioSegment.from_file(audio_file)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        audio.export(temp_wav.name, format="wav")
        with open(temp_wav.name, "rb") as f:
            transcript = openai.Audio.transcribe("whisper-1", f)
    return transcript['text']

# Streamlit UI
st.set_page_config(page_title="Medical QA Voice Assistant", layout="centered")
st.title("👩‍⚕️ Medical QA Assistant (Voice Enabled)")

st.markdown("""
This assistant uses **only legally permitted medical sources** such as CDC, NIH, MedlinePlus, and PubMed Central (Open Access). It avoids using any proprietary clinical content.
""")

st.markdown("### 🎙️ Record Your Question")

# Voice input
audio_bytes = st.audio_recorder(
    text="Click to record your medical question",
    pause_threshold=2.0,
    sample_rate=44100,
    key="recorder"
)

question_text = ""

if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio.flush()
        try:
            question_text = transcribe_audio(temp_audio.name)
            st.success("Transcription: " + question_text)
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")

# Text input fallback
user_input = st.text_area("Or type your question below:", value=question_text, height=150)

if st.button("Get Answer") and user_input.strip():
    with st.spinner("Generating medically verified response..."):
        try:
            answer = get_medical_answer(user_input.strip())
            st.success("Response:")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Record or type a question and click 'Get Answer'.")
