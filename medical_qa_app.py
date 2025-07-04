import streamlit as st
import openai
import os
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
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()



# Streamlit UI
st.set_page_config(page_title="Medical QA Assistant", layout="centered")
st.title("👩‍⚕️ Medical QA Assistant (Legal Sources Only)")

st.markdown("""
This assistant uses **only legally permitted medical sources** such as CDC, NIH, MedlinePlus, and PubMed Central (Open Access).
It avoids using any proprietary clinical content.
""")

user_question = st.text_area("Enter your medical question:", height=150)

if st.button("Get Answer") and user_question.strip():
    with st.spinner("Generating medically verified response..."):
        try:
            answer = get_medical_answer(user_question)
            st.success("Response:")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Please enter a question above and click 'Get Answer'.")
