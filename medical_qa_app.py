import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment variables
load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def build_prompt(question: str) -> str:
    import re
    mcq_match = re.findall(r"^[A-Da-d][\.\)]\s", question, re.MULTILINE)
    is_mcq = len(mcq_match) >= 2

    base_prompt = """
You are a medically accurate AI assistant.

Use only these legally safe public sources:
- CDC, NIH, FDA, WHO (public info only)
- MedlinePlus, PubMed Central (Open Access)
- NICE (UK), PLOS, BMC, DOAJ
- Johns Hopkins Medicine, Mount Sinai, Harvard Health (public info)
- WebMD (basic/general explanations only)

Do not use Mayo Clinic, UpToDate, BMJ, or any proprietary/uncertain sources.

If unsure, say: “I don’t have enough verified information to answer that.”

---
Return your final answer using this **strict format exactly** (including line breaks and labels):

*Answer:* [short answer summary]  
*Correct Option:* [e.g., A / B / C / D]  
*Confidence Level:* [High / Medium / Low]  
*Supporting Sources Used:* [CDC, NIH, etc.]  
*Validation Notes:* [Why this answer is correct, what criteria were met, any ambiguity]  
*Citation Links:*  
- [https://source1...]  
- [https://source2...]

If the question includes answer options, explain **why the correct one is best** and why the others are incorrect.

If the question has no options, just provide the best answer and explain the reasoning.
""".strip()

    return f"{base_prompt}\n\nQuestion:\n{question.strip()}\n\nAnswer:"


def get_medical_answer(question):
    prompt = build_prompt(question)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="Medical QA Assistant", layout="centered")
st.title("👩‍⚕️ Medical QA Assistant (Legal Sources Only)")

st.markdown("""
This assistant uses **only legally permitted medical sources** like CDC, NIH, MedlinePlus, and PubMed Central.  
It avoids any proprietary content and focuses on **explainable, source-cited responses**.
""")

user_question = st.text_area("Enter your medical question (with or without options):", height=150)

if st.button("Get Answer") and user_question.strip():
    with st.spinner("Generating medically verified response..."):
        try:
            answer = get_medical_answer(user_question)
            st.success("Response:")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Please enter a medical question above and click 'Get Answer'.")
