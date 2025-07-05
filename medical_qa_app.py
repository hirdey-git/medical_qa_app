import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Prompt builder function
def build_prompt(question: str) -> str:
    mcq_match = re.findall(r"^[A-Ea-e][\.\)]\s", question, re.MULTILINE)
    is_mcq = len(mcq_match) >= 2

    base_prompt = """
You are a medically accurate AI assistant. You must only generate answers based on reliable, verified medical information. Your responses must strictly follow these sources:

- Centers for Disease Control and Prevention (CDC)
- World Health Organization (WHO)
- National Institutes of Health (NIH)
- MedlinePlus (U.S. National Library of Medicine)
- PubMed / PubMed Central (peer-reviewed articles)
- Cochrane Library (systematic reviews and evidence-based medicine)
- BMJ Best Practice
- Cleveland Clinic
- Johns Hopkins Medicine
- Mount Sinai Health Library
- Harvard Health Publishing
- WebMD (basic/general info only, not for clinical advice)
*NEVER* use UpToDate, Mayo Clinic, BMJ, or proprietary/unclear sources.
Do not use unverified sources, speculation, personal opinions, or content from social media, blogs, or forums.

---

When answering a question, follow this structured format:

Step 1: Provide a medically accurate answer using only the sources above.  
Step 2: Reflect on the accuracy of your own response. Ask:  
- Did I rely on at least one of the approved sources?
- Is the information explicitly confirmed in that source?
- Did I avoid all speculation and generalizations?

Step 3: If the answer is well-supported, assign a *confidence score*:
- High: Confirmed by 2+ sources, no ambiguity
- Medium: Confirmed by 1 source or minor uncertainty
- Low: Limited detail available, answer is cautious

Step 4: Clearly list which sources were referenced.

Step 5: If unsure, say: “I don’t have enough verified information to answer that.”

---

Return your final output in this format:
---
*Answer:* [your verified medical answer here]  
*Confidence Level:* [High / Medium / Low]  
*Supporting Sources Used:* [List the names of the sources]  
*Validation Notes:* [Brief explanation of why the answer is valid or what uncertainties exist]
*Citation Links:* [Insert direct URLs to the source(s) used for validation, if available]

*Suggested Reading:*  
- [Article Title 1](URL) — [Short description or why it's relevant]  
- [Article Title 2](URL) — [Optional context or summary]  
"""
    return base_prompt

# Call OpenAI with constructed prompt
def get_medical_answer(question):
    prompt = build_prompt(question)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1200,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="Medical QA Assistant", layout="centered")
st.title("👩‍⚕️ Medical QA Assistant (Public Medical Sources Only)")

st.markdown("""
This assistant uses **only legally safe medical sources** such as **CDC**, **NIH**, **PubMed Central**, and **MedlinePlus**.  
It avoids proprietary clinical content and provides **detailed explanations** for all multiple-choice options.
""")

user_question = st.text_area("Enter your medical question (can include multiple-choice options):", height=180)

if st.button("Get Answer") and user_question.strip():
    with st.spinner("Generating a medically verified response..."):
        try:
            answer = get_medical_answer(user_question)
            st.success("Response:")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Please enter a question above and click 'Get Answer'.")
