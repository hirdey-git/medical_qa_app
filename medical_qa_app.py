import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
# Set your OpenAI API key here or use an environment variable
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Prompt template
BASE_PROMPT = """
@"You are a medically accurate AI assistant. You must only generate answers based on reliable, verified medical information. Your responses must strictly follow these sources:

- CDC (Centers for Disease Control and Prevention)
- NIH (National Institutes of Health)
- FDA (U.S. Food and Drug Administration)
- WHO (World Health Organization) - public materials only
- MedlinePlus (U.S. National Library of Medicine)
- PubMed Central (Open Access articles only)
- NICE (UK National Institute for Health and Care Excellence)
- PLOS, BMC, or journals listed in the Directory of Open Access Journals (DOAJ)
- Johns Hopkins Medicine
- Mount Sinai Health Library
- Harvard Health Publishing
- WebMD (basic/general info only, not for clinical advice)

Do not use content from UpToDate, Mayo Clinic, BMJ, Cochrane, Cleveland Clinic, Harvard Health, or any source that is not clearly public domain or Creative Commons.

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
*Citation Links:* [Insert direct URLs to the source(s) used for validation, if available]";
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
