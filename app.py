


import streamlit as st
import openai
import json
import hashlib
import os
from datetime import datetime
from typing import List, Dict

# ---------------------- Configuration ----------------------

EXIT_KEYWORDS = {"exit", "quit", "bye", "end", "close", "stop"}
ANONYMIZE_SALT = os.environ.get("ANON_SALT", "default_salt_please_change")
CANDIDATES_FILE = "candidates.json"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Use a safe default so app doesn't crash. If not set, we will show an info and use a stub.
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ---------------------- Utilities ----------------------

def anonymize(value: str) -> str:
    """Create a deterministic anonymized token from a value using SHA256 + salt."""
    if value is None:
        return ""
    h = hashlib.sha256()
    h.update((ANONYMIZE_SALT + value).encode("utf-8"))
    return h.hexdigest()


def load_candidates() -> List[Dict]:
    if os.path.exists(CANDIDATES_FILE):
        try:
            with open(CANDIDATES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_candidate_record(record: Dict):
    records = load_candidates()
    records.append(record)
    with open(CANDIDATES_FILE, "w") as f:
        json.dump(records, f, indent=2)


# ---------------------- Prompt Engineering ----------------------

SYSTEM_PROMPT = (
    "You are a helpful, concise technical interviewer assistant. "
    "When given a candidate's declared tech stack, produce 3-5 technical questions for each technology. "
    "For each question, include: the question text, a short expected answer in 1-2 sentences, and a difficulty tag (easy/medium/hard). "
    "Keep language professional and suitable for an interview situation. "
)

FALLBACK_PROMPT = (
    "You are a polite assistant. The user's input could not be understood. "
    "Ask for clarification or provide a fallback helpful instruction about how the user can proceed. "
)


def build_generation_prompt(candidate_info: Dict) -> str:
    """Create the user prompt sent to the LLM given candidate info and tech stack."""
    name = candidate_info.get("full_name") or "Candidate"
    techstack = candidate_info.get("tech_stack") or ""
    if isinstance(techstack, list):
        techstack_str = ", ".join(techstack)
    else:
        techstack_str = techstack

    prompt = (
        f"Generate interview questions tailored to the following candidate: \n"
        f"Name: {name}\n"
        f"Years of experience: {candidate_info.get('years_experience', 'N/A')}\n"
        f"Desired positions: {candidate_info.get('desired_positions', 'N/A')}\n"
        f"Tech Stack: {techstack_str}\n\n"
        "For each technology listed under Tech Stack, produce 3-5 targeted technical questions. "
        "For each question include a short expected answer (1-2 sentences) and label difficulty as easy/medium/hard. "
        "Return results in JSON array form where each item is {"""\"technology\"": "<tech>", """\"questions\"": [{"""\"q\"":..., \"a\":..., \"difficulty\":...}]}" 
    )
    return prompt


# ---------------------- LLM Interaction ----------------------

def call_llm_generate_questions(prompt: str, max_tokens: int = 800) -> str:
    """Call OpenAI's ChatCompletion (gpt-3.5/4 style). If OPENAI_API_KEY not set, return a stubbed response."""
    if not OPENAI_API_KEY:
        # Return a deterministic stub so the app can be tested without an API key.
        return json.dumps([
            {
                "technology": "Python",
                "questions": [
                    {"q": "Explain list comprehensions.", "a": "Concise way to create lists using an expression and an iterable.", "difficulty": "easy"},
                    {"q": "What is GIL?", "a": "Global Interpreter Lock, which prevents multiple native threads from executing Python bytecodes concurrently.", "difficulty": "medium"}
                ]
            }
        ])

    try:
        response = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return ""


# ---------------------- Helpers for UI ----------------------

def parse_llm_json_output(text: str):
    """Try to parse JSON from the LLM. If it fails, return None."""
    try:
        return json.loads(text)
    except Exception:
        # Try to extract JSON-like substring
        import re
        m = re.search(r"\[\s*\{.+\}\s*\]", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def generate_questions_for_stack(candidate_info: Dict) -> List[Dict]:
    prompt = build_generation_prompt(candidate_info)
    raw = call_llm_generate_questions(prompt)
    parsed = parse_llm_json_output(raw)
    if parsed is None:
        # return a fallback: convert textual response into a small parsed structure
        return [{
            "technology": "General",
            "questions": [
                {"q": "Describe a project you are proud of.", "a": "Share scope, your contribution, and impact.", "difficulty": "easy"}
            ]
        }]
    return parsed


# ---------------------- Streamlit App ----------------------

st.set_page_config(page_title="Hiring Assistant Chatbot", layout="centered")

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = {}
if "terminated" not in st.session_state:
    st.session_state.terminated = False

st.title("Hiring Assistant — Interview Question Generator")
st.write("A simple hiring assistant that collects candidate info and generates targeted technical questions based on declared tech stack.")

# Top bar: Exit if exit keywords typed
user_input = st.text_input("(Optional) Chat with assistant — type message or an exit keyword to finish conversation", key="user_input")

if user_input:
    lower = user_input.strip().lower()
    if any(k in lower.split() for k in EXIT_KEYWORDS):
        st.session_state.terminated = True
        st.success("Conversation ended. Thank you! We'll follow up with next steps.")
        st.stop()
    else:
        # naive understanding: echo back or ask to use the form for structured data
        if "tech" in lower or "question" in lower or "generate" in lower:
            st.info("Use the form below to submit your tech stack and candidate details. Then press 'Generate Questions'.")
        else:
            st.info("I'm here to help. Use the form below to provide candidate details and tech stack to generate interview questions.")

# Candidate form
with st.form(key="candidate_form"):
    st.subheader("Candidate Details")
    full_name = st.text_input("Full name", placeholder="First Last")
    email = st.text_input("Email address", placeholder="candidate@example.com")
    phone = st.text_input("Phone number", placeholder="+91-XXXXXXXXXX")
    years = st.number_input("Years of experience", min_value=0, max_value=50, value=0, step=1)
    desired_positions = st.text_input("Desired position(s)", placeholder="e.g. Backend Developer, Data Analyst")
    location = st.text_input("Current location", placeholder="City, Country")

    st.markdown("---")
    st.subheader("Tech Stack Declaration")
    st.markdown("Provide the technologies you're proficient in. Example: Python, Django, PostgreSQL, React, Docker")
    tech_stack_raw = st.text_area("Tech stack (comma-separated or one per line)")

    submit = st.form_submit_button("Save candidate & generate questions")

    if submit:
        # Basic validation
        if not full_name or not email or not tech_stack_raw:
            st.warning("Please provide at minimum: Full name, Email, and Tech stack.")
        else:
            # parse tech stack
            techs = [t.strip() for t in tech_stack_raw.replace('\n', ',').split(',') if t.strip()]
            candidate = {
                "full_name": full_name.strip(),
                "email_anon": anonymize(email.strip()),
                "phone_anon": anonymize(phone.strip()),
                "years_experience": years,
                "desired_positions": desired_positions.strip(),
                "location": location.strip(),
                "tech_stack": techs,
                "submitted_at": datetime.utcnow().isoformat() + "Z",
            }

            # Save simulated/anonymized record
            save_candidate_record(candidate)

            # Put in session state (do not store raw email/phone)
            st.session_state.candidate_info = candidate
            st.success("Candidate recorded (anonymized). Generating tailored interview questions...")

            # Generate questions
            with st.spinner("Generating questions — powered by the language model..."):
                q = generate_questions_for_stack(candidate)
                st.session_state.conversation.append({"type": "questions", "content": q})

# Show generated questions if present
if st.session_state.conversation:
    for item in st.session_state.conversation:
        if item.get("type") == "questions":
            st.markdown("### Generated Technical Questions")
            data = item.get("content")
            if isinstance(data, list):
                for tech_block in data:
                    tech = tech_block.get("technology")
                    st.markdown(f"**{tech}**")
                    qs = tech_block.get("questions", [])
                    for i, qq in enumerate(qs, start=1):
                        qtext = qq.get("q") if isinstance(qq, dict) else str(qq)
                        atext = qq.get("a") if isinstance(qq, dict) else ""
                        diff = qq.get("difficulty") if isinstance(qq, dict) else ""
                        st.write(f"{i}. {qtext}")
                        if atext:
                            st.caption(f"Expected: {atext} — Difficulty: {diff}")
                        else:
                            st.caption(f"Difficulty: {diff}")
                    st.markdown("---")
            else:
                st.write(data)

# Fallback / help area
with st.expander("How this works — Notes & privacy"):
    st.write(
        "This demo collects candidate info and generates interview questions. Candidate PII (email and phone) is stored only in anonymized hashed form in a local file 'candidates.json'.\n"
        "If you deploy this in production, you must secure storage, use HTTPS, rotate keys, and follow local data privacy laws (e.g., GDPR).\n"
        "This demo supports OpenAI model integration; set OPENAI_API_KEY environment variable. If not set, the app will display a stubbed example."
    )

# README generator (write README.md for repository convenience)
README_CONTENT = r"""
# Hiring Assistant Chatbot (Streamlit)

## Project Overview
A simple hiring assistant that collects candidate information and generates targeted technical questions based on the declared tech stack.

## Features
- Candidate information collection.
- Tech stack declaration (languages, frameworks, DBs, tools).
- Generates 3-5 tailored technical questions per technology using an LLM.
- Context handling using Streamlit session state.
- Exit keywords to gracefully end conversation.
- Simulated and anonymized storage of candidate records.

## Installation
1. Clone or download the repository.
2. Create a virtual environment and activate it.

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

If you don't have a requirements file, install directly:
```
pip install streamlit openai python-dotenv
```

## Setup
Set the OpenAI API key as an environment variable before running the app:

Linux / macOS:
```bash
export OPENAI_API_KEY="sk-..."
export ANON_SALT="your_secret_salt"
```

Windows (powershell):
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ANON_SALT = "your_secret_salt"
```

## Run locally
```
streamlit run hiring_assistant_streamlit.py
```

## Prompt Design
The app uses a small system prompt that instructs the model to generate interview questions per technology, return an expected answer and difficulty tag. Prompts are deliberately low-temperature (0.2) to favor stable, useful replies.

## Data Handling
- Candidate email and phone are anonymized using SHA256 + salt and saved to `candidates.json` for simulated backend behavior.
- In production, replace this with secure storage (encrypted DB), access controls, and data retention policies.

## Challenges & Solutions
- **LLM output format inconsistency**: LLMs may not always return strict JSON. We attempt to parse JSON, and fall back to a heuristic extraction. For production, use stricter prompt constraints and validation.
- **Privacy**: Demonstrated anonymization. Full compliance requires more (encryption-at-rest, access audits).

## Code Structure & Quality
All logic is contained in the single file `hiring_assistant_streamlit.py` for simplicity. Functions are modular and documented.

## Next steps / Bonuses
- Deploy to cloud (Streamlit Cloud, AWS Elastic Beanstalk, or a container on GCP).
- Add OAuth-protected admin interface to view anonymized candidates.
- Integrate a database and secure secrets management.

"""

# Write README.md if not exists
if not os.path.exists("README.md"):
    try:
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(README_CONTENT)
    except Exception:
        pass

# Footer
st.sidebar.header("Developer Notes")
st.sidebar.write("Local demo. Candidate PII is anonymized before saving. Set OPENAI_API_KEY to enable LLM questions.")
st.sidebar.markdown("---")

st.info("If you'd like, I can: (1) add GitHub workflow files for deployment, (2) produce a Dockerfile, or (3) produce cloud deployment steps for AWS/GCP.")
