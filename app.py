import google.generativeai as genai
import streamlit as st

st.set_page_config(page_title="AI CodeCraft Studio", layout="wide")

# Automatically load API key from secrets.toml or session state
if "api_key" not in st.session_state:
    if "GEMINI_API_KEY" in st.secrets:
        st.session_state["api_key"] = st.secrets["GEMINI_API_KEY"]
    else:
        st.session_state["api_key"] = ""

# Sidebar Configuration & State Management
with st.sidebar:
    st.title("⚙️ IDE Settings")

    user_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state["api_key"],
        placeholder="Loaded from secrets.toml or paste here...",
    )
    if user_api_key:
        st.session_state["api_key"] = user_api_key

    st.markdown("---")
    st.subheader("🎯 Problem Generator")
    topic = st.selectbox(
        "Topic",
        [
            "Data Structures & Algorithms",
            "Python OOP & Design Patterns",
            "Concurrency & System Utilities",
        ],
    )
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

    if st.button("✨ Generate New Problem", type="primary"):
        current_key = st.session_state.get("api_key", "")
        if not current_key:
            st.error(
                "Please provide a Gemini API Key via secrets.toml or the sidebar."
            )
        else:
            try:
                # Configure the legacy google.generativeai module
                genai.configure(api_key=current_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                prompt = f"""
                Generate a unique coding challenge about {topic} with {difficulty} difficulty.
                Provide:
                1. Problem Title & Background Story
                2. Detailed Problem Statement
                3. Input / Output Format Specifications
                4. Two explicit Test Cases (Input & Expected Output)
                Format the output cleanly in Markdown.
                """
                response = model.generate_content(prompt)
                st.session_state["current_problem"] = response.text
                st.session_state["user_code"] = (
                    "# Write your solution from scratch here\n\ndef solution():\n    pass\n"
                )
                st.success("New challenge loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating challenge: {e}")

# Main IDE Dashboard Layout
st.title("⚡ AI CodeCraft Studio: Interactive AI Coding Dojo")

# Initialize Session States
if "current_problem" not in st.session_state:
    st.session_state["current_problem"] = (
        "### Welcome to AI CodeCraft!\nClick **'Generate New Problem'** in the sidebar to fetch a challenge."
    )
if "user_code" not in st.session_state:
    st.session_state["user_code"] = "# Select a problem to start coding..."

# Two-Column Workspace Layout
col_left, col_right = st.columns([1, 1.2], gap="medium")

with col_left:
    st.subheader("📖 Problem Description")
    with st.container(height=520, border=True):
        st.markdown(st.session_state["current_problem"])

with col_right:
    st.subheader("💻 Code Workspace (IDE)")

    user_code = st.text_area(
        "Write code from scratch:",
        value=st.session_state["user_code"],
        height=380,
        placeholder="Write your python solution here...",
    )
    st.session_state["user_code"] = user_code

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        submit_btn = st.button(
            "🚀 Submit Code for AI Audit",
            type="primary",
            use_container_width=True,
        )
    with eval_col2:
        hint_btn = st.button("💡 Ask AI for a Hint", use_container_width=True)

# Evaluation Section
active_api_key = st.session_state.get("api_key", "")

if submit_btn:
    if not active_api_key:
        st.error(
            "Please enter your Gemini API key in the sidebar or secrets.toml."
        )
    else:
        with st.spinner("🤖 AI Code Auditor is analyzing your solution..."):
            genai.configure(api_key=active_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            audit_prompt = f"""
            You are a Principal Software Architect and Technical Interviewer. 
            Evaluate the following student code submission against the problem description.

            ### Problem Statement:
            {st.session_state['current_problem']}

            ### Student Submission:
            ```python
            {st.session_state['user_code']}
            ```

            Provide your assessment structured as follows:
            1. **Correctness Verdict:** Does it pass all edge cases? (Pass / Fail / Partial)
            2. **Code Quality & Bugs:** Identify syntax errors, logic flaws, or missed edge cases.
            3. **Complexity Analysis:** State the Time and Space Big-O complexities.
            4. **Optimized Refactoring:** Provide clean, production-grade refactored code if improvements are needed.
            """
            feedback = model.generate_content(audit_prompt)

            st.markdown("---")
            st.subheader("🔍 AI Audit & Validation Report")
            st.markdown(feedback.text)

if hint_btn:
    if not active_api_key:
        st.error(
            "Please enter your Gemini API key in the sidebar or secrets.toml."
        )
    else:
        with st.spinner("Thinking of a gentle nudge..."):
            genai.configure(api_key=active_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            hint_prompt = f"""
            The student is stuck on this problem:
            {st.session_state['current_problem']}

            Their current code snippet is:
            ```python
            {st.session_state['user_code']}
            ```
            Give a helpful nudge or architectural hint **without** giving away the full solution code.
            """
            hint_response = model.generate_content(hint_prompt)
            st.info(f"💡 **AI Hint:**\n\n{hint_response.text}")