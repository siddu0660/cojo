# AI CodeCraft Studio

AI CodeCraft Studio is a Python coding tutor and judge. Pick a topic and a difficulty, generate an original challenge, and solve it in a blank editor. Submit the draft for an audit, or ask for a hint that stops short of the solution.

Audits come back in four parts:

1. Correctness verdict
2. Code quality and bugs
3. Big-O time and space complexity
4. A clean refactored solution

The review is Gemini's judgment. The app does not execute your code.

## Run locally

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints, usually `http://localhost:8501`.

## API key

The app looks for `GEMINI_API_KEY` in Streamlit secrets first. If that secret is missing, paste the key into the sidebar. The sidebar value stays in `st.session_state` for the browser session and is not written to disk.

Local secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

Create a key in [Google AI Studio](https://aistudio.google.com/apikey). `.streamlit/secrets.toml` is gitignored. Do not commit a real key.

The example file ships with the placeholder `your-gemini-api-key`. The app treats that placeholder as missing and falls back to the sidebar.

## Deploy

The app is a Streamlit script (`app.py`) with dependencies pinned in `requirements.txt`.

On [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this project to a Git repository.
2. Create a new app and set the main file to `app.py`.
3. In the app settings, open **Secrets** and paste:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

The deployed app reads that value through `st.secrets` and does not show the sidebar password field.

## Tests

```bash
python -m unittest tests.test_app -v
```

The tests mock `google-genai`. They do not call Gemini and do not need an API key.

## Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit UI, prompts, and Gemini calls |
| `requirements.txt` | `streamlit` and `google-genai` |
| `.streamlit/secrets.toml.example` | Secrets template for local use and deployment |
| `.streamlit/config.toml` | Dark theme used by the studio |
| `tests/test_app.py` | Prompt, client, and page tests |
