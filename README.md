# 📄 AI Candidate Screening Agent & Web UI

An automated candidate screening application powered by **LangGraph** and **Google Gemini (gemini-3.6-flash)**, featuring an interactive **Gradio** web user interface. This repository houses a complete pipeline to automatically ingest candidate resumes/applications, analyze their qualifications, and route them to corresponding recruitment actions.

---

## 🛠️ System Architecture & Workflow

The core application uses a stateful agentic design implemented via **LangGraph**:

```
          [ START ]
              │
              ▼
   [ categorize_experience ] (Determines Entry/Mid/Senior level)
              │
              ▼
      [ assess_skillset ]    (Evaluates Python core skills matching)
              │
      [ route_app (Edge) ] ─── (Conditional Routing Logic)
         ╱    │    ╲
        ╱     │     ╲
       ▼      ▼      ▼
  [Shortlist] [Escalate] [Reject]
     (Match)  (Senior)  (No Match/Other)
       │      │      │
       └──────┼──────┘
              ▼
           [ END ]
```

### Nodes:
- **Categorize Experience**: Dynamically evaluates the application to classify the candidate as `Entry-level`, `Mid-level`, or `Senior-level`.
- **Assess Skillset**: Evaluates specific technical stack alignments (Python developer skills) to return a `Match` or `No Match` decision.
- **Action Nodes**:
  - **Shortlist**: Shortlists the candidate and flags for HR scheduling.
  - **Escalate**: Flags senior profiles with structural skill gaps for specialized recruiter review.
  - **Reject**: Automatically filters out profiles that do not match criteria.

---

## 🖥️ User Interface Overview (Gradio UI)

The repository includes a built-in Gradio Interface allowing users to easily test candidate screening from any web browser:

- **Input panel**: Simple, multiline textbox to paste a candidate's profile, application text, or resume summary.
- **Output panel**: Real-time display showing:
  - Classified **Experience Level**
  - Determined **Skill Match Result**
  - Automated **Next Action / Decision** with clear reasoning.

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
Make sure you have Python 3.10+ installed. Clone this repository:
```bash
git clone https://github.com/Abhishekbv6/agenticaiproject.git
cd agenticaiproject
```

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Set API Credentials
Set your Google Gemini API key as an environment variable:
```bash
# Linux/macOS
export GOOGLE_API_KEY="your-gemini-api-key-here"

# Windows Command Prompt
set GOOGLE_API_KEY="your-gemini-api-key-here"

# Windows PowerShell
$env:GOOGLE_API_KEY="your-gemini-api-key-here"
```

### 4. Run the Application
Launch the Gradio web interface locally:
```bash
python app.py
```
Once run, open the local address (usually `http://127.0.0.1:7860`) in your browser to interact with the screening agent!
