import os
import gradio as gr
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Securely load API Key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("Warning: GOOGLE_API_KEY environment variable not found.")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)

# Define LangGraph State
class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str

# Helper function to extract clean text from any structured or list outputs
def clean_model_output(value):
    if not value:
        return ""
    if isinstance(value, list):
        extracted = []
        for part in value:
            if isinstance(part, dict):
                extracted.append(part.get("text", ""))
            elif isinstance(part, str):
                extracted.append(part)
        value = "".join(extracted)
    return str(value).replace("**", "").strip()

# Workflow Nodes
def categorize_experience(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on the following job application, categorize the candidate as 'Entry-level', 'Mid-level' or 'Senior-level'\n"
        "Application : {application}"
    )
    chain = prompt | llm
    raw_response = chain.invoke({"application": state["application"]}).content
    experience_level = clean_model_output(raw_response)
    return {"experience_level": experience_level}

def assess_skillset(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on the job application for a Python Developer, assess the candidate's skillset\n"
        "Respond with either 'Match' or 'No Match'\n"
        "Application : {application}"
    )
    chain = prompt | llm
    raw_response = chain.invoke({"application": state["application"]}).content
    skill_match = clean_model_output(raw_response)
    return {"skill_match": skill_match}

def schedule_hr_interview(state: State) -> State:
    return {"response": "Candidate has been shortlisted for an HR interview."}

def escalate_to_recruiter(state: State) -> State:
    return {"response": "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: State) -> State:
    return {"response": "Candidate doesn't meet JD and has been rejected."}

# Routing Logic with Robust Substring Comparisons
def route_app(state: State) -> str:
    skills = state["skill_match"].lower()
    experience = state["experience_level"].lower()
    
    # Check if 'no match' is explicitly specified first
    if "no match" in skills:
        if "senior" in experience:
            return "escalate_to_recruiter"
        else:
            return "reject_application"
    elif "match" in skills:
        return "schedule_hr_interview"
    else:
        # Default fallback
        if "senior" in experience:
            return "escalate_to_recruiter"
        return "reject_application"

# Assemble LangGraph Workflow
workflow = StateGraph(State)
workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("assess_skillset", assess_skillset)
workflow.add_node("schedule_hr_interview", schedule_hr_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)

workflow.add_edge("categorize_experience", "assess_skillset")
workflow.add_conditional_edges("assess_skillset", route_app)
workflow.add_edge(START, "categorize_experience")
workflow.add_edge("escalate_to_recruiter", END)
workflow.add_edge("reject_application", END)
workflow.add_edge("schedule_hr_interview", END)

app = workflow.compile()

def screen_candidate_ui(application_text):
    try: 
        results = app.invoke({"application": application_text})
        
        exp_lvl = clean_model_output(results.get('experience_level', ''))
        sk_match = clean_model_output(results.get('skill_match', ''))
        resp = clean_model_output(results.get('response', ''))
        
        return exp_lvl, sk_match, resp
    except Exception as e:
        return f"Error: {str(e)}", "Error", "Error"

# Gradio Web Interface
with gr.Blocks() as demo:
    gr.Markdown("# 📄 AI Candidate Screening Agent")
    gr.Markdown("Enter the candidate's resume/application below to run the screening workflow.")
    
    with gr.Row():
        with gr.Column(scale=2):
            app_input = gr.Textbox(label="Job Application / Resume Text", lines=8)
            submit_btn = gr.Button("Screen Candidate", variant="primary")
        with gr.Column(scale=1):
            output_exp = gr.Textbox(label="Experience Level", interactive=False)
            output_skills = gr.Textbox(label="Skill Match Result", interactive=False)
            output_action = gr.Textbox(label="Next Action / Decision", interactive=False)
            
    submit_btn.click(
        fn=screen_candidate_ui,
        inputs=[app_input],
        outputs=[output_exp, output_skills, output_action]
    )

if __name__ == "__main__":
    demo.launch()
