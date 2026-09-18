import os
import gradio as gr
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Securely load API Key (assumes environment variable is set on host machine)
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

# Workflow Nodes
def categorize_experience(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on the following job application, categorize the candidate as 'Entry-level', 'Mid-level' or 'Senior-level'\n"
        "Application : {application}"
    )
    chain = prompt | llm
    experience_level = chain.invoke({"application": state["application"]}).content
    return {"experience_level": experience_level}

def assess_skillset(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on the job application for a Python Developer, assess the candidate's skillset\n"
        "Respond with either 'Match' or 'No Match'\n"
        "Application : {application}"
    )
    chain = prompt | llm
    skill_match = chain.invoke({"application": state["application"]}).content
    return {"skill_match": skill_match}

def schedule_hr_interview(state: State) -> State:
    return {"response": "Candidate has been shortlisted for an HR interview."}

def escalate_to_recruiter(state: State) -> State:
    return {"response": "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: State) -> State:
    return {"response": "Candidate doesn't meet JD and has been rejected."}

# Routing Logic
def route_app(state: State) -> str:
    if state["skill_match"] == "Match":
        return "schedule_hr_interview"
    elif state["experience_level"] == "Senior-level":
        return "escalate_to_recruiter"
    else:
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
        
        exp_lvl = results.get('experience_level', '')
        if isinstance(exp_lvl, list):
            exp_lvl = "".join(part.get("text", "") for part in exp_lvl if isinstance(part, dict))
            
        sk_match = results.get('skill_match', '')
        if isinstance(sk_match, list):
            sk_match = "".join(part.get("text", "") for part in sk_match if isinstance(part, dict))
            
        resp = results.get('response', '')
        if isinstance(resp, list):
            resp = "".join(part.get("text", "") for part in resp if isinstance(part, dict))
            
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
