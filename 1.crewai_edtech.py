#####
# pip install crewai crewai-tools
## Sequential and independent execution of Agents
##  In the example I provided, the agents are working in isolation — meaning each agent handles its task independently,
# without directly exchanging information with other agents during the process.
####
##pip install crewai crewai-tools streamlit
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
import data_info
import os

os.environ['OPENAI_API_KEY'] = data_info.open_ai_key
# -------------------------
# 1. Define LLM
# -------------------------
# Use GPT-4o model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
# -------------------------
# 2. Define Agents
# -------------------------
# Define agents
content_guide = Agent(
    role="Educational Content Guide",
    goal="Provide relevant and engaging learning resources for students",
    backstory="An AI trained on curriculum and learning methods to suggest helpful content based on student needs.",
    llm=llm
)

student_support = Agent(
    role="Student Support Assistant",
    goal="Help students with academic or platform-related concerns",
    backstory="Handles student inquiries about the EdTech platform, course access, deadlines, or grading.",
    llm=llm
)

feedback_analyst = Agent(
    role="Feedback Classifier",
    goal="Extract and classify sentiment and feedback from student messages",
    backstory="Helps product and content teams understand how students feel and what they want improved.",
    llm=llm
)
# Agent -> Task
# -------------------------
# 3. Define Task
# -------------------------
support_task = Task(
    description="Respond to the student's question or concern: '{{student_message}}'. Be empathetic, informative, and friendly.",
    expected_output="A clear and kind support message addressing the student's need.",
    agent=student_support
)

content_task = Task(
    description="Based on this message: '{{student_message}}', suggest relevant study materials (e.g., videos, articles, practice).",
    expected_output="At least one high-quality resource recommendation tailored to the topic.",
    agent=content_guide
)

feedback_task = Task(
    description="Extract feedback from the message: '{{student_message}}' and classify it (e.g., Praise, Complaint, Feature Request, Confusion, N/A).",
    expected_output="Feedback category and a short summary.",
    agent=feedback_analyst
)

# Assemble the crew
crew = Crew(
    agents=[student_support, content_guide, feedback_analyst],
    tasks=[support_task, content_task, feedback_task],
    verbose=True
)

# Example student message input
student_input = "I'm struggling with calculus integrals and I wish there were more interactive examples. Also, the quiz timer is too short."

# Kickoff the crew with dynamic input
result = crew.kickoff(inputs={
    "student_message": student_input})

print("\n=== FINAL OUTPUT ===\n")
print(result)
