#####
## Sequential and independent execution of Agents
####
from crewai import Agent, Task, Crew
import data_info
from langchain.chat_models import ChatOpenAI  # Instead of OpenAI
import os

os.environ['OPENAI_API_KEY'] = data_info.open_ai_key
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# -------------------------
# 1. Define Agents
# -------------------------
support_agent = Agent(
    role="Customer Support Specialist",
    goal="Provide accurate, empathetic support responses to customers",
    backstory="An experienced customer service professional trained to handle billing, subscription, and technical issues.",
    llm=llm
)

escalation_agent = Agent(
    role="Escalation Manager",
    goal="Evaluate urgency and escalate critical issues appropriately",
    backstory="Trained in incident triage, prioritizing critical issues like fraud, legal risks, or high-value complaints.",
    llm=llm
)

feedback_agent = Agent(
    role="Customer Feedback Analyst",
    goal="Extract and categorize useful product feedback from messages",
    backstory="Monitors user messages for patterns and insights to inform the product roadmap.",
    llm=llm
)

# -------------------------
# 2. Define Tasks with Placeholders
# -------------------------
support_task = Task(
    description="Generate a professional and friendly support response to the message: '{{customer_message}}'.",
    expected_output="A helpful response addressing the customer issue clearly and politely.",
    agent=support_agent
)

escalation_task = Task(
    description="Analyze the following message for urgency, red flags, or legal/risk triggers: '{{customer_message}}'. "
                "Decide if escalation is needed, and if so, recommend a department.",
    expected_output="YES/NO escalation decision and a brief reason or destination (e.g., Legal, Billing).",
    agent=escalation_agent
)

feedback_task = Task(
    description="Extract any feedback or sentiment from the customer message: '{{customer_message}}'. "
                "Classify it as a bug, complaint, suggestion, praise, or N/A.",
    expected_output="Feedback category and summary (e.g., 'Complaint - Billing confusion').",
    agent=feedback_agent
)

# -------------------------
# 3. Build the Crew
# -------------------------
crew = Crew(
    agents=[support_agent, escalation_agent, feedback_agent],
    tasks=[support_task, escalation_task, feedback_task],
    verbose=True
)

# -------------------------
# 4. Run the Crew with Dynamic Input
# -------------------------
customer_input = "Hi, I was double charged for my plan this month and I haven’t heard back from support. Please fix this ASAP or I’ll cancel."

result = crew.kickoff(inputs={
    "customer_message": customer_input
})

print("\n=== FINAL OUTPUT ===\n")
print(result)
