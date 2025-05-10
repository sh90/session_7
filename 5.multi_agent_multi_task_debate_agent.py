from crewai import Agent, Task, Crew
import os
import data_info
from langchain.chat_models import ChatOpenAI
os.environ["OPENAI_API_KEY"] = data_info.open_ai_key
gpt4o_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
# Define agents
pro_agent = Agent(
    role='Affirmative Debater',
    goal='Support the position that remote work should be the default in tech companies',
    backstory='An AI trained on productivity, tech industry trends, and remote culture benefits.',
    llm=gpt4o_mini,
    verbose=True
)

con_agent = Agent(
    role='Negative Debater',
    goal='Refute the idea that remote work should be the default in tech companies',
    backstory='An AI specializing in organizational behavior, in-office collaboration, and leadership dynamics.',
    llm=gpt4o_mini,
    verbose=True
)

judge_agent = Agent(
    role='Debate Judge',
    goal='Provide constructive feedback on arguments and determine the stronger case',
    backstory='An impartial moderator trained in critical thinking, communication, and decision-making.',
    llm=gpt4o_mini,
    verbose=True
)

# Round 1: Opening Arguments
task1 = Task(
    description="Present an opening argument *for* remote work being the default in tech companies.",
    expected_output="A compelling opening statement with 2-3 key points.",
    agent=pro_agent
)

task2 = Task(
    description="Present an opening argument *against* remote work being the default in tech companies.",
    expected_output="A compelling opening statement with 2-3 key counterpoints.",
    agent=con_agent
)

# Round 2: Judge Feedback
task3 = Task(
    description="Evaluate the opening statements from both sides. Provide feedback on strengths and gaps for each agent.",
    context=[task1, task2],
    expected_output="Feedback for both agents with suggestions for strengthening rebuttals.",
    agent=judge_agent
)

# Round 3: Rebuttals
task4 = Task(
    description="Based on the judge's feedback, write a rebuttal to the ConAgent’s points.",
    context=[task2, task3],
    expected_output="A thoughtful rebuttal incorporating judge's suggestions.",
    agent=pro_agent
)

task5 = Task(
    description="Based on the judge's feedback, write a rebuttal to the ProAgent’s points.",
    context=[task1, task3],
    expected_output="A thoughtful rebuttal incorporating judge's suggestions.",
    agent=con_agent
)

# Final Judgement
task6 = Task(
    description="Evaluate rebuttals and provide a final verdict: which side made a stronger overall case and why.",
    context=[task4, task5],
    expected_output="Final verdict with reasoning.",
    agent=judge_agent
)

# Define the crew and run the workflow
crew = Crew(
    agents=[pro_agent, con_agent, judge_agent],
    tasks=[task1, task2, task3, task4, task5, task6],
    verbose=True
)

result = crew.kickoff()
print("\nFinal Verdict:\n", result)
