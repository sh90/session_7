#pip install streamlit
import streamlit as st
import os
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
import data_info

os.environ['OPENAI_API_KEY'] = data_info.open_ai_key
gpt4o_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

st.set_page_config(page_title="AI Debate: Remote Work", layout="wide")
st.title("🤖 AI Debate: Should Remote Work Be the Default in Tech?")

topic = st.text_input("Debate Topic", "Should remote work be the default in tech companies?")

if st.button("Start Debate"):
    with st.spinner("Running debate... please wait"):

        # Define agents
        pro_agent = Agent(
            role='Affirmative Debater',
            goal='Support the position that remote work should be the default in tech companies',
            backstory='An AI trained on productivity, tech industry trends, and remote culture benefits.',
            llm=gpt4o_mini,
            verbose=False
        )

        con_agent = Agent(
            role='Negative Debater',
            goal='Refute the idea that remote work should be the default in tech companies',
            backstory='An AI specializing in organizational behavior, in-office collaboration, and leadership dynamics.',
            llm=gpt4o_mini,
            verbose=False
        )

        judge_agent = Agent(
            role='Debate Judge',
            goal='Provide constructive feedback on arguments and determine the stronger case',
            backstory='An impartial moderator trained in critical thinking, communication, and decision-making.',
            llm=gpt4o_mini,
            verbose=False
        )

        # Define tasks
        task1 = Task(
            description=f"Present an opening argument *for* the topic: '{topic}'.",
            expected_output="A compelling opening statement with 2-3 key points.",
            agent=pro_agent
        )

        task2 = Task(
            description=f"Present an opening argument *against* the topic: '{topic}'.",
            expected_output="A compelling opening statement with 2-3 key counterpoints.",
            agent=con_agent
        )

        task3 = Task(
            description="Evaluate the opening statements from both sides. Provide feedback on strengths and gaps.",
            context=[task1, task2],
            expected_output="Feedback for both agents with suggestions for strengthening rebuttals.",
            agent=judge_agent
        )

        task4 = Task(
            description="Write a rebuttal to ConAgent’s points based on the judge's feedback.",
            context=[task2, task3],
            expected_output="A thoughtful rebuttal incorporating judge's suggestions.",
            agent=pro_agent
        )

        task5 = Task(
            description="Write a rebuttal to ProAgent’s points based on the judge's feedback.",
            context=[task1, task3],
            expected_output="A thoughtful rebuttal incorporating judge's suggestions.",
            agent=con_agent
        )

        task6 = Task(
            description="Evaluate rebuttals and provide a final verdict.",
            context=[task4, task5],
            expected_output="Final verdict with reasoning.",
            agent=judge_agent
        )

        crew = Crew(
            agents=[pro_agent, con_agent, judge_agent],
            tasks=[task1, task2, task3, task4, task5, task6],
            verbose=False
        )

        result = crew.kickoff()

    # Display output in sections
    st.subheader("🏁 Final Verdict")
    st.success(result)

    st.subheader("📜 Full Debate Transcript")
    st.markdown(f"**Opening Argument (Pro):**\n\n{task1.output}")
    st.markdown(f"**Opening Argument (Con):**\n\n{task2.output}")
    st.markdown(f"**Judge's Feedback:**\n\n{task3.output}")
    st.markdown(f"**Rebuttal (Pro):**\n\n{task4.output}")
    st.markdown(f"**Rebuttal (Con):**\n\n{task5.output}")
