from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import data_info
import os
os.environ['OPENAI_API_KEY'] = data_info.open_ai_key
os.environ["SERPER_API_KEY"] = data_info.serp_key
# ----------- LLM Setup -----------
llm = ChatOpenAI(
    model="gpt-4o-mini",  # 'gpt-4o' includes gpt-4o-mini under the hood
    temperature=0.1
)
os.environ['OPENAI_API_KEY'] = data_info.open_ai_key
# ----------- SerpAPI Tool Setup -----------
search_tool = SerperDevTool()

# ----------- Agent Setup -----------
babel_agent = Agent(
    role="Trend Analyst for Babel Tonic",
    goal="Research and synthesize key flavor, packaging, and functional ingredient trends from top industry sources",
    backstory="A specialized AI assistant for the beverage and spirits industry. Helps Babel Tonic and clients identify market trends and opportunities.",
    tools=[search_tool],
    llm=llm,
    verbose=True
)

# ----------- Task Setup -----------
task = Task(
    description="""
    Use real-time search to find the latest articles and trend reports of 2025 about the beverage and spirits industry, especially regarding:
    - Flavor innovation (e.g., botanicals, adaptogens)
    - Functional claims (e.g., mood boosting, nootropics)
    - Packaging or branding strategies

    Categorize findings under Flavor, Function, and Brand/Packaging.
    Provide a concise summary suitable for inclusion in a client-facing report.
    Format suggestions for follow-up actions or product ideas.
    """,
    agent=babel_agent,
    expected_output="A well-structured report with 3 sections: Flavor, Function, and Brand/Packaging, followed by 2-3 product or strategy ideas.",

)

# ----------- Crew Run -----------
crew = Crew(
    agents=[babel_agent],
    tasks=[task],
    verbose=True
)

if __name__ == '__main__':
    result = crew.kickoff()  # Use kickoff() instead of run() for latest CrewAI versions
    with open("babel_tonic_report.txt", "w") as f:
        f.write(str(result))
    print("✅ Report generated as babel_tonic_report.txt")
