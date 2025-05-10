import os
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI
import data_info
import os

os.environ['OPENAI_API_KEY'] = data_info.open_ai_key
os.environ["SERPER_API_KEY"] = data_info.serp_key
# Use GPT-4o model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# Agent configuration

# Agent initialization
researcher_agent = Agent(
    role= "Senior Research Specialist for {topic}",
    goal="Find comprehensive and accurate information about {topic} with a focus on recent developments and key insights",
    backstory="You are an experienced research specialist with a talent for finding relevant information from various sources. You excel at organizing information in a clear and structured manner, making complex topics accessible to others.",
    llm=llm,
    verbose=True,
    tools=[SerperDevTool()]  # This tool can be used for searches
)

analyst_agent = Agent(
    role="Data Analyst and Report Writer for {topic}",
    goal="Analyze research findings and create a comprehensive, well-structured report that presents insights in a clear and engaging way",
    backstory="You are a skilled analyst with a background in data interpretation and technical writing. You have a talent for identifying patterns and extracting meaningful insights from research data, then communicating those insights effectively through well-crafted reports.",
    llm=llm,
    verbose=True
)

# Task initialization
research_task = Task(  description="""
            Conduct thorough research on {topic}. Focus on:
            1. Key concepts and definitions
            2. Historical development and recent trends
            3. Major challenges and opportunities
            4. Notable applications or case studies
            5. Future outlook and potential developments

            Make sure to organize your findings in a structured format with clear sections.
        """,
    agent=researcher_agent,
    expected_output =
                       """  A comprehensive research document with well-organized sections covering
                                  all the requested aspects of {topic}. Include specific facts, figures,
                                  and examples where relevant.""")



analysis_task = Task(
    description=""" Analyze the research findings and create a comprehensive report on {topic}. 
                Your report should:
                1. Begin with an executive summary"
                2. Include all key information from the research"
                3. Provide insightful analysis of trends and patterns"
                4. Offer recommendations or future considerations"
                5. Be formatted in a professional, easy-to-read style with clear headings""",
    agent=analyst_agent,
    context=[research_task],  # uses the output of research_task internally
    output_file='output/report.md',
    expected_output = """
           A polished, professional report on {topic} that presents the research
           findings with added analysis and insights. The report should be well-structured
           with an executive summary, main sections, and conclusion.""",
)


# Crew initialization
research_crew = Crew(
    agents=[researcher_agent, analyst_agent],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True
)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Run the crew
inputs = {
    'topic': 'Artificial Intelligence in Healthcare'
}

result = research_crew.kickoff(inputs=inputs)

# Print the result
print("\n\n=== FINAL REPORT ===\n\n")
print(result.raw)

print("\n\nReport has been saved to output/report.md")
