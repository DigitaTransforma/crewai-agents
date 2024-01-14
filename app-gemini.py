import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun


# Set gemini pro as llm
gemini = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.5,
                             google_api_key="") #add google_api_key in btw quote


#create searches using duck search
tool_search = DuckDuckGoSearchRun()

# Define your agents with roles and goals
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science in',
    backstory="""You are a Senior Research Analyst at a leading tech think tank.
    Your expertise lies in identifying emerging trends and technologies in AI and
    data science. You have a knack for dissecting complex data and presenting
    actionable insights.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini,
    tools=[
        tool_search
      ]
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Tech Content Strategist, known for your insightful
    and engaging articles on technology and innovation. With a deep understanding of
    the tech industry, you transform complex concepts into compelling narratives.""",
    verbose=True,
    allow_delegation=True,
    llm=gemini
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report. Your final answer MUST be a full analysis report""",
  agent=researcher
)

task2 = Task(
  description="""Using the insights from the researcher's report, develop an engaging linkedin
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future. Your final answer MUST be the full blog post of at least 3 paragraphs.""",
  agent=writer
)

# Create a Single Crew (This is a linkedin Crew)
linkedin_crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    process=Process.sequential
)

# Execution Flow
print("############################################")
print("Linkedin crew are now working on the task")
print("############################################")
linked_output = linkedin_crew.kickoff()
