import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "" #add huggingface api token in btw the quote

from langchain_community.llms import HuggingFaceHub
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_tokens":4096}
)

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
    llm=llm,
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
    llm=llm
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
