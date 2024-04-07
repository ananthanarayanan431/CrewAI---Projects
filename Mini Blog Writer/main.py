import constant

agent_finishes  = []

import json
from typing import Union, List, Tuple, Dict
from langchain.schema import AgentFinish
from crewai import Crew, Agent, Task, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import os

os.environ['GOOGLE_API_KEY'] = constant.GOOGLE_API_KEY

from datetime import datetime
from random import randint
import requests
from langchain.tools import tool

from crewai import Agent, Task, Crew, Process


call_number = 0

def print_agent_output(agent_output: Union[str, List[Tuple[Dict, str]], AgentFinish], agent_name: str = 'Generic call'):
    global call_number
    call_number += 1
    with open("crew_callback_logs.txt", "a") as log_file:
        if isinstance(agent_output, str):
            try:
                agent_output = json.loads(agent_output)
            except json.JSONDecodeError:
                pass

        if isinstance(agent_output, list) and all(isinstance(item, tuple) for item in agent_output):
            print(f"-{call_number}----Dict------------------------------------------", file=log_file)
            for action, description in agent_output:
                print(f"Agent Name: {agent_name}", file=log_file)
                print(f"Tool used: {getattr(action, 'tool', 'Unknown')}", file=log_file)
                print(f"Tool input: {getattr(action, 'tool_input', 'Unknown')}", file=log_file)
                print(f"Action log: {getattr(action, 'log', 'Unknown')}", file=log_file)
                print(f"Description: {description}", file=log_file)
                print("--------------------------------------------------", file=log_file)

        elif isinstance(agent_output, AgentFinish):
            print(f"-{call_number}----AgentFinish---------------------------------------", file=log_file)
            print(f"Agent Name: {agent_name}", file=log_file)
            agent_finishes.append(agent_output)
            output = agent_output.return_values
            print(f"AgentFinish Output: {output['output']}", file=log_file)
            print("--------------------------------------------------", file=log_file)

        else:
            print(f"-{call_number}-Unknown format of agent_output:", file=log_file)
            print(type(agent_output), file=log_file)
            print(agent_output, file=log_file)


OpenAIGPT35=ChatGoogleGenerativeAI(model="gemini-pro")

@tool("save_content")
def save_content(task_output):
    """Useful to save content to a markdown file"""
    print('in the save markdown tool')
    today_date = datetime.now().strftime('%Y-%m-%d')
    filename = f"{today_date}_{randint(0,100)}.md"
    with open(filename, 'w') as file:
        file.write(task_output)

    print(f"Blog post saved as {filename}")

    return f"Blog post saved as {filename}, please tell the user we are finished"

search_tool = DuckDuckGoSearchRun()

human_tools = load_tools(["human"])

info_getter = Agent(
    role='AI Research Specialist',
    goal='Leverage advanced search techniques to surface the most relevant, credible, and impactful information on AI and data science breakthroughs',
    backstory="""As a top AI Research Specialist at a renowned technology
    research institute, you have honed your skills in crafting sophisticated
    search queries, filtering information from trusted sources, and synthesizing
    key insights. You have the ability to take a topic suggested by a human and
    rewrite multiple searches for that topic to get the best results overall.

    Your extensive knowledge of AI and data science, combined
    with your mastery of machine learning and Large Language models, allows you
    to unearth groundbreaking research that others often overlook. You excel
    at critically evaluating the credibility and potential
    impact of new developments, enabling you to curate a focused feed of the most
    significant advances. Your talent for clear and concise summarization helps
    you distill complex technical concepts into easily digestible executive
    briefings and reports. With a track record of consistently identifying
    paradigm-shifting innovations before they hit the mainstream, you have become
    the go-to expert for keeping your organization at the forefront of the AI revolution.""",
    verbose=True,
    allow_delegation=False,
    llm=OpenAIGPT35,
    max_iter=5,
    memory=True,
    step_callback=lambda x: print_agent_output(x,"Senior Research Analyst Agent"),
    tools=[search_tool]+human_tools,
)

writer = Agent(
    role='Tech Content Writer and rewriter',
    goal='Generate compelling content via first drafts and subsequent polishing to get a final product. ',
    backstory="""As a renowned Tech Content Strategist, you have a gift for transforming complex technical
    concepts into captivating and easily digestible articles. Your extensive knowledge of the tech
    industry allows you to identify the most compelling angles and craft narratives that resonate
    with a wide audience.

    Your writing prowess extends beyond simply conveying information; you have a knack for restructuring
    and formatting content to enhance readability and engagement. Whether it's breaking down intricate
    ideas into clear, concise paragraphs or organizing key points into visually appealing lists,
    your articles are a masterclass in effective communication.

    Some of your signature writing techniques include:

    Utilizing subheadings and bullet points to break up long passages and improve scannability

    Employing analogies and real-world examples to simplify complex technical concepts

    Incorporating visuals, such as diagrams and infographics, to supplement the written content

    Varying sentence structure and length to maintain a dynamic flow throughout the article

    Crafting compelling introductions and conclusions that leave a lasting impact on readers

    Your ability to rewrite and polish rough drafts into publishable masterpieces is unparalleled.
    You have a meticulous eye for detail and a commitment to delivering content that not only informs
    but also engages and inspires. With your expertise, even the most technical and dry subject matter
    can be transformed into a riveting read.""",
    llm=OpenAIGPT35,
    verbose=True,
    max_iter=5,
    memory=True,
    step_callback=lambda x: print_agent_output(x,"Tech Content Writer and rewriter"),
    allow_delegation=True,
)

archiver = Agent(
    role='File Archiver',
    goal='Take in information and write it to a Markdown file',
    backstory="""You are a efficient and simple agent that gets data and saves it to a markdown file. in a quick and efficient manner""",
    llm=OpenAIGPT35,
    verbose=True,
    step_callback=lambda x: print_agent_output(x,"Archiver Agent"),
    tools=[save_content],
)

get_source_material = Task(
  description=f"""Conduct a comprehensive analysis of the latest news advancements in an area
  determined by the human. ASK THE HUMAN for the area of interest.\n
  The current time is {datetime.now()}. Focus on recent events related to the human's topic.
  Identify key facts and useful information related to the Human's topic

  Compile you results into a useful and helpful report for the writer to use to write an article""",
  expected_output='A comprehensive full report on the latest AI advancements in the specified human topic, leave nothing out',
  agent=info_getter
)

write_the_content = Task(
  description="""Using the source material from the research specialist's report,
  develop a nicely formated article that is brief, to the point and highlights the
  most significant information and advancements.
  Your article should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future (both for research, but also for industy).
  DON'T overly 'Hype' the topic. Be factual and clear.
  Your final answer MUST be a full article post of at least 3 paragraphs and should contain
  a set of bullet points with the key facts at the end for a summary""",
  expected_output="""A compelling 3 paragraphs article with a set of bullet points with
  the key facts at the end for a summay. This should all be formated as markdown in an easy readable manner""",
  agent=writer
)


saving_the_output = Task(
  description="""Taking the post created by the writer, take this and save it to a markdown file.
  Your final answer MUST be a response must be showing that the file was saved .""",
  expected_output='A saved file name',
  agent=archiver
)

crew = Crew(
    agents=[info_getter, writer, archiver],
    tasks=[get_source_material,
           write_the_content,
           saving_the_output],
    verbose=2,
    process=Process.sequential,
    full_output=True,
    share_crew=False,
    step_callback=lambda x: print_agent_output(x,"MasterCrew Agent")
)

results = crew.kickoff()

print("Crew Work Results:")
print(results)