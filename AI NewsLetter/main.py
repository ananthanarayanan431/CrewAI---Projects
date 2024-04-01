import os

from crewai import Crew,Process
from langchain_openai.chat_models import ChatOpenAI
from agents import AINewsLetterAgents
from tasks import AINewsLetterTasks
from file_io import save_markdown

from dotenv import load_dotenv
load_dotenv()

from constant import OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

agents = AINewsLetterAgents()
tasks = AINewsLetterTasks()

OpenAI35 = ChatOpenAI(model="gpt-3.5-turbo")
editor = agents.editor_agent()
news_fetcher = agents.news_fetcher_agent()
news_analyzer = agents.news_analyzer_agent()
newsletter_complier = agents.newsletter_compiler_agent()

fectch_news_task = tasks.fectch_news_task(news_fetcher)
analyse_news_task = tasks.analyze_news_tasks(news_analyzer,[fectch_news_task])
complie_newsletter_task = tasks.complie_newsletter_task(
    newsletter_complier,[analyse_news_task],save_markdown
)

crew = Crew(
    agents = [editor,news_fetcher,news_analyzer,newsletter_complier],
    tasks = [fectch_news_task,analyse_news_task,complie_newsletter_task],
    process = Process.hierarchical,
    manager_llm=OpenAI35,
    verbose=4
)

results = crew.kickoff()

print("Crew Work Results: ")
print(results)

