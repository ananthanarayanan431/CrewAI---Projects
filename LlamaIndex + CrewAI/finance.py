
import os 
from constant import OPENAI_API_KEY_PAID,GROQ_API_KEY
from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool

os.environ['OPENAI_API_KEY']=OPENAI_API_KEY_PAID
os.environ['GROQ_API_KEY']=GROQ_API_KEY
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,load_index_from_storage
from llama_index.core import ServiceContext,Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq.base import Groq

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import set_global_service_context
from llama_index.core.storage import StorageContext

reader=SimpleDirectoryReader(input_files=['mastek.pdf'])
docs=reader.load_data()

llm=Groq(model="llama3-8b-8192")
service_context = ServiceContext.from_defaults(llm=llm)
Settings.embed_model = OpenAIEmbedding()
set_global_service_context(service_context)

storage_path = "./vectorstore"

if not os.path.exists(storage_path):
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=storage_path)
else:
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)

query_tool=LlamaIndexTool.from_query_engine(
    query_engine,
    name="Mastek Finacial Report",
    description="Use this tool to look up the finacial report of Mastek in 2023"
)
print("----Agent Worked Started----")

researcher=Agent(
    role="Senior Financial Analyst",
    goal="Uncover insights about different tech companies",
    backstory="""You work at an asset management firm.
              Your goal is to understand tech stocks like Mastek
              Perform comparative Analysis of growth over the Years""",
    verbose=True,
    allow_delegation=False,
    tools=[query_tool],
)

writer=Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content on tech advancement",
    backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
                You transform complex concepts into compelling narrative""",
    verbose=True,
    allow_delegation=False,
)

task1 = Task(
    description="""Conduct a comprehensive analysis of Mastek's growth in 2023 compare to previous years.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher,
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the headwinds that Mastek faces.
  Your post should be informative yet accessible, catering to a casual audience.
  Make it sound cool, avoid complex words.""",
    expected_output="Full blog post of at least 4 paragraphs",
    agent=writer,
)

crew=Crew(
    agents=[researcher,writer],
    tasks=[task1,task2],
    verbose=2
)
result=crew.kickoff()
print(result)

