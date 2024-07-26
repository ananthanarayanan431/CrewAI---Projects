from agents import MarketingAnalysisAgents
from tasks import MarketingAnalysisTasks
from crewai import Crew


import os,constant

os.environ['GROQ_API_KEY']=constant.GROQ_API_KEY
os.environ['SERPER_API_KEY']=constant.SERPER_API_KEY

tasks = MarketingAnalysisTasks()
agents = MarketingAnalysisAgents()

product_website = "https://www.launchventures.co/"
product_details = """
Launch Ventures partners with promising teams to build and launch their products from concept to scale. We provide strategic product consulting, design the product experience, engineer products from MVP to scale and setup analytics to provide business insights.
We started out as a niche engineering team, building and scaling software products for a range of startups to Fortune 500 companies. We have built and scaled over a 100 products, some of which are transforming the lives of people for the better and others disrupting complete industries.
We are a small nimble closely knit team who cares deeply about the quality and success of the products we create and the partners we work with. Doing this makes us valuable and brings us joy. Every year, we seek to partner with a few solid teams applying software to build products or businesses with a meaningful impact and we bring to bear our experience and passion to accelerate their growth. We want to be Great, not big!
"""


product_competitor_agent = agents.product_competitor_agent()
strategy_planner_agent = agents.strategy_planner_agent()
creative_agent = agents.creative_content_creator_agent()

website_analysis = tasks.product_analysis(product_competitor_agent, product_website, product_details)

market_analysis = tasks.competitor_analysis(product_competitor_agent, product_website, product_details)

campaign_development = tasks.campaign_development(strategy_planner_agent, product_website, product_details)

write_copy = tasks.instagram_ad_copy(creative_agent)

copy_crew = Crew(
    agents=[
        product_competitor_agent,
        strategy_planner_agent,
        creative_agent
    ],
    tasks=[
        website_analysis,
        market_analysis,
        campaign_development,
        write_copy
    ],
    verbose=1,
    # max_rpm=1
)

ad_copy = copy_crew.kickoff()


senior_photographer = agents.senior_photographer_agent()
chief_creative_diretor = agents.chief_creative_diretor_agent()

take_photo = tasks.take_photograph_task(senior_photographer, ad_copy, product_website, product_details)
approve_photo = tasks.review_photo(chief_creative_diretor, product_website, product_details)

image_crew = Crew(
    agents=[
        senior_photographer,
        chief_creative_diretor
    ],
    tasks=[
        take_photo,
        approve_photo
    ],
    verbose=1,
    # max_rpm=1
    # Remove this when running locally. This helps prevent rate limiting with groq.
)

image = image_crew.kickoff()


print("\n\n########################")
print("## Here is the result")
print("########################\n")
print("Your post copy:")
print(ad_copy)
print("'\n\nYour midjourney description:")
print(image)