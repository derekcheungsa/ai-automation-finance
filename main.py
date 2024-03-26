import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import BraveSearch
from tools.sec_tools import SecTools
#from tools.openbb_tools import OpenBBTools

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

# Get BRAVE_API_KEY from environment variables
# api_key = os.getenv('BRAVE_API_KEY')

#search_tool = BraveSearch.from_api_key(api_key=api_key,
#                                       search_kwargs={"count": 3})
#openbb_tool = OpenBBTools()

sec_tool = SecTools()

# Create a chat model
# Using service http://together.ai
#
llm = ChatOpenAI(model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
                 temperature=0.7,
                 api_key=os.getenv("OPENAI_API_KEY"),
                 base_url="https://api.together.xyz")

#llm_writer = ChatOpenAI(model="teknium/OpenHermes-2p5-Mistral-7B",
#                        temperature=0.7,
#                        base_url="https://api.together.xyz")

#llm = ChatMistralAI(model="mistral-medium", temperature=0.7)
#llm_writer = ChatAnthropic(model='claude-3-haiku-20240307')

# Define your agents with roles and goals
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover insights into the company Nvdia',
    backstory=
    """You work as a research analyst at Goldman Sachs, focusing on fundamental research for tech companies""",
    verbose=True,
    allow_delegation=False,
    tools=[SecTools.sec_nvda],
    llm=llm)

visionary = Agent(
    role='Visionary',
    goal='Deep thinking on the implications of an analysis',
    backstory=
    """you are a visionary technologist with a keen eye for identifying emerging trends and predicting their potential impact on various industries. Your ability to think critically and connect seemingly disparate dots allows you to anticipate disruptive technologies and their far-reaching implications.""",
    verbose=True,
    allow_delegation=False,
    llm=llm)

writer = Agent(
    role='Senior editor',
    goal='Writes professional quality articles that are easy to understand',
    backstory=
    """You are a details-oriented senior editor at the Wall Street Journal known for your insightful and engaging articles. You transform complex concepts into factual and impactful narratives.""",
    verbose=True,
    llm=llm,
    allow_delegation=True)

# Create tasks for your agents
task1 = Task(
    description=
    """please conduct a comprehensive analysis of Nvidia's latest SEC 10-K filing. The analysis should include the following key points:

  Business Overview: Briefly describe Nvidia's business model, its products and services, and its target market.

  Risk Factors: Identify and discuss the major risk factors that Nvidia has disclosed in its 10-K filing.

  Management's Discussion and Analysis (MD&A): Summarize the key points from the MD&A section, including any significant changes in operations, financial condition, or liquidity.

  Competitive Landscape: Discuss Nvidia's competitive position in its industry and how it compares to its major competitors.

  Future Outlook: Based on the information in the 10-K filing and your analysis, provide a brief outlook on Nvidia's future performance.

  Please ensure that all information is sourced from Nvidia's latest SEC 10-K filing and that the analysis is unbiased and factual.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher)

task2 = Task(
    description=
    """Using the insights provided by the Senior Research Analyst, think through deeply the future implications of the points that are made. Consider the following questions as you craft your response:

What are the current limitations or pain points that this technology mentioned in the Senior Research Analyst report could address?

How might this technology disrupt traditional business models and create new opportunities for innovation?

What are the potential risks and challenges associated with the adoption of this technology, and how might they be mitigated?

How could this technology impact consumers, employees, and society as a whole?

What are the long-term implications of this technology, and how might it shape the future of the industry?

Provide a detailed analysis of the technology's potential impact, backed by relevant examples, data, and insights. Your response should demonstrate your ability to think strategically, anticipate future trends, and articulate complex ideas in a clear and compelling manner.""",
    expected_output="Analysis report with deeper insights in implications",
    agent=visionary)

task3 = Task(
    description=
    """Using the insights provided by the Senior Research Analyst and Visionary,please craft an expertly styled report that is targeted towards the investor community. Make sure to also include the long-term implications insights that your co-worker, Visionary, has shared.  
    
    Please ensure that the report is written in a professional tone and style, and that all information is sourced from Nvidia's latest SEC 10-K filing. Write in a format and style worthy to be published in the wall street journal.""",
    expected_output=
    "A detailed comprehensive report on NVDIA that expertly presents the research done by your co-worker, Senior Research Analyst and Visionary",
    agent=writer)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, visionary, writer],
    tasks=[task1, task2, task3],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
