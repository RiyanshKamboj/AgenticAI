from crewai import Agent, LLM
from tools import yt_tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()

# Configure Gemini API
import google.generativeai as genai
import os

# Retrieve Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set it in your environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize LLM with Gemini model
my_LLM =ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                               verbose=True,
                               temperature=0.5,
                               GOOGLE_API_KEY=GOOGLE_API_KEY)
    

# Create a senior blog content researcher agent
blog_researcher = Agent(
    role='Blog researcher from YouTube videos',
    goal='Get the relevant video content for the topic {topic} from the YouTube channel.',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI, Data Science, Machine Learning, and GenAI, "
        "and providing suggestions."
    ),
    tools=[yt_tool],
    llm=my_LLM,
    allow_delegation=True
)

# Create a senior blog writer agent with YT tool
blog_writer = Agent(
    role='Blog writer',
    goal='Narrate compelling tech stories about the video {topic} from the YouTube channel.',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft engaging narratives that captivate "
        "and educate, bringing new discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    llm=my_LLM,
    allow_delegation=False
)
