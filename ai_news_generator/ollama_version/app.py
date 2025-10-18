import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from langchain_community.llms import Ollama
import json
import urllib.request
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GNewsTool(BaseTool):
    name: str = "GNews Search Tool"
    description: str = "A tool for performing real-time news searches using the GNews API. Use this to find the latest articles, headlines, and general web information on a specific query."

    def _run(self, query: str) -> str:
        # Check for placeholder key before running the tool
        if os.getenv("GNEWS_API_KEY") == "YOUR_GNEWS_API_KEY":
            return "GNews Search Error: API key is not set."

        # Safely encode the query string for use in a URL
        safe_query = quote_plus(query)

        # Construct the URL dynamically (max=10 results)
        url = f"https://gnews.io/api/v4/search?q={safe_query}&lang=en&max=5&apikey={os.getenv("GNEWS_API_KEY")}"

        try:
            # Use urllib.request to fetch the data
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode("utf-8"))
                articles = data.get("articles", [])

            if not articles:
                return f"No news articles found for the query: {query}"

            # Format the news articles into a string the agent can easily read
            results = []
            for i, article in enumerate(articles):
                results.append(
                    f"Article {i+1}:\n"
                    f"  Title: {article.get('title')}\n"
                    f"  Description: {article.get('description', 'N/A')}\n"
                    f"  Source: {article.get('source', {}).get('name', 'N/A')}\n"
                    f"  URL: {article.get('url', 'N/A')}\n"
                )

            return "GNews Search Results:\n\n" + "\n".join(results)

        except urllib.error.HTTPError as e:
            # Handle API-specific errors
            return f"GNews API Error (HTTP {e.code}): Could not retrieve results for '{query}'. Check API key and usage limits."
        except Exception as e:
            # Handle general errors
            return f"An unexpected error occurred during GNews search for '{query}': {str(e)}"



# Streamlit page config
st.set_page_config(page_title="AI News Generator", page_icon="📰", layout="wide")

# Title and description
st.title(f"🤖 AI News Generator, powered by CrewAI and **{os.getenv('OLLAMA_MODEL_NAME')}**")
st.markdown("Generate comprehensive blog posts about any topic using AI agents.")

# Sidebar
with st.sidebar:
    st.header("Content Settings")

    # Make the text input take up more space
    topic = st.text_area(
        "Enter your topic",
        height=100,
        placeholder="Enter the topic you want to generate content about..."
    )

    # Add more sidebar controls if needed
    st.markdown("### Advanced Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    # Add some spacing
    st.markdown("---")

    # Make the generate button more prominent in the sidebar
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)

    # Add some helpful information
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Enter your desired topic in the text area above
        2. Adjust the temperature if needed (higher = more creative)
        3. Click 'Generate Content' to start
        4. Wait for the AI to generate your article
        5. Download the result as a markdown file
        """)

def generate_content(topic):
    llm = LLM(
        model=os.getenv("OLLAMA_MODEL_NAME"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
        temperature=temperature
    )

    search_tool = GNewsTool()

    # First Agent: Senior Research Analyst
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal=f"Research, analyze, and synthesize comprehensive information on {topic} from reliable web sources",
        backstory="You're an expert research analyst with advanced web research skills. "
                  "You excel at finding, analyzing, and synthesizing information from "
                  "across the internet using search tools. You're skilled at "
                  "distinguishing reliable sources from unreliable ones, "
                  "fact-checking, cross-referencing information, and "
                  "identifying key patterns and insights. You provide "
                  "well-organized research briefs with proper citations "
                  "and source verification. Your analysis includes both "
                  "raw data and interpreted insights, making complex "
                  "information accessible and actionable.",
        allow_delegation=False,
        verbose=True,
        tools=[search_tool],
        llm=llm
    )

    # Second Agent: Content Writer
    content_writer = Agent(
        role="Content Writer",
        goal="Transform research findings into engaging blog posts while maintaining accuracy",
        backstory="You're a skilled content writer specialized in creating "
                  "engaging, accessible content from technical research. "
                  "You work closely with the Senior Research Analyst and excel at maintaining the perfect "
                  "balance between informative and entertaining writing, "
                  "while ensuring all facts and citations from the research "
                  "are properly incorporated. You have a talent for making "
                  "complex topics approachable without oversimplifying them.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    # Research Task
    research_task = Task(
        description=("""
            1. Conduct comprehensive research on {topic} including:
                - Recent developments and news
                - Key industry trends and innovations
                - Expert opinions and analyses
                - Statistical data and market insights
            2. Evaluate source credibility and fact-check all information
            3. Organize findings into a structured research brief
            4. Include all relevant citations and sources
        """),
        expected_output="""A detailed research report containing:
            - Executive summary of key findings
            - Comprehensive analysis of current trends and developments
            - List of verified facts and statistics
            - All citations and links to original sources
            - Clear categorization of main themes and patterns
            Please format with clear sections and bullet points for easy reference.""",
        agent=senior_research_analyst
    )

    # Writing Task
    writing_task = Task(
        description=("""
            Using the research brief provided, create an engaging blog post that:
            1. Transforms technical information into accessible content
            2. Maintains all factual accuracy and citations from the research
            3. Includes:
                - Attention-grabbing introduction
                - Well-structured body sections with clear headings
                - Compelling conclusion
            4. Preserves all source citations in [Source: URL] format
            5. Includes a References section at the end
        """),
        expected_output="""A polished blog post in markdown format that:
            - Engages readers while maintaining accuracy
            - Contains properly structured sections
            - Includes Inline citations hyperlinked to the original source url
            - Presents information in an accessible yet informative way
            - Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections""",
        agent=content_writer
    )

    # Create Crew
    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    return crew.kickoff(inputs={"topic": topic})

# Main content area
if generate_button:
    with st.spinner('Generating content... This may take a moment.'):
        try:
            result = generate_content(topic)
            st.markdown("### Generated Content")
            st.markdown(result)

            # Add download button
            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown(f"Built with CrewAI, Streamlit and powered by **{os.getenv('OLLAMA_MODEL_NAME')}**")