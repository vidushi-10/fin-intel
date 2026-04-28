import streamlit as st
import os
import time
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool

# --- 1. SETUP ---
st.set_page_config(page_title="Fin-Intel 2026 Pro", page_icon="📊", layout="wide")

# Your Provided Key
api_key = "AIzaSyCR56-g9T4ft9qarvNr6aNV2DnkbnbUE40"
os.environ["GOOGLE_API_KEY"] = api_key

# --- 2. 2026 "LITE" MODEL CONFIG ---
# We use 3.1 Flash-Lite because it allows 15 requests per minute (3x more than standard Flash).
gemini_llm = LLM(
    model="gemini/gemini-3.1-flash-lite-preview", 
    api_key=api_key,
    temperature=0.5
)

search_instance = DuckDuckGoSearchRun()

@tool("DuckDuckGoSearch")
def search_tool(search_query: str):
    """Search the internet for the latest 2026 financial data and news."""
    return search_instance.run(search_query)

# --- 3. AGENT DEFINITIONS ---
def build_crew(company_name):
    researcher = Agent(
        role="Lead Financial Researcher",
        goal=f"Gather the latest news and funding data for {company_name}.",
        backstory="Expert at navigating the web to find real-time financial signals.",
        tools=[search_tool],
        llm=gemini_llm,
        verbose=True,
        max_rpm=10 # Safety limit per agent
    )

    analyst = Agent(
        role="Senior Risk Strategist",
        goal=f"Create a high-level SWOT and risk rating for {company_name}.",
        backstory="Specializes in converting news into structured risk intelligence.",
        llm=gemini_llm,
        verbose=True,
        max_rpm=10 # Safety limit per agent
    )

    research_task = Task(
        description=f"Find the 5 most important recent updates for {company_name}.",
        expected_output="A bulleted list of 5 key developments.",
        agent=researcher
    )

    analysis_task = Task(
        description=f"Generate a SWOT analysis and Risk Rating (Low/Medium/High) for {company_name}.",
        expected_output="A markdown report with SWOT and a final risk assessment.",
        agent=analyst,
        context=[research_task]
    )

    # --- 4. THE CREW (WITH RATE LIMIT WORKAROUND) ---
    return Crew(
        agents=[researcher, analyst], 
        tasks=[research_task, analysis_task], 
        process=Process.sequential,
        # We limit the whole crew to 10 RPM to stay safe under the 15 RPM free limit
        max_rpm=10 
    )

# --- 5. UI ---
st.title("📊 Fin-Intel: 2026 Agentic Research")
st.info("Using Gemini 3.1 Flash-Lite for optimized free-tier performance.")

company = st.text_input("Enter Company Name:", placeholder="e.g. NVIDIA, Tesla")

if st.button("🚀 Run Analysis"):
    if not company:
        st.warning("Please enter a company name.")
    else:
        with st.spinner(f"Agents are coordinating... this takes about 45-60 seconds."):
            try:
                # Add a tiny delay before start to ensure quota window is fresh
                time.sleep(2) 
                
                crew = build_crew(company)
                result = crew.kickoff()
                
                st.markdown("---")
                st.subheader(f"Intelligence Report: {company}")
                st.markdown(str(result))
                st.success("✅ Analysis Complete")
            except Exception as e:
                if "429" in str(e):
                    st.error("Rate Limit Hit: The agents are working too fast for the free tier. Wait 30 seconds and try again.")
                else:
                    st.error(f"Error: {e}")

st.divider()
st.caption("2026 AI Agent Prototype · CrewAI + Gemini 3.1")
