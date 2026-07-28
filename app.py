import streamlit as st
import os
import time
import json
import re
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Fin-Intel Pro", page_icon="📊", layout="wide")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found. Add it to your .env file.")
    st.stop()

gemini_llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=api_key,
    temperature=0.5
)

search_instance = DuckDuckGoSearchRun()

@tool("DuckDuckGoSearch")
def search_tool(search_query: str):
    """Search the internet for the latest financial data, news, and market intelligence."""
    return search_instance.run(search_query)


def build_crew(company_name):
    researcher = Agent(
        role="Lead Financial Intelligence Researcher",
        goal=(
            f"Conduct a 3-pillar investigation into {company_name}: "
            "1. Quantitative (Revenue/Funding/Valuation), "
            "2. Qualitative (Leadership/Brand/Partnerships), "
            "3. Contextual (Market trends/Regulatory environment)."
        ),
        backstory="Expert at parsing news, filings, and market data to surface hidden financial signals.",
        tools=[search_tool],
        llm=gemini_llm,
        verbose=True,
        max_rpm=10
    )

    analyst = Agent(
        role="Senior Risk Strategist",
        goal=f"Synthesize the 3-pillar research into a SWOT analysis and Risk Rating for {company_name}.",
        backstory="Specializes in translating raw intelligence into structured risk ratings and investment verdicts.",
        llm=gemini_llm,
        verbose=True,
        max_rpm=10
    )

    # --- NEW: Critic agent ---
    critic = Agent(
        role="Report Quality Critic",
        goal=(
            "Evaluate the financial intelligence report for completeness and reasoning quality. "
            "Check that all 4 SWOT quadrants are present, the risk rating is justified by evidence, "
            "and flag any unsupported claims or missing data."
        ),
        backstory="A rigorous editor who ensures every report meets institutional-grade standards before publication.",
        llm=gemini_llm,
        verbose=True,
        max_rpm=10
    )

    research_task = Task(
        description=(
            f"Investigate {company_name} across three lenses:\n"
            "1. QUANTITATIVE: Revenue, valuation, funding rounds, burn rate.\n"
            "2. QUALITATIVE: Leadership quality, brand reputation, key partnerships.\n"
            "3. CONTEXTUAL: 2025-2026 market trends or regulatory issues affecting them."
        ),
        expected_output="A structured research brief with clearly labelled Quantitative, Qualitative, and Contextual sections.",
        agent=researcher
    )

    analysis_task = Task(
        description=(
            "Using the research brief, produce a Financial Intelligence Report with:\n"
            "- A SWOT matrix (all 4 quadrants: Strengths, Weaknesses, Opportunities, Threats)\n"
            "- A Risk Rating: LOW / MEDIUM-LOW / MEDIUM-HIGH / HIGH\n"
            "- A Strategic Recommendation (2-3 sentences)\n"
            "Each point must be tied to a specific finding from the research."
        ),
        expected_output="A markdown report with SWOT matrix, Risk Rating, and Strategic Recommendation.",
        agent=analyst,
        context=[research_task]
    )

    # --- NEW: Critic task ---
    critic_task = Task(
        description=(
            "Review the financial intelligence report and return a structured quality evaluation:\n"
            "1. COMPLETENESS: Are all 4 SWOT quadrants present? Is the risk rating stated?\n"
            "2. EVIDENCE QUALITY: Are claims backed by specific data from the research?\n"
            "3. GAPS: List any missing information or unsupported assertions.\n"
            "4. OVERALL SCORE: Rate the report 1-10 with a one-sentence justification.\n"
            "Append your evaluation at the end of the report under '## Critic Review'."
        ),
        expected_output="The original report with a '## Critic Review' section appended containing scores and gap analysis.",
        agent=critic,
        context=[analysis_task]
    )

    return Crew(
        agents=[researcher, analyst, critic],
        tasks=[research_task, analysis_task, critic_task],
        process=Process.sequential,
        max_rpm=10
    )


def eval_report(report_text: str) -> dict:
    """Simple rule-based eval: checks structural completeness of the report."""
    checks = {
        "has_strengths": bool(re.search(r"strength", report_text, re.IGNORECASE)),
        "has_weaknesses": bool(re.search(r"weakness", report_text, re.IGNORECASE)),
        "has_opportunities": bool(re.search(r"opportunit", report_text, re.IGNORECASE)),
        "has_threats": bool(re.search(r"threat", report_text, re.IGNORECASE)),
        "has_risk_rating": bool(re.search(r"risk rating", report_text, re.IGNORECASE)),
        "has_recommendation": bool(re.search(r"recommend", report_text, re.IGNORECASE)),
        "has_critic_review": bool(re.search(r"critic review", report_text, re.IGNORECASE)),
    }
    score = sum(checks.values())
    return {"checks": checks, "score": score, "max": len(checks)}


# --- UI ---
st.title("📊 Fin-Intel: Agentic Financial Intelligence")
st.caption("3-agent pipeline: Researcher → Analyst → Critic  ·  CrewAI + Gemini 1.5 Flash")

company = st.text_input("Enter Company Name:", placeholder="e.g. NVIDIA, Zithara.ai, Eka.care")

if st.button("Run Analysis", type="primary"):
    if not company.strip():
        st.warning("Please enter a company name.")
    else:
        with st.spinner(f"3 agents coordinating on {company}... (~60-90 seconds)"):
            try:
                time.sleep(2)
                crew = build_crew(company)
                result = crew.kickoff()
                report_text = str(result)

                st.markdown("---")
                st.subheader(f"Intelligence Report: {company}")
                st.markdown(report_text)

                # --- NEW: Eval panel ---
                st.markdown("---")
                st.subheader("Report Quality Eval")
                eval_result = eval_report(report_text)
                checks = eval_result["checks"]

                col1, col2 = st.columns(2)
                with col1:
                    for key, passed in checks.items():
                        label = key.replace("has_", "").replace("_", " ").title()
                        icon = "✅" if passed else "❌"
                        st.write(f"{icon} {label}")
                with col2:
                    score = eval_result["score"]
                    total = eval_result["max"]
                    st.metric("Completeness Score", f"{score}/{total}")
                    if score == total:
                        st.success("Report meets all structural requirements.")
                    elif score >= total - 2:
                        st.warning("Report is mostly complete. Minor gaps found.")
                    else:
                        st.error("Report has significant structural gaps.")

                st.success("Analysis complete.")

            except Exception as e:
                if "429" in str(e):
                    st.error("Rate limit hit. Wait 30 seconds and retry.")
                else:
                    st.error(f"Error: {e}")

st.divider()
st.caption("Fin-Intel  ·  CrewAI + Gemini 1.5 Flash  ·  3-agent sequential pipeline")
