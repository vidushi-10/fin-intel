Project Overview

Fin-Intel is a professional-grade autonomous research platform. It utilizes a
Sequential Process where specialized AI agents coordinate to search the live web
and synthesize financial intelligence into actionable risk reports.

Key Innovation: Successfully bypasses 2026 API limitations (429 Rate Limits)
by implementing intelligent max_rpm governors and the optimized Gemini
3.1 Flash-Lite model.

The Agentic Crew

Lead Financial Researcher: Scours web data for funding rounds, revenue
news, and market shifts.

Senior Risk Strategist: Transforms raw data into SWOT analyses and high-
level risk ratings.

2026 Technical Architecture

The system was built to handle strict infrastructure constraints while maintaining
high reasoning quality:
Model: gemini-3.1-flash-lite-preview (15 RPM Support)
•

•

•

Orchestration: CrewAI Framework
Search: DuckDuckGo Search API
Rate Control: 10 RPM Global Safety Cap

Installation & Setup

pip install streamlit crewai langchain-community \
google-genai duckduckgo-search
streamlit run app.py

Troubleshooting Legacy Errors

1. 404 NOT_FOUND

Fixed by migrating from legacy 1.5/2.0 strings to the April 2026 gemini-3.1-flash-
lite endpoint.

2. 429 RESOURCE_EXHAUSTED
Resolved using agent-level max_rpm=10 settings to stay within the Google Free Tier
quota window.
