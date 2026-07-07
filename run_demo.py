# Temporary demo run script
import os
import sys
from dotenv import load_dotenv

# Load env variables including the new GEMINI_API_KEY
load_dotenv()

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.judge_agent import judge_agent

def main():
    ticker = "NVDA"
    end_date = "2025-10-31"  # A recent target date
    print(f"=== Running Agent Committee on {ticker} for date {end_date} ===")
    print(f"Default Gemini Model: {os.environ.get('GEMINI_MODEL', 'gemini-3.5-flash')}")
    print("Fetching data, executing normalizer, running 4 parallel agents, enriching sentiment, and generating final verdict...")
    
    try:
        report = judge_agent(ticker, end_date, parallel=True)
        print("\n=== RUN COMPLETED SUCCESSFULLY ===")
        print(f"Verdict: {report.verdict}")
        print(f"Confidence: {report.confidence}%")
        print(f"Position Size: {report.position_size}")
        print(f"\nStrengths:\n" + "\n".join(f"- {s}" for s in report.strengths))
        print(f"\nRisks:\n" + "\n".join(f"- {r}" for r in report.risks))
        print(f"\nFinal Reasoning:\n{report.reasoning}")
    except Exception as e:
        print(f"\nRun failed: {e}")

if __name__ == "__main__":
    main()
