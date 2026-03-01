**RULE ONE**
Problem Title
Give your data science problem a clear, descriptive title.

Goal: Create Challenging Tasks
Your task should push the limit on an AI agent's ability to:

Explore and understand unfamiliar data
Apply domain-specific business logic
Handle edge cases and exceptions correctly
Produce accurate, well-formatted outputs
Not Just Simple Operations
Each data point should require genuine data science skills - not just simple aggregations or lookups that can be solved with basic operations.

Problem Title *

example "Multi Indicator Trading Signal Analysis"

Choose a clear, descriptive title that hints at the complexity involved.

**RULE TWO**
Data Upload
Upload only the task input files the agent should use to solve this problem.

File Requirements
Supported formats: .csv, .txt, .xlsx, .json, .parquet
Maximum 25 files allowed
Each file max 10.0 MB, total max 100.0 MB
Include all task inputs the solver will need to complete the problem
Task Inputs Only
Do upload: files the agent should read as input to solve the task
DO NOT upload oracle_solution.py, verifier.py, or any ground truth / answer files
Data Quality Guidelines
Data Sources
1.
Public/Real Data - Actual datasets (anonymized if needed)
2.
Generated Data - Synthetic data with fixed random seed
If Data is Generated
• Use realistic values and distributions
• Fetch external data (FX rates, etc.) from real APIs
• Include edge cases at realistic rates
Drag and drop your task input files here
or
Browse Files
Accepts task input files only: .csv, .txt, .xlsx, .json, .parquet

**RULE THREE**
Problem Prompt
Write the prompt that describes what solvers need to accomplish with the data you uploaded.

The Prompt Must Be
Clear
Unambiguous instructions that don't require guessing

Fair
All requirements discoverable from data or stated

Complete
Specify output format, file names, columns

Aligned
What you ask for must match the golden output

DO
Specify exact output file names
List required columns/keys
Provide formulas for non-standard calculations
Define domain terms explicitly
Specify thresholds if not obvious
DON'T
Expect column names not mentioned
Use formulas the model must "discover"
Ask to "reverse engineer" patterns
Assume deep domain knowledge
Have hidden expectations
Prompt Examples
IMPORTANT

We are looking for prompts that sound natural and human-written, not rigid templates.

Good Prompt
I need to prepare our 2024 claims performance report for the quarterly review. Can you analyze the claims data and help me understand our loss experience? I want to know...

Read more
Bad Prompt 1
Why this is bad: Ambiguous.

Can you do the Q1 2025 energy audit in the usual way? I dropped in the Q3 file and a few supporting files somewhere in the folder. Use whatever you...

Read more
Bad Prompt 2
Why this is bad: Robotic and unnatural.

Execute Q1 2025 energy audit procedure. Reference artifact: q3_2024_audit_report.xlsx. Use this artifact to replicate prior-state reporting format. Perform same analysis pipeline as previous cycle. Validate bill values against meter values...

Read more
Your Problem Prompt *

Be specific about input files and expected output
0 words (0 / 10,000 characters)
Your Uploaded Data Files
Reference these files in your prompt:

**RULE FOUR**
Solution & Test Set
Provide your reference solution, verifier tests, and ground truth answer.

Workspace Layout & Execution
Both scripts run inside a workspace with this structure:

workspace/
├── task/               ← data files + solver output
│   ├── prompt.txt
│   ├── data.csv        ← your uploaded data files
│   └── ...
└── ground_truth/       ← cwd for both scripts
    ├── oracle_solution.py
    ├── verifier.py
    └── golden_answer.csv
Both scripts are executed from ground_truth/ as the working directory. Use Path(__file__).parent to reference sibling files in ground_truth/, and Path(__file__).parent.parent / "task" to access data files and solver output in task/.

Supported Output Formats
Solutions can output one or more files in any of these formats:

.csv
CSV
Comma-separated values for tabular data
.xlsx
Excel
Excel spreadsheets for complex tabular outputs
.json
JSON
Structured data for nested or hierarchical results
.txt
Text
Plain text for simple outputs
Your solution can produce multiple output files of different types. Specify the expected output files clearly in your problem prompt.

Python 3.12 Environment
Scripts run on Python 3.12 (Debian) with these packages pre-installed. No need for a requirements.txt.

pandas
(Data manipulation)
numpy
(Numerical computation)
scipy
(Scientific computing)
scikit-learn
(Machine learning)
networkx
(Graph analysis)
openpyxl
(.xlsx read/write)
pyxlsb
(.xlsb read)
pyxlsb2
(.xlsb read (alt))
fastparquet
(Parquet read/write)
pyarrow
(Parquet & Arrow support)
1
Solution (oracle_solution.py)
Your reference solution that produces the correct output. This verifies the problem is solvable.


Example oracle_solution.py

import pandas as pd
import numpy as np

def main():
    # All files are in the same directory - use simple relative paths
    transactions = pd.read_csv("transactions.csv")
    adjustments = pd.read_csv("adjustments.csv")

    # Calculate gross revenue per business unit
    gross_revenue = transactions.groupby("business_unit")["amount"].sum().reset_index()
    gross_revenue.columns = ["business_unit", "gross_revenue"]

    # Calculate total adjustments per business unit
    total_adjustments = adjustments.groupby("business_unit")["amount"].sum().reset_index()
    total_adjustments.columns = ["business_unit", "adjustments"]

    # Merge and calculate net revenue
    result = gross_revenue.merge(total_adjustments, on="business_unit", how="left")
    result["adjustments"] = result["adjustments"].fillna(0)
    result["net_revenue"] = result["gross_revenue"] + result["adjustments"]

    # Calculate discrepancy flag (adjustment > 5% of gross)
    result["discrepancy_flag"] = np.where(
        np.abs(result["adjustments"] / result["gross_revenue"]) > 0.05,
        "YES", "NO"
    )

    # Save results
    result.to_csv("revenue_summary.csv", index=False)
    print("Results saved to revenue_summary.csv")

if __name__ == "__main__":
    main()

oracle_solution.py *

2
Verifier (verifier.py)
The verifier compares solver output against your ground truth. It must follow these principles:

1
Deterministic
Same input always produces same score

2
Fair
Only checks what the prompt specifies

3
Appropriate Tolerances
Account for rounding, format variations

4
Layered Checks
File → Schema → Counts → Values

5
Output Format
Last stdout line must end with X/Y (e.g. Score: 48/48), or use unittest

6
Exit Codes
Exit 0 for perfect score, exit 1 otherwise

What NOT to Verify
• Column ordering (unless explicitly required)
• Exact whitespace or formatting
• Case sensitivity (unless meaningful)
• Intermediate results not requested

Example verifier.py

import pandas as pd
import sys
from pathlib import Path


def load_golden():
    """Load golden answer from ground_truth/."""
    golden_path = Path(__file__).parent / "golden_revenue_summary.csv"
    return pd.read_csv(golden_path)


def load_submission(task_dir):
    """Load solver output from task/."""
    submission_path = task_dir / "revenue_summary.csv"
    if not submission_path.exists():
        return None
    return pd.read_csv(submission_path)


def check_value(expected, actual, tol_pct=1.0, tol_abs=0.01):
    """Compare two values with percentage and absolute tolerance."""
    if pd.isna(expected) and pd.isna(actual):
        return True
    if pd.isna(expected) or pd.isna(actual):
        return False
    try:
        expected, actual = float(expected), float(actual)
    except (ValueError, TypeError):
        return str(expected).strip().lower() == str(actual).strip().lower()
    if expected == 0:
        return abs(actual) <= tol_abs
    return abs(actual - expected) / abs(expected) * 100 <= tol_pct or abs(actual - expected) <= tol_abs


def verify(task_dir):
    golden = load_golden()
    submission = load_submission(task_dir)
    max_score = len(golden)

    if submission is None:
        print("No submission found (revenue_summary.csv)")
        print(f"0/{max_score}")
        sys.exit(1)

    score = 0
    required_cols = ["business_unit", "gross_revenue", "adjustments",
                     "net_revenue", "discrepancy_flag"]

    # Check required columns
    missing = [c for c in required_cols if c not in submission.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print(f"0/{max_score}")
        sys.exit(1)

    # Check row count
    if len(submission) != len(golden):
        print(f"Row count mismatch: expected {len(golden)}, got {len(submission)}")

    # Compare values per row
    merged = golden.merge(submission, on="business_unit", suffixes=("_expected", "_actual"))
    for _, row in merged.iterrows():
        bu = row["business_unit"]
        for col in ["gross_revenue", "net_revenue"]:
            if check_value(row[f"{col}_expected"], row[f"{col}_actual"]):
                score += 1
                print(f"  PASS  {bu} / {col}")
            else:
                print(f"  FAIL  {bu} / {col}: expected {row[f'{col}_expected']}, got {row[f'{col}_actual']}")

        if str(row["discrepancy_flag_expected"]).strip() == str(row["discrepancy_flag_actual"]).strip():
            score += 1
            print(f"  PASS  {bu} / discrepancy_flag")
        else:
            print(f"  FAIL  {bu} / discrepancy_flag: expected {row['discrepancy_flag_expected']}, got {row['discrepancy_flag_actual']}")

    # ── Required output format ─────────────────────────────────────────────
    # Last stdout line must END with "X/Y" (e.g. "15/15" or "Score: 15/15").
    # Exit code MUST be 0 for a perfect score, 1 otherwise.
    print(f"{score}/{max_score}")
    sys.exit(0 if score == max_score else 1)


if __name__ == "__main__":
    task_dir = Path(__file__).parent.parent / "task"
    verify(task_dir)

verifier.py *

3
Ground Truth Files
The expected correct output files. Your verifier will compare solver outputs against these.

Accepted formats: csv, xlsx, txt, json, parquet

Upload Ground Truth Files *
📁
Drag and drop your ground truth files here
or
Browse Files
Accepts .csv, .xlsx, .txt, .json, .parquet
Uploaded Files

**RULE FIVE**
Review & Submit
Review your problem and ensure all checks pass before submitting.

Multi Indicator Trading Signal Analysis
Data Files:
1 file(s)
Ground Truth:
1 file(s)
Solution:
Provided
Verifier:
Provided
Verifier Pre-Check Passed


Verifier passed (130/130).
Checks
(12 passed)
Ready to submit

Problem Title
2/2
Title Provided
Title provided (39 characters)
Valid Characters
Title uses valid characters

Data Upload
3/3
Files Uploaded
1 data file(s) uploaded
Valid File Formats
All files have valid formats
Within Size Limits
All files within size limits

Problem Prompt
1/1
Prompt Provided
Prompt provided (2,485 characters)

Solution & Verification
6/6
Solution Script (oracle_solution.py)
Solution script provided (10,351 characters)
Verifier Script (verifier.py)
Verifier script provided (4,980 characters)
Ground Truth Files
1 file(s) uploaded • File types valid • Upload complete • Files are non-empty • Total size: 1006 bytes
No Overlap Between Task and Ground Truth
No overlap between task assets and ground truth
Originality Check
No similar problems found
Verifier Pre-Check (oracle + verify)
Verifier passed (130/130).

**RULE SIX**
What an easy challenge after evaluation will look like 

Evaluation Pipeline
1/10
POOR
100% weighted pass rate across 10 runs
Difficulty (-3/3 pts) — Way too easy
100% pass rate — problem is too easy.
Failure Quality (0/3 pts) — No failures
All runs passed, so failure quality can't be assessed.
Prompt Conciseness (+2/2 pts) — Very concise
Prompt is 1,650 chars — under the 4,000 char ideal.
Fairness (+2/2 pts) — Fair
No unfair conditions detected.
Verdict Distribution
PASS CORRECT
10/10
100%
Score Distribution (out of 12)
10-14
10
No unfair conditions detected — all 10 judges assessed the evaluation as fair.

Individual Run Details (10 runs)
Download All (10)

#1
PASS CORRECT
12/12
high

#2
PASS CORRECT
12/12
high

#3
PASS CORRECT
12/12
high

#4
PASS CORRECT
12/12
high

#5
PASS CORRECT
12/12
high

#6
PASS CORRECT
12/12
high

#7
PASS CORRECT
12/12
high

#8
PASS CORRECT
12/12
high

#9
PASS CORRECT
12/12
high

#10
PASS CORRECT
12/12
high

What a problematic evaluation after submission will look like 

Evaluation Pipeline
Unfair evaluation condition
Score invalidated — the agent failed due to unfair evaluation conditions, not its own shortcomings. See fairness assessment below for details.
Verdict Distribution
FAIL WRONG CALCULATION
5/10
50%
FAIL UNFAIR EVAL
4/10
40%
FAIL NO OUTPUT
1/10
10%
Score Distribution (out of 130)
0-9
1
10-19
1
20-29
4
120-129
4
Failure Breakdown
FAIL_WRONG_CALCULATION
5
FAIL_UNFAIR_EVAL
4
FAIL_NO_OUTPUT
1

Unfair Evaluation Detected (4 judges flagged)

Individual Run Details (10 runs)
Download All (10)

#1
FAIL NO OUTPUT
0/130
high
Download output.json
Explanation
The AI agent produced no output whatsoever. The workspace diff is empty, no signal_analysis.csv was created, and no Python scripts or intermediate files exist in the after/ directory. The verifier confirmed a score of 0/130.

Root Cause
The agent failed to execute any analysis or produce the required signal_analysis.csv output file.

Cheating Check
No cheating detected — The workspace.diff is empty (0 bytes) and the after/ directory is identical to the before/ directory, containing only the original ohlcv_data.csv and prompt.txt. No code files, scripts, logs, output files, or execution traces exist in after/. The agent appears to have not executed any work at all. There is no evidence of accessing ground_truth/, oracle_solution.py, verifier.py, web searches, or any other unauthorized means.

Oracle Integrity Check
Oracle reads input data — The oracle_solution.py is a legitimate data-processing script. It reads the input file via pd.read_csv(task_dir / 'ohlcv_data.csv') at line 176, extracts OHLCV columns, and computes technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA50) from the data. Buy/sell signal scores are derived from these computed indicators through conditional logic on each data row. All output values depend on the actual input data — changing the CSV would produce different results. No hardcoded output values are present.


#2
FAIL WRONG CALCULATION
20/130
high
POSSIBLY UNFAIR
Download output.json
Explanation
The agent produced a signal_analysis.csv with 11 rows instead of the expected 13, scoring only 20/130. Multiple compounding errors caused widespread mismatches: (1) RSI seeding used rolling(14).mean() instead of averaging first 13 changes, producing different RSI values throughout; (2) RSI 2-point threshold was implemented as both bars < 25 instead of current < 25 and previous < 35, changing composite scores; (3) anti-overtrading used a gap of 7 bars instead of 6, causing 2 missed signals; (4) warmup started at index 50 instead of 49, shifting the first signal date; (5) MACD used pandas ewm(adjust=False) instead of manual EMA with SMA seed, causing histogram value differences.

Root Cause
Multiple implementation errors: wrong RSI seeding formula, wrong RSI threshold interpretation for 2-point scores, off-by-one in anti-overtrading gap (7 vs 6 bars), off-by-one in warmup period (50 vs 49), and different MACD EMA calculation method. These errors compounded to produce wrong dates, wrong scores, wrong indicator values, and 2 missing signals.

Cheating Check
No cheating detected — The agent solved the problem legitimately. The workspace.diff shows only two files were created: analyze.py (which reads ohlcv_data.csv and computes technical indicators using pandas/numpy/math) and signal_analysis.csv. There is no evidence of accessing ground_truth/, oracle_solution.py, verifier.py, or any external resources. The agent's output (11 signals) differs from the golden output (13 signals), consistent with an independent implementation with minor methodology differences (e.g., pandas ewm for MACD vs manual EMA, different RSI initialization seeding, warmup period of 50 vs 49, and anti-overtrading gap of 7 vs 6 bars).

Oracle Integrity Check
Oracle reads input data — The oracle solution reads ohlcv_data.csv (line 176) and computes all indicators from the data: Wilder RSI (lines 29-57), MACD with manual EMA (lines 60-97), Bollinger Bands with population std dev (lines 100-120), Wilder ATR (lines 123-141), and SMA50 (line 187). The composite scoring, signal generation, and position sizing are all derived from these computed indicator values. All output values depend on the input data, and changing the CSV would produce different results.

Fairness Check
The judge found ambiguity or issues in the evaluation that may have caused the agent to fail. However, the agent also made mistakes that were independent of that unfairness. Resolving the eval issues would make the result clearer.

Unfair conditions identified:
RSI 2-point threshold interpretation is ambiguous: 'RSI below 35 on both current and previous bar (2 points if below 25 instead)' - the agent's reading (both bars < 25) is the more natural grammatical interpretation, while the oracle uses an asymmetric reading (current < 25, previous < 35) that is not clearly stated in the prompt.
Anti-overtrading gap is ambiguous: 'minimum 6 bars between signals' can reasonably mean either a distance of 6 (oracle: gap > 5) or 6 intervening bars (agent: gap >= 7). The agent's literal reading of 'between' is defensible.
MACD EMA initialization method is ambiguous: 'standard MACD (12/26/9)' does not specify whether to use SMA-seeded EMA or first-value-seeded EMA. The agent's use of pandas ewm(adjust=False) is a widely-used legitimate implementation.
Oracle's MACD crossover uses <= 0 for lookback (zero counts as negative), while the prompt says 'after being negative within last 3 bars' - zero is not negative, making the agent's strict < 0 check arguably more correct.

Independent Agent Mistakes (2)
unrelated to unfairness
These mistakes were made by the agent on its own, separate from any evaluation issues. Fixing the eval ambiguities would strengthen the result.

RSI seeding bug: The agent's RSI implementation uses gain.rolling(window=14).mean() which includes a spurious zero at index 0 (from series.diff() producing NaN, which .where() converts to 0). This makes the seed (13 real gains + 1 artificial zero) / 14 instead of the correct 14 real gains / 14. The oracle's RSI is mathematically equivalent to the standard Wilder RSI. This is a coding bug, not an interpretation issue.
Warmup off-by-one: The agent uses 'if i < 50: continue' (starts at index 50), but SMA50 first becomes valid at index 49. The oracle correctly starts at index 49 (min_warmup = 49). This is a straightforward off-by-one error with no ambiguity in the requirements.

#3
FAIL UNFAIR EVAL
26/130
high
UNFAIR
Download output.json
Explanation
The agent's solution has multiple calculation errors: (1) MACD uses pandas ewm(adjust=False) seeded from bar 0 instead of the correct SMA-seeded EMA approach, producing different histogram values throughout; (2) RSI scoring thresholds are wrong — for the 2-point condition, the agent requires the previous bar RSI to also cross the extreme threshold (25/75) when the oracle only requires the sustained threshold (35/65); (3) ATR seed window is off by one bar (skips first TR value); (4) Anti-overtrading cooldown requires 7+ bars instead of 6+ bars. These compounding errors result in only 11 of 13 expected signals with many value mismatches.

Root Cause
Multiple calculation errors: incorrect MACD EMA seeding method (pandas ewm vs SMA-seeded), wrong RSI scoring thresholds for the 2-point condition (prev_rsi threshold too strict), ATR off-by-one in seed window, and anti-overtrading cooldown 1 bar too restrictive.

Cheating Check
No cheating detected — The agent implemented the trading signal analysis independently without accessing ground_truth/, oracle_solution.py, or verifier.py. The workspace.diff shows only two files created (analyze.py and signal_analysis.csv). The agent's code differs from the oracle in multiple fundamental ways: (1) uses pandas ewm() for EMA vs oracle's manual loop, (2) RSI 2-point check requires both bars below 25 vs oracle's current<25 and previous<35, (3) anti-overtrading uses >=7 bar gap vs oracle's <=5, (4) tie-breaking favors SELL vs oracle favoring BUY. The agent produced 11 signals vs oracle's 14, with materially different dates and scores. No web searches or external API calls were detected.

Oracle Integrity Check
Oracle reads input data — The oracle reads ohlcv_data.csv at line 176 using pd.read_csv(), extracts close/high/low/volume columns, and computes all technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA50) through detailed algorithmic functions that operate on the data. The output signals are derived entirely from these data-dependent calculations. Changing the input data would produce different results.

Fairness Check
Evaluation flagged as unfair — the agent failed due to issues with the evaluation, not its own mistakes. This indicates the eval needs to be fixed.

Unfair conditions identified:
RSI 2-point scoring threshold is ambiguous: The prompt says '1 point if RSI below 35 on both current and previous bar (2 points if below 25 instead)'. The word 'instead' most naturally reads as substituting 25 for 35 in the entire condition (both bars must be below 25), which is the agent's interpretation. The oracle instead requires only the current bar below 25 while the previous bar only needs to be below 35. The agent's reading is arguably MORE natural than the oracle's.
MACD EMA seeding method is ambiguous: The prompt says 'standard MACD (12/26/9)' without specifying the EMA initialization method. The oracle uses SMA-seeded EMA (SMA of first N values as seed), while the agent uses pandas ewm(adjust=False) which seeds from bar 0. Both are widely used 'standard' MACD implementations. This difference propagates through all MACD histogram values.
Anti-overtrading cooldown has classic fencepost ambiguity: The prompt says 'minimum 6 bars between signals'. The oracle interprets this as bar_gap >= 6 (allowing signal at bar+6), while the agent interprets it as 6 bars strictly between signals requiring bar_gap >= 7. 'N bars between' is a well-known fencepost ambiguity.
ATR seed window is underspecified: The prompt says '14-period Wilder ATR' without specifying how to handle TR[0] (which has no previous close). The oracle defines TR[0] = high-low and includes it in the seed; the agent treats TR[0] as undefined (NaN) and seeds from TR[1:15]. Both are defensible interpretations.

#4
FAIL WRONG CALCULATION
124/130
high
Download output.json
Explanation
The agent scored 124/130 with 6 failed checks across RSI (2 mismatches) and MACD histogram (4 mismatches). The RSI error stems from using pandas rolling(window=14).mean() to seed Wilder's smoothing (averaging 14 values), whereas the oracle seeds with the first 13 changes divided by 13. The MACD error comes from using pandas ewm(span=, adjust=False) which initializes EMA from the first data point, while the oracle seeds the EMA with an SMA of the first N values. Both are subtle initialization differences in well-known technical indicator calculations.

Root Cause
RSI and MACD calculations used pandas built-in ewm/rolling functions with different initialization semantics than the expected manual Wilder smoothing and SMA-seeded EMA implementations.

Cheating Check
No cheating detected — The agent's code (analyze_signals.py) only reads ohlcv_data.csv and implements all technical indicators independently. No access to ground_truth/, oracle_solution.py, verifier.py, or golden files was detected. No web searches, HTTP requests, or external API calls were made. The solution is a legitimate independent implementation of the trading signal analysis.

Oracle Integrity Check
Oracle reads input data — The oracle solution reads ohlcv_data.csv (line 176), extracts price/volume data (lines 178-181), computes all technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA50) from the data (lines 183-187), and derives signal scores through data-dependent computations (lines 199-298). All output values are computed from the input data. Changing the input CSV would change the results.


#5
FAIL UNFAIR EVAL
124/130
high
UNFAIR
Download output.json
Explanation
The agent scored 124/130 (95.4%). The 6 mismatches are in RSI (2) and MACD histogram (4) values for the earliest signals. The root cause is different initialization seeding for both Wilder RSI and MACD EMA calculations. The oracle seeds RSI's initial average with (period-1) values divided by (period-1), while the agent used pandas rolling(period).mean() which seeds with period values divided by period. For MACD, the oracle seeds EMA with an SMA of the first `period` values, while the agent used pandas ewm(adjust=False) which initializes from the first data point. These initialization differences cause small numerical deviations that diminish over time but are still outside tolerance for early signals.

Root Cause
Different initialization seeding for Wilder RSI (rolling(14).mean vs sum of first 13 changes / 13) and MACD EMA (pandas ewm from first value vs SMA-seeded EMA), causing small numerical deviations in early signal rows.

Cheating Check
No cheating detected — The agent created analyze_signals.py which reads ohlcv_data.csv and implements all technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA50) from scratch using pandas/numpy. The workspace.diff shows only two new files created (analyze_signals.py and signal_analysis.csv) with no references to ground_truth/, oracle_solution.py, verifier.py, or any web requests. No evidence of answer leakage, web lookup, or prompt injection.

Oracle Integrity Check
Oracle reads input data — The oracle solution reads ohlcv_data.csv (line 176: pd.read_csv(task_dir / 'ohlcv_data.csv')), extracts close/high/low/volume columns, and computes all technical indicators (RSI via Wilder's method, MACD, Bollinger Bands with population std, ATR, SMA50) using custom functions that operate on the data. The composite scoring, signal generation, and position sizing are all derived from these computed indicators. Changing the input data would produce different output.

Fairness Check
Evaluation flagged as unfair — the agent failed due to issues with the evaluation, not its own mistakes. This indicates the eval needs to be fixed.

Unfair conditions identified:
The oracle's RSI implementation uses a non-standard initialization: it seeds with sum(gains[:period-1]) / (period-1), i.e., 13 values divided by 13, rather than the canonical Wilder RSI initialization of period (14) values divided by 14. The prompt says '14-period Wilder RSI' which, per Wilder's original book, seeds with the first 14 changes. The agent's rolling(14).mean() approach is actually more faithful to the standard definition.
The prompt says 'standard MACD (12/26/9)' without specifying EMA initialization/seeding method. There are multiple legitimate implementations: SMA-seeded EMA (oracle's approach) vs first-value-seeded EMA (agent's approach using pandas ewm(adjust=False)). Both are widely used in industry and neither is more 'standard' than the other.
The MACD histogram tolerance of 0.0001 absolute is unreasonably tight for an indicator sensitive to initialization methodology. MACD histogram values range in single digits, so initialization differences cascade through two EMA layers (fast/slow then signal), producing deviations of 0.001-0.12 on early rows that exceed this tolerance by orders of magnitude. Even the 0.001 relative tolerance is too tight for early-row values given the ambiguity in initialization.
The RSI tolerance of 0.01 absolute is tight enough that different but equally valid RSI initialization approaches produce failures on early signal rows.

#6
FAIL WRONG CALCULATION
13/130
high
POSSIBLY UNFAIR
Download output.json
Explanation
The agent's implementation has multiple calculation errors: (1) The Wilder RSI seeding uses indices 1..period (off by one) instead of the correct 0..period-2, producing different RSI values that shift which bars qualify as oversold/overbought. (2) The ATR similarly seeds at index=period instead of period-1. (3) The RSI scoring for the 2-point buy condition requires both current AND previous RSI < 25, but the oracle only requires current < 25 with previous < 35 (asymmetric check). These cascading errors caused the agent to identify 15 signals instead of the correct 13, with different dates and scores for most rows.

Root Cause
Off-by-one error in Wilder RSI and ATR initialization (seeding at index=period using gains[1:period+1] instead of at index=period-1 using gains[:period-1]), combined with overly strict RSI scoring condition requiring both bars below 25 instead of the correct asymmetric thresholds.

Cheating Check
No cheating detected — The workspace.diff shows only two files created: analyze_signals.py and signal_analysis.csv. The agent's script is a legitimate implementation that reads ohlcv_data.csv and computes technical indicators (Wilder RSI, MACD, Bollinger Bands, ATR, SMA50) to generate trading signals. There is no evidence of accessing ground_truth/, oracle_solution.py, verifier.py, or golden_signal_analysis.csv. No web searches, HTTP requests, or prompt injection attempts are present. The code structure and approach differ substantially from the oracle solution (pandas vectorized operations vs list-based iteration, different variable naming, different code organization).

Oracle Integrity Check
Oracle reads input data — The oracle solution reads ohlcv_data.csv (line 176: df = pd.read_csv(task_dir / 'ohlcv_data.csv')), extracts OHLCV columns (lines 178-181), and computes all technical indicators from the data: RSI (line 183), MACD (line 184), Bollinger Bands (line 185), ATR (line 186), SMA50 (line 187). It then iterates through the data (lines 199-298) computing buy/sell scores based on these computed indicators, with position sizing derived from ATR values. All output values are derived from genuine data transformations and aggregations. Changing the input OHLCV data would produce different trading signals and scores. No hardcoded output values are present.

Fairness Check
The judge found ambiguity or issues in the evaluation that may have caused the agent to fail. However, the agent also made mistakes that were independent of that unfairness. Resolving the eval issues would make the result clearer.

Unfair conditions identified:
The prompt specifies 'standard MACD (12/26/9)' without defining the EMA initialization method. The oracle uses SMA-seeded EMA (first N values averaged as seed) while the agent uses pandas ewm(adjust=False) which seeds from the first data point. Both are valid implementations of 'standard MACD'. The verifier's MACD histogram tolerance (0.0001 absolute, 0.001 relative) is too tight to accommodate this legitimate ambiguity, causing 2 out of 11 matching signal rows to fail MACD checks even when signal selection is otherwise correct.
The RSI scoring specification is ambiguous: '1 point if RSI below 35 on both current and previous bar (2 points if below 25 instead)'. The oracle interprets this as current < 25 AND previous < 35 (asymmetric), while the agent interprets it as both < 25 (symmetric). The agent's interpretation is arguably the more natural reading of the English text. However, this particular ambiguity does not materially affect the outcome since the agent's interpretation is stricter and would only lower scores, not inflate them.

Independent Agent Mistakes (2)
unrelated to unfairness
These mistakes were made by the agent on its own, separate from any evaluation issues. Fixing the eval ambiguities would strengthen the result.

Critical NaN handling bug in SMA50 trend filter: The agent uses `df.loc[df['close'] >= df['sma50'], 'buy_score'] = 0` to enforce the trend filter. When SMA50 is NaN (for the first 49 bars before 50-period SMA is available), the comparison `close >= NaN` evaluates to False in pandas, so the buy_score is NOT zeroed out. This means the agent generates BUY signals before SMA50 is computed (Feb 9 at index 39, Feb 15 at index 45), which are invalid. The prompt clearly states BUY requires close < SMA50, and signals should not be generated when the indicator is not yet available. The oracle correctly skips all bars before index 49.
Cascading signal date errors from the NaN bug: The two invalid early signals (Feb 9, Feb 15) consume the 6-bar anti-overtrading spacing windows, causing the agent's subsequent February signals (Feb 21, Feb 28) to land on different dates than the oracle's correct signals (Feb 19, Feb 25). This produces 15 total signals instead of the correct 13, with completely different February dates.

#7
FAIL WRONG CALCULATION
26/130
high
POSSIBLY UNFAIR
Download output.json
Explanation
The agent produced 11 signals instead of the expected 13, with widespread calculation errors across RSI, MACD histogram, and other indicators. The root causes include: (1) incorrect RSI Wilder smoothing initialization using a different window of price changes than the standard approach, (2) MACD computed via pandas ewm(adjust=False) which starts EMA from the first value rather than initializing with SMA for the first N periods, (3) RSI 2-point scoring incorrectly requires both bars < 25 instead of current < 25 and previous < 35, and (4) MACD crossover logic is too strict requiring all intermediate bars to have consistent signs. These errors cascade to produce wrong indicator values, wrong signal dates, and missing signals.

Root Cause
Multiple calculation errors: RSI initialization window off-by-one, MACD EMA startup method (pandas ewm vs SMA-initialized EMA), RSI scoring threshold misinterpretation (both < 25 vs current < 25 and previous < 35), and overly strict MACD histogram crossover logic requiring consecutive same-sign bars.

Cheating Check
No cheating detected — The agent solved the problem independently. It created analyze.py using pandas vectorized operations, which is structurally distinct from the oracle's pure-Python approach. No access to ground_truth/, oracle_solution.py, or verifier.py was found in workspace.diff or agent code. No web searches or external API calls were made. The agent's output (11 signals) differs from the golden output (13 signals) in dates, scores, and indicator values, consistent with legitimate implementation differences in RSI seeding, anti-overtrading interpretation, and threshold conditions.

Oracle Integrity Check
Oracle reads input data — The oracle reads ohlcv_data.csv (line 176), extracts OHLCV columns (lines 178-181), and computes all technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA50) from the data using manual calculations (lines 183-187). The signal generation loop (lines 199-298) uses these computed indicators to derive composite scores and generate trading signals. All output values are derived from data transformations and aggregations. Changing the input data would change the output.

Fairness Check
The judge found ambiguity or issues in the evaluation that may have caused the agent to fail. However, the agent also made mistakes that were independent of that unfairness. Resolving the eval issues would make the result clearer.

Unfair conditions identified:
The oracle's 'minimum 6 bars between signals' implementation uses `bars_since_signal <= 5` (allowing next signal at last+6, i.e., only 5 bars between signals), while the prompt literally says 'minimum 6 bars between signals'. The agent's interpretation of requiring 6 actual bars between signals (next signal at last+7) is the more natural and arguably correct reading of the prompt. This ambiguity is the PRIMARY cause of the agent's failure, cascading into wrong dates, shifted signals, and 2 missing signals.
The verifier uses an extremely tight tolerance of 0.0001 for MACD histogram values, which is insufficient to accommodate the well-known difference between SMA-initialized EMA and pandas ewm(adjust=False) EMA for 'standard MACD'. Both are widely recognized as valid MACD implementations. The prompt specifies 'standard MACD (12/26/9)' without clarifying the EMA initialization method.

Independent Agent Mistakes (1)
unrelated to unfairness
These mistakes were made by the agent on its own, separate from any evaluation issues. Fixing the eval ambiguities would strengthen the result.

The agent used pandas ewm(span=N, adjust=False) for MACD EMA calculation instead of the more standard SMA-initialized EMA. While both are defensible as 'standard MACD', the SMA-initialized approach is the textbook method (Gerald Appel). The ewm(adjust=False) approach seeds the EMA from the first data point, causing persistent differences in early bars that exceed the verifier's MACD histogram tolerance at 4 out of 13 signal dates (indices 49, 55, 94, 106). Even if the spacing issue were fixed (giving the agent all 13 correct dates), these 4 MACD histogram checks would still fail, causing the verifier to reject the submission.

#8
FAIL UNFAIR EVAL
126/130
high
UNFAIR
Download output.json
Explanation
The agent scored 126/130 (96.9%). All 13 signal rows were correctly identified with matching dates, signal types, composite scores, close prices, RSI, Bollinger Bands, and position sizes. However, 4 MACD histogram values exceeded the tolerance threshold. The root cause is the agent used pandas `ewm(span=N, adjust=False)` for MACD EMAs, which seeds from the first data point, while the oracle uses a manual EMA seeded with an SMA of the first N values. This EMA initialization difference propagated through the MACD line and histogram calculations.

Root Cause
MACD EMA initialization: agent used pandas ewm(adjust=False) starting from first value, while oracle seeds EMA with SMA of first period values. This caused MACD histogram deviations of up to 0.124 on early signals.

Cheating Check
No cheating detected — The agent created two Python scripts (analyze_signals.py and debug_check.py) that independently implement the trading signal analysis by reading only the legitimate input file ohlcv_data.csv. The workspace.diff shows no access to ground_truth/, oracle_solution.py, verifier.py, or golden_signal_analysis.csv. No web searches, HTTP requests, or external API calls were made. The agent's MACD histogram values differ from the oracle's (e.g., -2.592403 vs -2.467945), which is consistent with an independent implementation using pandas.ewm rather than the oracle's manual EMA loops. The agent solved the problem legitimately through its own code.

Oracle Integrity Check
Oracle reads input data — The oracle (ground_truth/oracle_solution.py) genuinely reads and processes the input data. It reads ohlcv_data.csv via pd.read_csv() at line 176, extracts price and volume columns, then performs ~300 lines of computational logic: calculating RSI with Wilder smoothing (lines 29-57), MACD with manual EMA (lines 60-97), Bollinger Bands with population std (lines 100-120), ATR (lines 123-141), SMA50 (lines 19-26), and composite scoring (lines 199-298). All output values are derived from these data-dependent computations. Changing the input data would change the output.

Fairness Check
Evaluation flagged as unfair — the agent failed due to issues with the evaluation, not its own mistakes. This indicates the eval needs to be fixed.

Unfair conditions identified:
The task prompt specifies 'standard MACD (12/26/9)' without defining how the EMA should be initialized/seeded. The term 'standard MACD' is ambiguous - there is no single universal standard for EMA initialization. Some implementations seed the EMA with an SMA of the first N periods (as the oracle does), while others start from the first data point (as pandas ewm(adjust=False) does, which the agent used). Both are widely used legitimate approaches. The oracle chose one specific interpretation and the verifier enforces it with extremely tight tolerances (0.0001 absolute / 0.001 relative for MACD histogram), making it impossible for the agent's equally valid interpretation to pass.
The RSI scoring logic in the oracle deviates from the prompt's literal text. The prompt says '2 points if below 25 instead' (meaning both current and previous bar below 25), but the oracle implements 'current < 25 AND previous < 35'. The agent correctly followed the literal prompt text. While this didn't directly cause the MACD failures, it demonstrates the oracle itself doesn't faithfully implement the prompt specification, further undermining its authority as ground truth.

#9
FAIL WRONG CALCULATION
20/130
high
POSSIBLY UNFAIR
Download output.json
Explanation
The agent made multiple calculation errors: (1) RSI initial seeding used period values instead of period-1 values, shifting all RSI values; (2) Anti-overtrading used a 7-bar minimum gap (>=7) instead of the correct 6-bar gap (>5), missing valid signals; (3) RSI scoring thresholds were wrong — the agent required both current and previous bars at the extreme threshold (both <25 for 2pts buy, both >75 for 2pts sell), while the correct logic checks current at extreme and previous at the sustained threshold (current <25 with previous <35 for buy, current >75 with previous >65 for sell). These compounding errors resulted in wrong signal dates, scores, and 2 missing signals (11 vs expected 13).

Root Cause
Multiple interacting errors: wrong RSI Wilder smoothing initialization (period vs period-1 seed), incorrect anti-overtrading gap (7 bars vs 6 bars), and wrong RSI scoring thresholds for the 2-point sustained condition (both bars at extreme vs current at extreme + previous at sustained level).

Cheating Check
No cheating detected — The agent did not cheat. The workspace.diff shows only two files created: analyze_signals.py and signal_analysis.csv. The agent's code only reads ohlcv_data.csv (the legitimate input). There are no imports of network libraries (requests, urllib, etc.), no subprocess calls, and no file access to ground_truth/, oracle_solution.py, or verifier.py. Furthermore, the agent made several implementation errors (wrong RSI initialization using SMA of period values instead of period-1, stricter RSI threshold conditions requiring both bars below 25/above 75 instead of the oracle's asymmetric thresholds, and an off-by-one error in anti-overtrading logic using >= 7 instead of > 5), producing 11 signals instead of the expected 13. These errors are strong evidence of independent work — a cheating agent would have matched the oracle output exactly.

Oracle Integrity Check
Oracle reads input data — The oracle solution legitimately reads and processes the input data. At line 176, it reads ohlcv_data.csv via pd.read_csv(). It extracts closes, highs, lows, and volumes from the DataFrame and computes all technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA50) from these data values using custom calculation functions. The signal scoring logic operates on these computed indicators, and the output is fully derived from the input data. Changing the input CSV would produce different indicator values and different signals.

Fairness Check
The judge found ambiguity or issues in the evaluation that may have caused the agent to fail. However, the agent also made mistakes that were independent of that unfairness. Resolving the eval issues would make the result clearer.

Unfair conditions identified:
RSI 2-point scoring threshold ambiguity: The prompt says '1 point if RSI below 35 on both current and previous bar (2 points if below 25 instead)' — the parenthetical 'below 25 instead' is ambiguous about whether it replaces the threshold for both bars or just the current bar. The agent interpreted it as both bars below 25, while the oracle uses current < 25 AND previous < 35 (asymmetric thresholds). Both readings are defensible.
RSI Wilder smoothing initialization is underspecified: The prompt only says '14-period Wilder RSI' without specifying the seeding method. The oracle uses a non-standard initialization averaging the first (period-1)=13 changes divided by (period-1), while standard implementations typically average the first period=14 changes. The agent used a rolling(period).mean() seed which is the more common approach but differs from the oracle.
MACD EMA calculation method ambiguity: The prompt says 'standard MACD (12/26/9)' but doesn't specify whether the EMA should be SMA-seeded (traditional) or use recursive exponential smoothing from bar 1 (pandas ewm adjust=False). The oracle uses SMA-seeded EMA while the agent uses ewm(adjust=False), producing different values.

Independent Agent Mistakes (1)
unrelated to unfairness
These mistakes were made by the agent on its own, separate from any evaluation issues. Fixing the eval ambiguities would strengthen the result.

Anti-overtrading gap error: The prompt clearly states 'minimum 6 bars between signals.' The agent implemented `if idx - last_signal_idx >= 7` which requires a minimum gap of 7 bars, not 6. The correct implementation should allow signals at exactly 6 bars distance (gap >= 6, as the oracle implements with `bars_since_signal <= 5: continue`). This is a straightforward off-by-one error independent of any ambiguity in the prompt.

#10
FAIL UNFAIR EVAL
126/130
high
UNFAIR
Download output.json
Explanation
The agent scored 126/130 with 4 macd_histogram value mismatches. The root cause is that the agent used pandas `ewm(span=..., adjust=False)` for MACD EMA calculations, which starts exponential smoothing from the first data point. The oracle solution uses a manual EMA that seeds with an SMA of the first N periods before applying exponential smoothing, producing slightly different MACD line and signal line values that propagate into histogram differences exceeding the 0.0001 tolerance.

Root Cause
MACD EMA initialization: pandas ewm(adjust=False) starts smoothing from first value, while the expected approach seeds EMA with SMA of first N periods. This causes MACD histogram values to differ by 0.001-0.12, exceeding the verifier's 0.0001 absolute tolerance.

Cheating Check
No cheating detected — The agent solved the problem legitimately. Its code (analyze_signals.py) only imports pandas and numpy, reads only the input file ohlcv_data.csv, and writes signal_analysis.csv. The workspace.diff confirms only these two files were created. There is no access to ground_truth/, oracle_solution.py, verifier.py, or any external resources. The agent's implementation uses its own approach (e.g., pandas ewm for EMA calculations) that differs from the oracle's manual EMA implementation, indicating independent work.

Oracle Integrity Check
Oracle reads input data — The oracle solution reads ohlcv_data.csv via pd.read_csv() at line 176 and extracts closes, highs, lows, and volumes from the dataframe. It then computes all technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA50) from this data through substantive calculations. The signal detection logic iterates through the data applying scoring rules, and the output is entirely derived from these data-dependent computations. Changing the input data would produce different results.

Fairness Check
Evaluation flagged as unfair — the agent failed due to issues with the evaluation, not its own mistakes. This indicates the eval needs to be fixed.

Unfair conditions identified:
The task prompt specifies 'standard MACD (12/26/9)' without defining how to initialize the EMA calculation. There are two widely-used approaches: (1) seeding the EMA with an SMA of the first N periods (used by the oracle), and (2) starting exponential smoothing from the first data point (pandas ewm with adjust=False, used by the agent). Both are legitimate, widely-documented implementations of 'standard MACD.' The oracle uses approach (1) and the verifier enforces a tight absolute tolerance of 0.0001 on macd_histogram, which does not accommodate the small numerical differences arising from approach (2). This is an ambiguous requirement combined with an overly strict tolerance that penalizes a valid alternative implementation.
The verifier requires a perfect score (score == total_checks) to pass, meaning even a single column mismatch in a single row causes complete failure. Combined with the tight 0.0001 absolute tolerance on macd_histogram and the ambiguous EMA initialization specification, this creates an unfair pass/fail threshold for a legitimately ambiguous calculation.

**RULE SEVEN**
**now this must be followed strictly**

To make a challenge fair enough the challenge must be tested locally, The prompt.txt must short, small and concise no non ascii character, symbols, computerized or special character

The prompt.txt must cover what and why and now how, the tests and challenge must cover strictly super complex observable aspect and not implementation aspect... when i mean observable aspect a user can use its own method and still solve it so we dont strictly need to give exact api, string or patterns that a user must follow the user can follow any pattern 

a good observable prompt.txt must start like this eg add a, etc it must be short and concise covering the challenge

**now for complexity** 

where applicable and this applies add these to the format

**Time & Space Complexity Requirements:**
- Target O(log n), O(n), or O(n log n) complexity - avoid O(n²) or worse unless absolutely necessary
- Optimize space complexity: prefer O(1) auxiliary space where possible, O(log n) for recursive solutions
- Use amortized analysis for data structures (e.g., dynamic arrays, union-find with path compression)

**Algorithmic Techniques (Apply Where Applicable):**
- **Divide & Conquer**: Break problems into subproblems, solve recursively, merge results efficiently
- **Dynamic Programming**: Use memoization or tabulation for overlapping subproblems with optimal substructure
- **Greedy Algorithms**: When local optimal choices lead to global optimum with provable correctness
- **Binary Search**: For sorted data or monotonic functions - reduce search space logarithmically
- **Two Pointers / Sliding Window**: For array/string traversal with O(n) instead of O(n²)
- **Prefix Sums / Difference Arrays**: For range queries and updates in O(1) after O(n) preprocessing
- **Bit Manipulation**: Use bitwise operations for space-efficient solutions and O(1) operations
- **Union-Find with Path Compression & Rank**: For disjoint set operations in near O(1) amortized
- **Segment Trees / Fenwick Trees**: For range queries with O(log n) updates
- **Monotonic Stack/Queue**: For next greater element, sliding window maximum in O(n)
- **Trie / Radix Trees**: For prefix-based string operations
- **Graph Algorithms**: BFS/DFS, Dijkstra's, topological sort, strongly connected components as needed

**Code Quality Standards:**
- Use the language eg Python generics and type inference to maximize type safety
- Implement proper error handling with descriptive error types
- Write pure functions where possible - avoid side effects
- Use immutable data patterns unless mutation is necessary for performance
- Leverage lazy evaluation and generators for memory-efficient iteration
- Apply early returns and guard clauses to reduce nesting

**Advanced Patterns:**
- **Functional Composition**: Chain operations using map, filter, reduce with optimal short-circuiting
- **Iterator Protocol**: Implement custom iterables for memory-efficient streaming
- **Proxy/Reflect**: For meta-programming solutions requiring interception
- **WeakMap/WeakSet**: For cache implementations without memory leaks
- **Structural Sharing**: For immutable updates without full copies
- **Tail Call Optimization**: Structure recursion for TCO where supported
- **Object Pooling**: Reuse objects to minimize GC pressure in hot paths

**Performance Optimizations:**
- Minimize allocations in hot paths - preallocate arrays when size is known
- Use TypedArrays for numeric computations
- Prefer `for` loops over `.forEach()` in performance-critical sections
- Cache computed values and array lengths in tight loops
- Use `Map`/`Set` over plain objects for frequent lookups (O(1) guaranteed)
- Avoid unnecessary spreading/destructuring in loops
- Consider branch prediction - put common cases first in conditionals

**Must NOT:**
- Use naive nested loops when better algorithms exist
- Implement brute force when polynomial/logarithmic solutions are achievable
- Use simple `.includes()` or `.indexOf()` repeatedly when a Set/Map lookup suffices
- Create unnecessary intermediate arrays when streaming/generators work
- Ignore edge cases that could cause performance degradation

**Solution Must Demonstrate:**
- Deep understanding of algorithmic paradigms
- Mastery of language-specific optimizations
- Production-grade error handling
- Code that would pass rigorous code review

**DIFFICULTY LEVEL**
The challenge should be a challenge LLMs like Claude opus 4.5 cannot solve at a go