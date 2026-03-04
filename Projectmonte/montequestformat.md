# Monte Quest Format Guide (Updated)

## Goal: Create Challenging Tasks

**CRITICAL: DUPLICATE DETECTION IS ACTIVE**

The submission system automatically detects challenges that are structurally similar to existing challenges. If flagged as a potential duplicate, your submission may be rejected. Each challenge must be genuinely unique in:

1. **Domain** - Different industry/context (not just renaming "stations" to "vendors")
2. **Computation Pattern** - Different algorithmic structure (not just changing coefficient values)
3. **Data Flow Architecture** - Different relationships between inputs and outputs

**PATTERNS THAT TRIGGER DUPLICATE FLAGS:**
- Reusing the same "aggregate → rolling stateful score → tiered classification → penalty ledger" flow
- Same number of inputs/outputs with similar column structures
- Formulas that are coefficient swaps (0.85/0.15 vs 0.70/0.30)
- Same null handling patterns with renamed columns
- Copying your own previous challenge with new domain terminology

**HOW TO AVOID DUPLICATES:**
- Use fundamentally different computation patterns (graph traversal, simulation, optimization, constraint satisfaction)
- Change the data flow direction (bottom-up vs top-down aggregation)
- Use different mathematical foundations (linear algebra vs statistics vs combinatorics)
- Design outputs that require genuinely different algorithms to produce

---

**CRITICAL: DO NOT USE EXISTING REPOSITORIES OR DOCUMENTED FORMULAS**

AI models like Claude Opus 4.5 have been trained on virtually every public repository, technical indicator library, financial formula, and standard algorithm. If a formula has a Wikipedia page, a library implementation, or exists in any public codebase - AI knows it perfectly and will pass 100%.

**THE ONLY WAY TO CREATE DIFFICULT CHALLENGES:**

Create **SYNTHETIC DATASETS** with **INVENTED BUSINESS RULES** that exist nowhere else.

---

## Synthetic Dataset Methodology

### Why This Works

| Approach | AI Pass Rate | Why |
|----------|--------------|-----|
| Standard formulas (EMA, RSI, MACD) | 100% | AI knows these perfectly |
| Public repo code | 90-100% | In training data |
| **Invented custom rules** | **30-70%** | **Not in training data** |

### How to Create Synthetic Challenges

**Step 1: Invent a Fictional Domain**
- "Galactic Freight Consortium shipping manifests"
- "Municipal Water Quality Compliance Bureau"
- "Transnational Insurance Claims Adjudication"
- "Industrial Equipment Depreciation Authority"

**Step 2: Create Custom Formulas That Don't Exist Anywhere**

BAD (AI knows these):
```
EMA = price * α + prev_EMA * (1-α)  # Wikipedia page exists
RSI = 100 - (100 / (1 + RS))         # In every trading library
NPV = Σ(cashflow / (1+r)^t)          # Standard finance
```

GOOD (invented, unique):
```
AdjustedRiskScore = base_score * (1 + penalty_factor ^ severity_level) / sqrt(days_since_event + 1)

ComplianceIndex = (violations_weighted * decay_rate ^ months_elapsed) + regional_modifier * tier_multiplier

AllocationFactor = min(1.0, (priority_score / threshold) * sqrt(demand_ratio)) * seasonal_coefficient[quarter]
```

**Step 3: Create Cascading Dependencies**

Design calculations where:
- Output B requires Output A
- Output C requires both A and B with different weightings
- Output D compares C across time windows
- Getting A slightly wrong makes D completely wrong

**Step 4: Include Implicit Rules**

Rules that must be inferred from data patterns:
- "Apply the standard regional adjustment" (but factors differ by region in the data)
- "Use tier-appropriate thresholds" (tiers have different rules discoverable only from data)
- Null handling that varies by column context
- Date boundaries with non-obvious edge cases

**Step 5: Generate Realistic Data**

```python
import pandas as pd
import numpy as np
np.random.seed(42)  # Reproducible

# Create 7+ interrelated CSV files
# Include edge cases at realistic rates (2-5%, not 50%)
# Add data quality issues to detect
# Use plausible distributions
```

---

## PROVEN CLAUDE WEAKNESSES - EXPLOIT THESE

These are real, verified weaknesses that cause Claude Opus 4.5 to fail. Design your challenges to hit AS MANY of these as possible simultaneously.

### WEAKNESS 1: Row-by-Row Stateful Processing
Claude's instinct is to vectorize everything with pandas. It FAILS when calculations require iterating row-by-row where each row's output depends on the PREVIOUS row's computed result (not just input data).

**How to exploit:**
- Running balances that reset when a condition is met
- Cumulative scores where each row modifies a state variable
- "Carry forward the previous result, but only if condition X, otherwise reset to base value"
- State machines: status transitions where current status depends on previous status + new event

**Example that breaks Claude:**
```
For each record in chronological order, compute running_exposure.
If risk_flag is True, running_exposure = previous_running_exposure * 1.5 + current_amount.
If risk_flag is False AND previous_running_exposure > threshold, running_exposure = previous_running_exposure * 0.8.
If risk_flag is False AND previous_running_exposure <= threshold, running_exposure = current_amount.
The threshold itself is the rolling average of the last 5 running_exposure values.
```
Claude will try to vectorize this and get wrong results. It MUST be computed row-by-row.

### WEAKNESS 2: Intermediate Rounding That Cascades
Claude almost always applies rounding at the END. It fails when specific intermediate results must be rounded BEFORE being used in the next calculation.

**How to exploit:**
- "Round the base_score to 2 decimal places, THEN multiply by the factor"
- "Truncate (not round) the intermediate value to integers before division"
- Different rounding rules at different stages (round half up for scores, round half even for financial, truncate for counts)

**Example that breaks Claude:**
```
Step 1: base = amount * rate (round to 2 decimals)
Step 2: adjusted = base * modifier (truncate to integer)
Step 3: final = adjusted / count (round to 4 decimals)
```
If you don't round at step 1, your step 2 is wrong, and step 3 is completely off.

### WEAKNESS 3: Off-by-One in Boundary Conditions
Claude consistently gets boundary conditions wrong when "inclusive vs exclusive" is ambiguous or when counting intervals.

**How to exploit:**
- "Within 90 days" - does day 90 count? State explicitly but make it matter.
- "The previous 5 records" - does this include the current one?
- "Between January and March" - inclusive of March 31?
- Window calculations: "use a 7-day lookback" starting from when?
- "At least 3 occurrences" - Claude sometimes uses > 3 instead of >= 3

### WEAKNESS 4: Complex Multi-Way Conditional Logic (5+ Branches)
Claude handles if/else well. It FAILS when there are 5+ mutually exclusive conditions that depend on combinations of multiple fields, especially when priorities between conditions are implicit.

**How to exploit:**
```
Classification rules (applied in priority order):
- If region is "APAC" AND amount > 10000 AND days_overdue > 30: category = "CRITICAL_APAC"
- If region is "APAC" AND amount > 10000: category = "HIGH_APAC"  
- If amount > 50000 regardless of region: category = "HIGH_GLOBAL"
- If days_overdue > 90 AND amount > 5000: category = "ESCALATED"
- If previous_category was "ESCALATED" AND days_overdue <= 30: category = "RESOLVED"
- If region in ("EMEA", "LATAM") AND currency != "USD": apply conversion first, then re-evaluate all above rules
- Default: category = "STANDARD"
```
Claude will miss the priority ordering, mishandle the re-evaluation after conversion, or get the boundary between overlapping conditions wrong.

### WEAKNESS 5: Multi-File Joins with Conditional Logic
Claude handles simple joins well. It FAILS when:
- The join key must be CONSTRUCTED from multiple columns
- Different join strategies apply for different record types
- Records that DON'T match need special handling (not just drop/fill)
- Join order matters (A join B, THEN result join C gives different results than A join C then result join B)

**How to exploit:**
- "Match transactions to accounts using account_id, but for legacy records (before 2020), use old_account_id instead"
- "Join with the rate table using the rate valid ON the transaction_date, not the latest rate"
- "If no matching rate exists, use the nearest prior rate, but if that rate is more than 30 days old, flag as stale and use the default rate"

### WEAKNESS 6: Null/Missing Data with Context-Dependent Rules
Claude typically applies one null-handling strategy globally. It FAILS when nulls mean different things in different columns or contexts.

**How to exploit:**
- "Null in amount means zero. Null in rate means use the previous rate. Null in category means EXCLUDE the record entirely."
- "Null in score means not yet evaluated - carry forward the previous score. But null in score when status is 'CLOSED' means the score was intentionally cleared."
- Missing dates should be treated as "ongoing" for duration calculations but as "unknown" for reporting

### WEAKNESS 7: Temporal Alignment Across Datasets
Claude FAILS when multiple time series must be aligned with different granularities and different rules for what to do when timestamps don't match exactly.

**How to exploit:**
- Daily data joined to monthly data: which month does Jan 31 belong to?
- "Use the rate EFFECTIVE on the transaction date" (rates.csv has start_date, end_date)
- Business days vs calendar days: "30 business days" is NOT 30 calendar days
- Fiscal years that don't start in January
- "Use the PREVIOUS quarter's value if current quarter data is not yet available"

### WEAKNESS 8: Aggregation with Exclusion Rules
Claude FAILS when the same data must be aggregated differently for different metrics, with different exclusion criteria.

**How to exploit:**
- "Count ALL records for volume metrics, but EXCLUDE flagged records for financial metrics, and EXCLUDE both flagged AND pending records for ratio calculations"
- "Duplicates count once for unique counts, but their FULL values are included in sum calculations"
- "Include cancelled orders in the order_count but exclude them from revenue, average_value, and fill_rate calculations"

### WEAKNESS 9: Precision-Sensitive Chained Arithmetic  
Claude makes tiny floating-point errors that compound across many chained operations. Design formulas where 6+ operations chain together.

**How to exploit:**
- Formula A uses 4 inputs, Formula B uses A + 3 new inputs, Formula C uses A + B + 2 new inputs
- Verify to 4+ decimal places (but with fair tolerances)
- Include operations that amplify small errors: division by small numbers, exponentiation, logarithms

### WEAKNESS 10: Discovery from Data Patterns
Claude CANNOT look at data and infer patterns. When a rule says "apply the standard regional adjustment" and the adjustment factors must be reverse-engineered from examples in the data, Claude will guess or make up values.

**How to exploit:**
- Include a "reference" section in the data where the adjustment has already been applied to some records
- The solver must figure out the factors from those reference records, then apply to new records
- "Use the tier thresholds as established in the historical data" (thresholds must be discovered)

---

## CHALLENGE BLUEPRINT: How to Combine Weaknesses

Here is a concrete example of a challenge structure that hits 8+ weaknesses and will cause Claude to fail most checks:

**Domain:** "Regional Cargo Compliance Scoring System"

**Input Files (8):**
1. `shipments.csv` - 2000 shipment records with dates, amounts, regions, cargo_type
2. `inspections.csv` - inspection results linked to shipments (some shipments have multiple)
3. `rate_schedule.csv` - compliance rates by region with effective_date ranges
4. `regional_config.csv` - region-specific thresholds and multipliers
5. `carrier_history.csv` - carrier performance scores (running history)
6. `exceptions.csv` - override rules for specific cargo types
7. `calendar.csv` - business day calendar by region (holidays differ!)
8. `reference_scores.csv` - pre-computed scores for Q1 (solver must reverse-engineer the formula coefficients)

**Why this breaks Claude:**

| Step | Weakness Hit | What Goes Wrong |
|------|-------------|-----------------|
| Join shipments to rates using effective_date range | W5, W7 | Wrong rate if date logic is off |
| Null in amount = 0, null in inspection = "pending", null in rate = use previous | W6 | Claude applies uniform null handling |
| Compute running_compliance_score per carrier chronologically | W1 | Claude vectorizes instead of iterating |
| Round intermediate compliance_score to 3 decimals before penalty calc | W2 | Claude rounds at end, gets wrong penalties |
| Classify into 6 tiers based on score + region + cargo_type combinations | W4 | Claude misses priority ordering |
| "Within 30 business days" (not calendar days, region-specific holidays) | W3, W7 | Almost always gets count wrong |
| Count metrics include all records, financial metrics exclude pending, ratios exclude pending+cancelled | W8 | Claude uses same filter everywhere |
| Final risk_index = chain of 4 custom formulas feeding into each other | W9 | Precision errors compound |
| Reverse-engineer penalty coefficients from reference_scores.csv | W10 | Claude guesses values |

**Output Files (5):**
1. `shipment_scores.csv` - per-shipment compliance scores
2. `carrier_summary.csv` - per-carrier aggregated metrics
3. `regional_report.csv` - per-region statistics with tier distributions
4. `anomalies.csv` - flagged records with reasons
5. `quarterly_comparison.csv` - Q1 vs Q2 trend analysis

**This design makes it nearly impossible for Claude to get all 25+ checks right.**

---

## Challenge Requirements

Your task should push the limit on an AI agent's ability to:

- Implement custom formulas exactly as specified
- Discover implicit rules from data patterns
- Handle cascading dependencies correctly
- Navigate complex multi-file relationships
- Produce accurate, well-formatted outputs

**Not Just Simple Operations:**
Each data point should require genuine data science skills - not just simple aggregations or lookups that can be solved with basic operations.

---

## Evaluation Criteria

Your challenge is scored on a 10-point scale after submission:

| Category | Points | What It Measures |
|----------|--------|------------------|
| Difficulty | 3 | AI pass rate (target: 1-3 out of 10 passes, NOT 100%) |
| Failure Quality | 3 | How AI fails (partial credit, reasoning errors) |
| Prompt Conciseness | 2 | No strict char limit, just be concise and complete |
| Fairness | 2 | No unfair traps, all info stated |

**CRITICAL**: 100% AI pass rate = -3 difficulty points = POOR score

**IDEAL DIFFICULTY**: AI should pass 1 to 3 out of 10 runs. If AI passes 0/10, the challenge may be unfair or ambiguous. If AI passes 4+ out of 10, the challenge is too easy. The sweet spot is when AI gets most checks right (45-52 out of 55) but consistently fails on a few specific checks due to genuine reasoning errors, not ambiguity.

The tests should be at least 25 complex tests and solution should be more than 1000 lines to solve it for a highly complex challenge standards.

**DATA COMPLEXITY REQUIREMENTS:**
- At least 7 input CSV files in task/ directory with interrelated data
- At least 5 output files that the solver must generate
- Data must have complex relationships requiring multi-file joins and cross-referencing
- Include temporal dependencies, hierarchical structures, and conditional logic chains

---

## Directory Structure

```
challenge_name/
    task/
        prompt.txt           <- Human-readable task description  
        input_data_1.csv     <- Multiple data files (.csv, .json, .xlsx, .parquet, .txt)
        input_data_2.csv     <- Minimum 7 input files with interrelated data
        input_data_3.csv
        ...
    ground_truth/
        oracle_solution.py   <- Reference solution (NO COMMENTS) - should be 500+ lines
        verifier.py          <- Scoring script (NO COMMENTS) - should be 300+ lines
        golden_output_1.csv  <- Expected correct outputs
        golden_output_2.csv  <- Multiple golden files for complex verification
        ...
    other/
        generate_data.py     <- Data generation scripts, helper tools, etc.
        notes.txt            <- Any development notes or scratch work
        ...                  <- Any CSVs or files NOT uploaded to task/ or ground_truth/
```

**The `other/` folder** stores any files created during challenge development that are NOT part of the submission. This includes data generation scripts, intermediate CSVs, scratch files, and anything else that doesn't belong in task/ or ground_truth/. This keeps the submission folders clean and makes it clear exactly which files to upload.

**CODE STYLE REQUIREMENTS:**
- oracle_solution.py and verifier.py must contain NO inline comments and NO docstrings
- Remove all # comments, triple-quote docstrings, and explanatory text from code
- Code should be self-explanatory through clear variable and function names only

---

## RULE ONE: Problem Title

Choose a clear, descriptive title (under 100 chars).
Must hint at complexity involved.

Good: "Insurance Claims Loss Ratio Analysis with Recovery Adjustments"
Bad: "CSV Parser"

---

## RULE TWO: Data Upload

**File Requirements:**
- Formats: .csv, .txt, .xlsx, .json, .parquet
- Max 25 files, each under 10 MB, total under 100 MB

**Upload ONLY:**
- Input data files the solver needs

**DO NOT upload:**
- oracle_solution.py
- verifier.py  
- golden output / answer files
- Files from the other/ folder (generation scripts, scratch files, etc.)

**Data Quality:**
- Use realistic distributions
- Fixed random seed if synthetic
- Include edge cases at realistic rates (not obvious traps)

---

## RULE THREE: Problem Prompt

Write naturally, like a real analyst request.

### Must Be:
- **Clear**: Unambiguous instructions
- **Fair**: All requirements stated or discoverable from data
- **Complete**: Exact output filenames, columns, formulas
- **Aligned**: What you ask for MUST match golden output

### Formatting Style:
- Write as continuous flowing prose, not sectioned or bulleted lists
- No section headers, colons, hyphens, or markdown formatting
- No unnecessary story, context, or filler text that adds nothing to the requirements
- Every sentence must convey required information for solving the task
- Reads like a natural paragraph, not a specification document

### DO:
- State exact output filename
- List all required columns
- Provide formulas for non-standard calculations
- Define domain terms explicitly
- Specify thresholds and rounding rules

### DO NOT:
- List input file names and their column schemas in the prompt. Let the agent explore the CSV files in the task directory and discover the structure themselves. Just say "Read all CSV input files from the task directory."
- Expect columns not mentioned in prompt
- Use formulas the model must "discover"
- Ask to "reverse engineer" patterns
- Assume deep domain knowledge
- Hide requirements

### Good Prompt Example:

```
I need to prepare our 2024 claims performance report. Can you analyze the 
claims data and help me understand our loss experience?

The data includes claims.csv with policy info and payments.csv with 
subrogation recoveries. Recoveries reduce net losses.

Save results as claims_analysis.csv with columns: section, metric, value,
claim_count, gross_paid, recoveries, net_incurred, earned_premium, loss_ratio.

Include three sections:
- data_quality: lapsed_policy_claims (claims on cancelled policies), 
  date_anomalies (loss date after report date), duplicate_claims (same 
  policy + loss date + similar amount within $100). Exclude duplicates 
  from financial metrics.
- by_policy_type: One row per type (Auto, Home, Commercial, Umbrella)
- summary: totals and overall_loss_ratio

loss_ratio = net_incurred / earned_premium, rounded to 4 decimals.
```

### Bad Prompt (Too Easy):

```
Parse the CSV. Calculate distance between points. Categorize by size.
```

---

## RULE FOUR: Solution & Test Set

### Workspace Layout:

```
workspace/
├── task/               <- Data files + solver output goes here
│   ├── prompt.txt
│   ├── data.csv
│   └── output.csv      <- Solver writes here
├── ground_truth/       <- CWD for both scripts
│   ├── oracle_solution.py
│   ├── verifier.py
│   └── golden_output.csv
└── other/              <- NOT uploaded; generation scripts, scratch files, unused CSVs
    ├── generate_data.py
    └── ...
```

### Path References:
- From ground_truth/: Use `Path(__file__).parent` for sibling files
- To access task/: Use `Path(__file__).parent.parent / "task"`

### Available Libraries (Python 3.12):
pandas, numpy, scipy, scikit-learn, networkx, openpyxl, pyxlsb, pyxlsb2, fastparquet, pyarrow

### oracle_solution.py

**CRITICAL: NO COMMENTS (inline or docstrings)**

```python
import pandas as pd
import numpy as np
import re


def parse_data(text):
    pattern = r'(\d+),(\d+)'
    return re.findall(pattern, text)


def main():
    df = pd.read_csv("input_data.csv")
    
    results = []
    for _, row in df.iterrows():
        value = complex_calculation(row)
        results.append({'id': row['id'], 'result': value})
    
    output = pd.DataFrame(results)
    output.to_csv("output.csv", index=False)
    print("Results saved to output.csv")


if __name__ == "__main__":
    main()
```

### verifier.py

**CRITICAL: NO COMMENTS (inline or docstrings)**

```python
import pandas as pd
import sys
from pathlib import Path


def load_golden():
    golden_path = Path(__file__).parent / "golden_output.csv"
    return pd.read_csv(golden_path)


def load_submission(task_dir):
    submission_path = task_dir / "output.csv"
    if not submission_path.exists():
        return None
    return pd.read_csv(submission_path)


def check_value(expected, actual, tol_abs=0.01):
    if pd.isna(expected) and pd.isna(actual):
        return True
    if pd.isna(expected) or pd.isna(actual):
        return False
    try:
        expected, actual = float(expected), float(actual)
        return abs(actual - expected) <= tol_abs
    except (ValueError, TypeError):
        return str(expected).strip().lower() == str(actual).strip().lower()


def verify(task_dir):
    golden = load_golden()
    submission = load_submission(task_dir)
    max_score = len(golden)

    if submission is None:
        print("No submission found")
        print(f"0/{max_score}")
        sys.exit(1)

    score = 0
    required_cols = ["id", "result"]

    missing = [c for c in required_cols if c not in submission.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print(f"0/{max_score}")
        sys.exit(1)

    if len(submission) != len(golden):
        print(f"Row count mismatch: expected {len(golden)}, got {len(submission)}")

    merged = golden.merge(submission, on="id", suffixes=("_expected", "_actual"))

    for _, row in merged.iterrows():
        row_id = row["id"]
        if check_value(row["result_expected"], row["result_actual"]):
            score += 1
            print(f"  PASS  {row_id}")
        else:
            print(f"  FAIL  {row_id}: expected {row['result_expected']}, got {row['result_actual']}")

    print(f"{score}/{max_score}")
    sys.exit(0 if score == max_score else 1)


if __name__ == "__main__":
    task_dir = Path(__file__).parent.parent / "task"
    verify(task_dir)
```

### Verifier Principles:
1. Deterministic - same input = same score
2. Fair - only checks what prompt specifies
3. Tolerances - account for rounding (0.01 absolute typical)
4. Layered checks - File exists → Schema → Row count → Values
5. Output format - Last line MUST end with X/Y (e.g. "15/15")
6. Exit codes - 0 for perfect, 1 otherwise

### What NOT to Verify:
- Column ordering (unless required)
- Exact whitespace
- Case sensitivity (unless meaningful)
- Intermediate results

---

## RULE FIVE: Review & Submit

All automated checks must pass:

| Check | Requirement |
|-------|-------------|
| Title | Provided, valid characters |
| Data Upload | Files uploaded, valid formats, within size |
| Prompt | Provided |
| Solution | oracle_solution.py provided |
| Verifier | verifier.py provided |
| Ground Truth | Files uploaded, valid formats |
| No Overlap | Task and ground truth files don't overlap |
| Originality | No similar problems found |
| Pre-Check | Verifier passes (X/X score) |

---

---

## RULE NINE: Prompt-Oracle Alignment (Fairness)

**EVERY behavior the oracle implements silently must be explicitly stated in the prompt.**

Unfair verdicts are triggered when an agent makes a reasonable interpretation of the prompt that differs from the oracle's hidden logic. The agent does not fail because it cannot reason — it fails because the prompt left out a required specification. Evaluators will flag this every time.

---

### THE FIVE ANTI-PATTERNS THAT CAUSE UNFAIR VERDICTS

These are the exact root causes observed in real failed evaluations. Memorize and avoid all five.

---

**ANTI-PATTERN 1: Undocumented Sort Tie-Breaker**

Oracle sorts by `[timestamp, request_id]` but prompt only says "process chronologically". Agent sorts by `[timestamp]` alone — valid interpretation, wrong output.

With even 5% of rows sharing a timestamp, every running counter and cumulative value diverges, causing 3–10% mismatch on high-strictness checks.

Rule: Every `sort_values()` call with more than one key must have ALL keys stated in the prompt, in order.

BAD: "Process each request chronologically."

GOOD: "Sort requests by timestamp ascending, using request_id ascending as a secondary sort key when timestamps are equal, then process in that order."

---

**ANTI-PATTERN 2: Silent Field Validation (Implicit Whitelist)**

Oracle contains `if field_name in known_fields_dict:` and silently ignores rows where the field name does not match. The data contains a non-matching field name. Agent sees the row, applies it, oracle ignores it — results diverge on every affected tenant.

Rule: Any lookup, mapping, or dictionary key check in the oracle that can silently skip a row must be documented in the prompt as an explicit filter rule.

BAD: "Apply quota_overrides that are active on the request timestamp."

GOOD: "Apply quota_overrides that are active on the request timestamp. The override_field must exactly match one of: hourly_request_limit, daily_request_limit, daily_token_limit. Silently ignore any override row whose override_field does not exactly match one of those three names."

---

**ANTI-PATTERN 3: Ambiguous Sentinel Values in Data**

Data contains a cell like `override_value=0` with `reason='Unlimited enterprise'`. The value 0 is semantically ambiguous — does it mean zero limit (no requests allowed) or unlimited (no limit applied)? Agent picks one interpretation, oracle picks the other.

This is compounded when Anti-Pattern 2 also applies (the row may be silently ignored anyway). The ambiguity still causes an UNFAIR verdict because the evaluator flags the data design as misleading.

Rule: Either remove sentinel-value rows from the data entirely, or document the exact meaning of every special value in the prompt.

BAD: Data contains `override_value=0, reason=Unlimited enterprise` with no prompt explanation.

GOOD: Either remove the row, or add to prompt: "An override_value of 0 for a limit field means the limit is disabled and no throttling applies for that dimension."

---

**ANTI-PATTERN 4: Hardcoded Constant Not Derivable from Data**

Oracle computes `base_cost = price * (3 / 30)` — a hardcoded fraction the agent cannot derive from the input files. Agent uses a logically correct alternative like `days_in_period / 31` or `days_in_period / days_in_month`, producing a different result.

Rule: Any constant or formula used by the oracle must either be derivable from the data (no guessing) or explicitly stated in the prompt including intermediate rounding steps.

BAD: "Compute base_cost as the base monthly price scaled by the coverage period."

GOOD: "Compute base_cost by first computing period_fraction as the count of distinct calendar days in the tenant data period divided by the actual number of calendar days in the billing month, rounded to 4 decimal places, then multiply by the tier base_monthly_price and round to 2 decimal places."

---

**ANTI-PATTERN 5: Oracle Logic That Depends on Input File Schema Not Visible in Prompt**

Oracle looks up column names from an input CSV at runtime (e.g., a limits dictionary built from tier_policies columns). If any input row references a column name that does not exist in that CSV, the oracle skips it silently. The agent reads the same column name from the CSV, treats it as valid, and applies it.

Rule: When the valid values of a field in one CSV are directly determined by column names in another CSV, state this relationship in the prompt.

BAD: "Apply the override."

GOOD: "The override_field value must match a column name present in tier_policies. If no matching column exists, ignore the override row."

---

### PROMPT FAIRNESS CHECKLIST (Run Before Every Submission)

For every piece of logic in oracle_solution.py, answer these questions:

| Oracle Does This | Prompt Must State This |
|---|---|
| `sort_values([col_a, col_b])` | Both sort keys and their direction |
| `if field in dict:` (silent skip) | Exact valid values for the field and what happens when the field does not match |
| `value == 0` treated as sentinel | What 0 means vs a real zero |
| `round(x, n)` at intermediate step | That intermediate rounding happens and at which step |
| Uses a hardcoded constant (e.g. 30, 0.03, 1.5) | Either derive it from data or state the exact value and formula |
| Different null behavior per column | Each column's null rule stated separately |
| Priority ordering in conditional chain | Priority order explicitly listed (check rule 1 first, then rule 2, etc.) |
| Includes historical records from a separate file | State that historical records are included and which file they come from |
| Ignores records matching a filter (e.g. resolved=YES) | State the filter condition explicitly |
| Output sorted by specific columns | State the exact sort columns and direction |

---

### ORACLE SILENT LOGIC AUDIT

Before submitting, read through oracle_solution.py and flag every line that matches these patterns. Each flagged line needs a corresponding sentence in prompt.txt.

```
Patterns to flag in oracle:
  if x in dict:                        -> document valid keys in prompt
  if x in [list]:                       -> document the list or condition in prompt
  sort_values([a, b])                   -> document both sort keys in prompt
  if value == 0:                        -> document sentinel meaning in prompt
  constant = 3 / 30                     -> document formula or state the value in prompt
  if col not in df.columns:             -> document expected columns in prompt
  df[df['col'] == SPECIFIC_VALUE]       -> document the filter condition in prompt
  round(x, n)  # intermediate           -> document that intermediate rounding happens in prompt
  if prev_value is None: use_default    -> document the default value and condition in prompt
  .fillna(specific_value)               -> document null handling for that column in prompt
```

If you cannot add a prompt sentence for a flagged oracle line without "giving away" the answer, that means the logic should be moved to the difficulty layer (where it trips AI naturally), not hidden as a silent filter.

---

## Difficulty Guidelines

### Target: 1-3 out of 10 AI Pass Rate

The ideal challenge is one where AI scores high (45-52 out of 55 checks) but consistently gets a few specific checks wrong due to genuine reasoning errors in stateful logic, timing, or cascading dependencies. This means AI understands the problem but cannot perfectly execute the hardest parts.

**TOO EASY (will fail evaluation):**
- Simple parsing + basic math
- No interacting rules
- Obvious edge cases
- Single-step calculations
- AI passes 4 or more out of 10 runs

**GOOD DIFFICULTY:**
- Multi-step calculations with cascading dependencies
- Stateful row-by-row processing where order of operations matters
- Conditional business logic where timing of updates affects results
- Counter or accumulator logic where reset conditions are subtle
- Rolling or cumulative scores that depend on prior computed values
- Multiple interacting domain rules that compound errors
- Cross-file joins with conditional strategies
- AI passes 1 to 3 out of 10 runs, failing on specific checks not random noise

**PROVEN PATTERNS THAT TRIP AI (from real evaluations):**

These patterns consistently cause AI to fail 7 to 9 out of 10 runs while still scoring 45+ out of 55 checks. Use these as your blueprint.

1. Stateful counters with order-dependent updates: When a counter must be read BEFORE being updated in the same iteration, AI almost always gets the timing wrong. Example: consecutive violations count must be 0 when current tier is below threshold, but AI tends to carry the previous count forward before resetting.

2. Cascading penalty calculations: When a stateful value feeds into a multiplication chain, a single off-by-one in the counter propagates through penalty amount, then maintenance credit, then net penalty, then regional totals. One wrong counter can cause 20+ check failures across multiple output files.

3. Conditional carry-forward with different defaults per column: Nulls meaning different things for different columns in the same row. Example: null pH means carry forward last valid value defaulting to caution limit, null lead means 0, null coliform means exclude from that specific average only, null flow rate means offline and exclude from all quality calculations. AI usually handles 3 out of 5 correctly.

4. Rolling composite scores with branch-dependent weights: A score like RCS where the blending formula changes based on the input value (high quality uses 85/15, medium uses 60/40, low overwrites entirely, no data decays by 5%). AI frequently picks the wrong branch or applies weights in wrong order.

5. Priority classification with compound conditions: Six or more tiers where each has AND/OR conditions involving different computed values, and AI must check them in strict priority order. The middle tiers with compound conditions are where AI makes mistakes.

6. Cross-file temporal joins with effective date ranges: Joining penalty rates by date range, regional adjustments by quarter, and holidays by region, where each join has different matching rules. AI often gets one join slightly wrong affecting downstream calculations.

**Examples of Hard Requirements:**
- "Exclude duplicates from financial metrics BUT include in count metrics"
- "Use the PREVIOUS period's rate if current is missing, unless it's Q1"
- "Claims on lapsed policies count as data quality issues, not valid claims"
- "Recoveries offset losses only if recovery_date is within 90 days of claim_date"
- "Consecutive violations count for current month is 0 when current tier is below 2, otherwise count prior consecutive months with tier at or above 2"

---

## Pre-Submission Checklist

### Prompt:
- [ ] Output filename explicit
- [ ] All columns listed with types/units
- [ ] All formulas provided (no discovery required)
- [ ] Rounding rules stated for each step, not just final output
- [ ] Missing data handling stated per column (different nulls may mean different things)
- [ ] Edge case handling stated
- [ ] Sounds natural, not robotic
- [ ] Every `sort_values()` with multiple keys in the oracle has ALL keys and directions stated
- [ ] Every silent filter or field-name validation in the oracle is documented
- [ ] No sentinel values in data (0 meaning unlimited, -1 meaning disabled) without explicit prompt explanation
- [ ] No hardcoded constants in oracle without a formula or derivation stated in prompt
- [ ] Any priority ordering in conditional logic is explicitly stated as an ordered list

### Data:
- [ ] Realistic distribution
- [ ] Fixed seed if synthetic
- [ ] Edge cases at realistic rates (5-15% of rows)
- [ ] Multiple interacting conditions

### Oracle:
- [ ] NO COMMENTS anywhere
- [ ] Uses only approved libraries
- [ ] Exits with code 0
- [ ] Output matches golden exactly

### Verifier:
- [ ] NO COMMENTS anywhere
- [ ] Uses Path(__file__) for file references
- [ ] Deterministic scoring
- [ ] X/Y format on last line
- [ ] Exit 0 for perfect, 1 otherwise
- [ ] Appropriate tolerances

### Difficulty:
- [ ] Multi-step reasoning required
- [ ] At least 2 interacting business rules
- [ ] Data quality issues to handle
- [ ] NOT solvable with simple groupby + apply

---

## Final Principle

**Measure reasoning, not trick the model.**

If 100% of AI agents pass, your challenge is too easy.
If 0% pass but it's because of hidden rules, your challenge is unfair.

Target: 1 to 3 out of 10 pass rate with failures showing partial understanding (scoring 45+ out of 55 checks).

---

## RULE SIX: What Evaluations Look Like

### What an EASY Challenge Evaluation Looks Like (POOR SCORE)

```
Evaluation Pipeline: 1/10 POOR
100% weighted pass rate across 10 runs

Difficulty (-3/3 pts) — Way too easy
100% pass rate — problem is too easy.

Failure Quality (0/3 pts) — No failures
All runs passed, so failure quality can't be assessed.

Prompt Conciseness (+2/2 pts) — Very concise
Prompt is 1,650 chars — under the 4,000 char ideal.

Fairness (+2/2 pts) — Fair
No unfair conditions detected.

Verdict Distribution:
PASS CORRECT: 10/10 (100%)
```

**This is a FAILING evaluation** — your challenge was too easy.

### What an UNFAIR Challenge Evaluation Looks Like

```
Evaluation Pipeline: Unfair evaluation condition
Score invalidated — the agent failed due to unfair evaluation conditions,
not its own shortcomings.

Verdict Distribution:
FAIL WRONG CALCULATION: 5/10 (50%)
FAIL UNFAIR EVAL: 4/10 (40%)
FAIL NO OUTPUT: 1/10 (10%)

Unfair Evaluation Detected (4 judges flagged)
```

**Common Unfair Conditions Identified:**
- Ambiguous threshold interpretations (e.g., "minimum 6 bars between" - does this mean gap >= 6 or 6 intervening bars?)
- Underspecified initialization methods (e.g., "standard MACD" without specifying EMA seeding)
- Overly tight tolerances that penalize valid alternative implementations
- Asymmetric logic not clearly stated in prompt

---

## RULE SEVEN: Strict Fairness & Testing Requirements

**MUST BE FOLLOWED STRICTLY**

### Local Testing Required

The challenge MUST be tested locally before submission. Run both oracle_solution.py and verifier.py to ensure they work correctly.

### Prompt Requirements

The prompt.txt must be:
- **Start with "Generate ..."** — first line states the task, scope (entity count, groups, time range), date format, and expected primary output row count. Example: "Generate quarterly water quality compliance for 30 stations, 5 regions (NORTH/SOUTH/EAST/WEST/CENTRAL), Jan-Jun 2024. station_compliance must have 180 rows: one per station-month incl zero-reading months. Use YYYY-MM format for month columns."
- **Do not list input file names and column schemas** — do not enumerate input files or their columns. Let the agent explore the CSVs in the task directory. Just include "Read all CSV input files from the task directory." in the opening line.
- **Short, small, and concise**
- **No non-ASCII characters, symbols, or special characters**
- **No computerized or rigid language**
- **Cover WHAT and WHY, not HOW**

### Observable vs Implementation Aspects

**Focus on OBSERVABLE aspects, NOT implementation aspects.**

The tests and challenge must cover super complex observable behavior, not specific implementation details.

**What this means:**
- User can use their own method and still solve it
- Don't strictly require exact APIs, strings, or patterns
- Don't enforce a specific implementation approach
- Verify the RESULT, not HOW it was computed

**Good Observable Prompt Style:**
```
Add a, calculate b, return c...
```
Short, concise, covering the challenge requirements without dictating implementation.

---

## RULE EIGHT: Complexity Requirements

### Time & Space Complexity Requirements

Where applicable, apply these complexity standards:

- Target O(log n), O(n), or O(n log n) complexity - avoid O(n²) or worse unless absolutely necessary
- Optimize space complexity: prefer O(1) auxiliary space where possible, O(log n) for recursive solutions
- Use amortized analysis for data structures (e.g., dynamic arrays, union-find with path compression)

### Algorithmic Techniques (Apply Where Applicable)

| Technique | When to Use |
|-----------|-------------|
| **Divide & Conquer** | Break problems into subproblems, solve recursively, merge results efficiently |
| **Dynamic Programming** | Overlapping subproblems with optimal substructure (memoization or tabulation) |
| **Greedy Algorithms** | Local optimal choices lead to global optimum with provable correctness |
| **Binary Search** | Sorted data or monotonic functions - reduce search space logarithmically |
| **Two Pointers / Sliding Window** | Array/string traversal in O(n) instead of O(n²) |
| **Prefix Sums / Difference Arrays** | Range queries and updates in O(1) after O(n) preprocessing |
| **Bit Manipulation** | Space-efficient solutions and O(1) operations |
| **Union-Find with Path Compression** | Disjoint set operations in near O(1) amortized |
| **Segment Trees / Fenwick Trees** | Range queries with O(log n) updates |
| **Monotonic Stack/Queue** | Next greater element, sliding window maximum in O(n) |
| **Trie / Radix Trees** | Prefix-based string operations |
| **Graph Algorithms** | BFS/DFS, Dijkstra's, topological sort, strongly connected components |

### Code Quality Standards

- Use proper generics and type inference to maximize type safety
- Implement proper error handling with descriptive error types
- Write pure functions where possible - avoid side effects
- Use immutable data patterns unless mutation is necessary for performance
- Leverage lazy evaluation and generators for memory-efficient iteration
- Apply early returns and guard clauses to reduce nesting

### Advanced Patterns

- **Functional Composition**: Chain operations using map, filter, reduce with optimal short-circuiting
- **Iterator Protocol**: Implement custom iterables for memory-efficient streaming
- **Structural Sharing**: For immutable updates without full copies
- **Tail Call Optimization**: Structure recursion for TCO where supported
- **Object Pooling**: Reuse objects to minimize GC pressure in hot paths

### Performance Optimizations

- Minimize allocations in hot paths - preallocate arrays when size is known
- Prefer `for` loops over `.forEach()` in performance-critical sections
- Cache computed values and array lengths in tight loops
- Use `Map`/`Set` over plain objects for frequent lookups (O(1) guaranteed)
- Avoid unnecessary spreading/destructuring in loops
- Consider branch prediction - put common cases first in conditionals

### Must NOT Do

- Use naive nested loops when better algorithms exist
- Implement brute force when polynomial/logarithmic solutions are achievable
- Use simple `.includes()` or `.indexOf()` repeatedly when a Set/Map lookup suffices
- Create unnecessary intermediate arrays when streaming/generators work
- Ignore edge cases that could cause performance degradation

### Solution Must Demonstrate

- Deep understanding of algorithmic paradigms
- Mastery of language-specific optimizations
- Production-grade error handling
- Code that would pass rigorous code review

---

## DIFFICULTY LEVEL REQUIREMENT

**CRITICAL: The challenge should be difficult enough that LLMs like Claude Opus 4.5 cannot solve it at a go.**

### Minimum Weakness Coverage

**Your challenge MUST exploit at least 5 of the 10 proven Claude weaknesses listed above.**

Use this checklist when designing:

- [ ] WEAKNESS 1: Row-by-row stateful processing (not vectorizable)
- [ ] WEAKNESS 2: Intermediate rounding that cascades
- [ ] WEAKNESS 3: Off-by-one boundary conditions that matter
- [ ] WEAKNESS 4: 5+ branch conditional logic with priority ordering
- [ ] WEAKNESS 5: Multi-file joins with conditional join strategies
- [ ] WEAKNESS 6: Context-dependent null handling (nulls mean different things)
- [ ] WEAKNESS 7: Temporal alignment across different granularities
- [ ] WEAKNESS 8: Aggregation with different exclusion rules per metric
- [ ] WEAKNESS 9: Precision-sensitive chained arithmetic (6+ operations)
- [ ] WEAKNESS 10: Pattern discovery from reference data

**If your challenge hits fewer than 5, it WILL be too easy.**

### Challenge Complexity Scorecard

Rate your challenge before submitting:

| Factor | Points | Your Score |
|--------|--------|------------|
| Number of Claude weaknesses exploited (1 pt each) | /10 | ___/10 |
| Number of input files | /3 (1pt=3-4, 2pt=5-6, 3pt=7+) | ___/3 |
| Number of output files | /2 (1pt=2-3, 2pt=4+) | ___/2 |
| Solution line count | /3 (1pt=300-500, 2pt=500-800, 3pt=800+) | ___/3 |
| Number of verification checks | /2 (1pt=15-20, 2pt=25+) | ___/2 |
| Cascading dependency depth | /3 (1pt=2-deep, 2pt=3-deep, 3pt=4+-deep) | ___/3 |
| Custom invented formulas | /2 (1pt=1-2, 2pt=3+) | ___/2 |

**Minimum score to submit: 15/25**
**Target score for hard challenge: 20+/25**

### Target Difficulty

- 1 to 3 out of 10 AI pass rate (NOT 100%, NOT 0%)
- AI should score high on most checks (45+ out of 55) but fail on specific stateful or cascading checks
- Failures should show partial understanding, not complete inability
- Multiple interacting rules that require careful reasoning
- Stateful processing where order of operations determines correctness
- Edge cases that test domain understanding

### WHY AI KEEPS PASSING - AVOID THESE PATTERNS

**DO NOT USE well-documented formulas that AI knows:**
- Standard technical indicators (EMA, RSI, MACD, Bollinger Bands, ADX, Stochastic)
- Standard financial calculations (NPV, IRR, Sharpe Ratio, CAGR, WACC)
- Standard statistical measures (mean, std, correlation, z-score, p-value)
- Standard ML metrics (accuracy, precision, recall, F1, AUC-ROC)
- Any formula that has a Wikipedia page or is in common libraries like pandas, numpy, scipy, scikit-learn, ta-lib

**AI will pass 100% if:**
- All formulas are clearly stated AND are simple/standard
- The logic is sequential without hidden dependencies
- Edge cases are mentioned or obvious
- The domain is well-known (finance, trading, insurance, etc.)
- Calculations can be vectorized with pandas
- Null handling is uniform across all columns
- Only 1-2 business rules interact

### HOW TO CREATE TRULY DIFFICULT CHALLENGES

**Combine multiple weaknesses simultaneously:**

Example challenge structure that will break Claude:
```
1. Load 8 CSV files with non-obvious relationships (WEAKNESS 5)
2. Clean data with context-dependent null rules (WEAKNESS 6) 
3. Align daily transactions to monthly rates using effective dates (WEAKNESS 7)
4. Compute running_risk_score row-by-row where each score depends on previous (WEAKNESS 1)
5. Apply 6-way classification with priority ordering (WEAKNESS 4)
6. Round intermediate scores before using in next formula (WEAKNESS 2)
7. Aggregate with different exclusion rules per metric (WEAKNESS 8)
8. Chain 3 custom formulas where output of each feeds next (WEAKNESS 9)
9. Final output compares results across rolling 90-day windows (WEAKNESS 3)
```

This hits 9/10 weaknesses. Claude WILL fail multiple checks.

### Signs Your Challenge is Too Easy

- Single-step calculations
- Simple parsing + basic math
- No interacting rules
- Obvious edge cases
- Solvable with simple groupby + apply
- Uses standard/documented formulas
- All logic is explicitly stated in prompt
- Less than 500 lines to solve
- Hits fewer than 3 weaknesses
- All calculations are vectorizable
- Uniform null handling

### Signs Your Challenge is Appropriately Difficult

- 800+ lines needed for correct solution
- 7+ input files with complex relationships
- 5+ output files with interdependencies
- Custom formulas not found anywhere online
- Logic must be inferred from data patterns
- 25+ verification checks
- Cascading calculations where early errors propagate
- Hits 5+ of the 10 proven Claude weaknesses
- Requires row-by-row iteration (not vectorizable)
- Context-dependent null/edge case handling
- Multiple competing valid interpretations that must be resolved
