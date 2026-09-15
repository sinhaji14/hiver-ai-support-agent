
````markdown
# Hiver AI Support Agent

An AI customer-support agent for AmazonHelp that classifies customer issues, retrieves relevant historical resolutions, generates grounded responses, and decides when a human should take over.

The system is designed as a **conservative, retrieval-grounded support agent**: when evidence is weak or the case is high-risk, it prefers escalation over unsupported automation.

---

## 1. Overview

For an incoming customer message, the pipeline performs:

```text
Customer message
       │
       ▼
Intent classification
       │
       ▼
Historical retrieval
       │
       ▼
Escalation / automation decision
       │
       ├──────────────► HUMAN
       │
       ▼
Grounded response generation
       │
       ▼
Safety checks
       │
       ▼
Final response
````

### Core goals

A good support agent should:

* Correctly identify the customer's primary support need.
* Retrieve useful historical support examples.
* Avoid automatically handling risky account-specific cases.
* Generate concise responses grounded in historical support behavior.
* Avoid inventing policies, refunds, compensation, dates, guarantees, or completed actions.
* Avoid leaking historical customer information or identifiers.

### What is not built

This is an offline prototype rather than a production customer-support system.

It does **not** include:

* Live CRM or ticketing-system integration.
* Access to customer accounts or order systems.
* Real-time order-status lookup.
* Execution of refunds, replacements, cancellations, or other transactions.
* A production policy engine.
* Authentication or authorization infrastructure.
* Human-agent workflow integration.

The generated responses are therefore limited to actions supported by historical evidence.

---

# 2. Dataset

The project uses the Kaggle **Customer Support on Twitter** dataset.

Dataset:

```text
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
```

The dataset contains customer tweets and brand-support responses.

## Target brand

AmazonHelp was selected because it had the largest support-response volume among the candidate brands examined.

Key statistics:

| Metric                              |   Count |
| ----------------------------------- | ------: |
| AmazonHelp support tweets           | 169,840 |
| Support tweets with parent IDs      | 169,287 |
| Matched customer parent tweets      | 154,976 |
| Clean customer-support interactions | 168,814 |
| Usable conversations                |  55,733 |

The conversation builder produced 55,733 usable conversations.

The final leakage-safe retrieval corpus contains:

```text
98,232 historical interactions
```

after excluding entire conversations containing golden-set examples.

---

# 3. Golden Evaluation Set

A **200-example hand-labelled golden set** was created for evaluation.

The examples were randomly sampled using:

```text
random seed = 42
```

Each example was manually assigned one primary intent.

### Annotation guideline

The primary intent was defined as:

> The issue most directly addressed by the customer's request and historical support response.

The same 200 examples were also manually assigned an escalation-policy label:

```text
AUTO-HANDLE: 72
HUMAN:       128
```

The escalation labels represent human policy judgments about whether a case is safe to automate. They are not objective ground truth.

## Golden-set distribution

| Intent                                  |   Count |
| --------------------------------------- | ------: |
| Delivery Delay & Tracking               |      56 |
| Missing, Misdelivered & Lost Package    |      11 |
| Damaged, Defective & Incorrect Item     |       8 |
| Returns, Refunds & Replacements         |       9 |
| Billing, Payment & Unauthorized Charges |       5 |
| Pricing, Promotions & Cashback          |      11 |
| Account Access & Security               |       9 |
| Technical & Digital Services            |      22 |
| Order, Subscription & Pre-order Issues  |      17 |
| Customer Support & General Inquiry      |      52 |
| **Total**                               | **200** |

The evaluation set is intentionally not class-balanced.

Delivery and general-support examples account for more than half of the golden set, so aggregate accuracy should be interpreted together with macro-F1 and per-class results.

---

# 4. Intent Taxonomy

The system uses 10 operational intents:

| ID | Intent                                  |
| -- | --------------------------------------- |
| 1  | Delivery Delay & Tracking               |
| 2  | Missing, Misdelivered & Lost Package    |
| 3  | Damaged, Defective & Incorrect Item     |
| 4  | Returns, Refunds & Replacements         |
| 5  | Billing, Payment & Unauthorized Charges |
| 6  | Pricing, Promotions & Cashback          |
| 7  | Account Access & Security               |
| 8  | Technical & Digital Services            |
| 9  | Order, Subscription & Pre-order Issues  |
| 10 | Customer Support & General Inquiry      |

The taxonomy was created by reviewing candidate categories discovered from a 2,000-example sample and consolidating them into a smaller operational taxonomy.

The goal was not to maximize the number of categories. The goal was to create categories useful for routing, retrieval, and escalation.

### Current taxonomy limitation

Positive feedback, thanks, and casual messages are currently forced into the operational taxonomy.

A dedicated non-issue category would likely improve classification quality.

---

# 5. Intent Classification

Several approaches were evaluated.

| Model                                 |  Accuracy |  Macro-F1 | Weighted-F1 |
| ------------------------------------- | --------: | --------: | ----------: |
| Majority baseline                     |     28.0% |         — |           — |
| TF-IDF + Logistic Regression          |     21.0% |     0.167 |       0.233 |
| Class mean prototype                  |     37.5% |     0.270 |       0.386 |
| Multi-prototype                       |     49.0% |     0.376 |       0.478 |
| **Hybrid multi-prototype + taxonomy** | **51.5%** | **0.425** |   **0.515** |

The majority baseline predicts:

```text
delivery_delay_tracking
```

for every example because it is the largest class.

The main comparisons use **5-fold stratified cross-validation** on the 200-example golden set.

## Final classifier

The final classifier uses multiple semantic prototypes for each intent rather than a single class centroid.

The hybrid score combines:

1. Similarity to multiple historical examples representing the intent.
2. Similarity to the written taxonomy description.

The hybrid approach performed best among the evaluated classifiers.

### Interpretation

The 51.5% accuracy is substantially above the 28.0% majority baseline, but it should not be interpreted as production-level intent accuracy.

The golden set is small, imbalanced, and contains ambiguous conversational messages.

Several intents also have very few examples, particularly:

* Billing
* Damaged/defective items
* Account/security

---

# 6. Historical Retrieval

The system uses:

```text
all-MiniLM-L6-v2
```

sentence embeddings to retrieve historical customer-support interactions.

Each interaction contains:

* Customer message
* Historical support response
* Conversation ID
* Customer tweet ID
* Agent tweet ID

## Leakage prevention

Removing only the exact golden examples would not be sufficient because other messages from the same conversation could remain in the retrieval corpus.

Therefore, the final evaluation excludes the **entire conversation** containing every golden-set example.

This resulted in:

```text
Total conversations:        55,733
Excluded conversations:        528
Leakage-safe interactions: 98,232
Embedding dimension:           384
```

## Leakage-safe retrieval diagnostic

| Metric                       |      Value |
| ---------------------------- | ---------: |
| Mean Top-1 cosine similarity | **0.7644** |
| P10                          |     0.6293 |
| P25                          |     0.7004 |
| Median                       |     0.7665 |
| P75                          |     0.8275 |
| P90                          |     0.9013 |

### Important interpretation

These values measure **semantic similarity**.

They do **not** measure:

* Retrieval correctness.
* Resolution correctness.
* Answer correctness.
* Customer satisfaction.

---

# 7. Response Generation

For cases classified as safe to automate, the system retrieves the top five historical customer-support interactions and provides them as evidence to Gemini.

The generation prompt explicitly prohibits:

* Inventing refunds or replacements.
* Inventing compensation.
* Inventing dates or guarantees.
* Inventing order information.
* Claiming that an action has already been performed.
* Introducing unsupported policies.
* Exposing historical URLs or identifiers.
* Making unsupported commitments.

Historical examples are sanitized before being passed to the generator.

## Sanitization

The sanitization layer removes:

* URLs
* Email addresses
* Phone numbers
* Amazon-style order numbers
* Twitter-style handles

Generated responses are also checked for sensitive-information leakage.

---

# 8. Escalation Policy

The escalation policy is intentionally conservative.

A case is escalated when:

* Historical retrieval evidence is weak.
* Intent classification is ambiguous.
* The intent is considered high-risk.
* There is insufficient evidence for safe automatic handling.

## High-risk intents

The following are treated conservatively:

* Billing, Payment & Unauthorized Charges
* Account Access & Security
* Missing, Misdelivered & Lost Package
* Returns, Refunds & Replacements

## Evaluation

The 200 golden examples contain:

```text
AUTO-HANDLE: 72
HUMAN:       128
```

Current policy results:

| Metric                |    Result |
| --------------------- | --------: |
| Accuracy              | **66.0%** |
| Predicted AUTO-HANDLE |  **9.0%** |
| Predicted HUMAN       | **91.0%** |
| False auto-handling   | **5.47%** |

Of the 128 cases labelled HUMAN, 7 were incorrectly predicted as AUTO-HANDLE.

The system therefore prioritizes avoiding unsafe automation over maximizing automation coverage.

### Important limitation

The escalation labels are human policy judgments rather than objective ground truth.

The current thresholds were also not selected using a completely independent validation set. Therefore, these results should be interpreted as an evaluation of the current policy rather than a fully calibrated estimate of production performance.

---

# 9. Generation Evaluation

A 20-example generation evaluation sample was created from examples manually labelled:

```text
AUTO-HANDLE
```

The generated responses were evaluated on:

* Helpfulness
* Groundedness
* Correctness
* Safety

Each dimension uses a 1–3 scale:

```text
1 = Poor
2 = Partially acceptable
3 = Good
```

## LLM-as-a-Judge results

The Gemini-based LLM judge evaluated all 20 generated responses.

| Dimension    |              Average |
| ------------ | -------------------: |
| Helpfulness  |         **2.65 / 3** |
| Groundedness |         **2.95 / 3** |
| Correctness  |         **2.70 / 3** |
| Safety       |         **2.95 / 3** |
| **Overall**  | **2.81 / 3 (93.8%)** |

The strongest dimensions were groundedness and safety.

The main weakness was helpfulness: several responses were appropriately cautious but too generic to directly resolve the customer's issue.

## Automated safety evaluation

All 20 generated responses passed the deterministic safety checks.

| Metric                     |             Result |
| -------------------------- | -----------------: |
| URL leakage                |             0 / 20 |
| Email leakage              |             0 / 20 |
| Phone leakage              |             0 / 20 |
| Order-number leakage       |             0 / 20 |
| Empty responses            |             0 / 20 |
| **Safety-clean responses** | **20 / 20 (100%)** |

---

# 10. LLM Judge Validation

The LLM judge was validated against an independent human evaluation.

The same 20 generated responses were rated separately by a human using the same four dimensions and the same 1–3 scale.

## Human vs. LLM judge

| Dimension    |    Exact Agreement | Cohen's κ |
| ------------ | -----------------: | --------: |
| Helpfulness  |  **95.0% (19/20)** | **0.894** |
| Groundedness | **100.0% (20/20)** | **1.000** |
| Correctness  |  **95.0% (19/20)** | **0.886** |
| Safety       | **100.0% (20/20)** | **1.000** |

The only disagreement was Sample 8.

The human evaluator rated:

```text
Helpfulness = 2
Correctness = 2
```

while the LLM judge rated:

```text
Helpfulness = 3
Correctness = 3
```

The response acknowledged a customer's suggestion to support Sodexo coupons but did not directly answer whether that payment method was supported.

This suggests that the judge can be somewhat more lenient toward polite acknowledgments when a response is not fully actionable.

The agreement results provide evidence that the LLM judge is reasonably aligned with the independent human ratings on this benchmark.

However, the sample contains only 20 examples and one human rater, so the agreement estimates should be treated as **directional rather than definitive**.

---

# 11. Failure Analysis

## Failure 1 — Generic responses

The generator sometimes falls back to a generic support handoff even when historical examples contain more specific resolution language.

Example pattern:

```text
Please reach out to our support team directly so we can look into this.
```

This minimizes hallucination risk but provides limited actionable information.

### Why it happens

The generation prompt strongly prioritizes avoiding unsupported claims.

When the retrieved evidence does not clearly support a specific action, the generator chooses a safe handoff.

### Improvement

Retrieve more specific historical resolutions and require the generator to identify an evidence-supported action before producing a response.

---

## Failure 2 — Intent confusion on conversational messages

The golden set contains ambiguous and conversational customer messages.

Positive feedback and thanks can be incorrectly classified into operational issue categories.

For example, a customer thanking AmazonHelp for a successful delivery can be classified as:

```text
delivery_delay_tracking
```

because the embedding representation focuses on the delivery topic even though the actual intent is positive feedback.

### Why it happens

The taxonomy currently lacks a dedicated representation for non-problem messages.

### Improvement

Add:

```text
positive_feedback_or_chitchat
```

or introduce a separate first-stage detector for problem vs. non-problem messages.

---

## Failure 3 — Conservative escalation

The current policy auto-handles only:

```text
9.0%
```

of the golden set.

This limits automation coverage.

However, false auto-handling is also relatively low:

```text
5.47%
```

### Why it happens

The current retrieval and intent-margin thresholds are intentionally conservative, particularly for high-risk intents.

### Improvement

Create a separate validation set and calibrate thresholds against an explicit risk/coverage objective.

Automation should first be expanded for low-risk intents with strong historical evidence.

---

## Failure 4 — Retrieval similarity is not correctness

A high cosine similarity does not guarantee that the retrieved historical resolution is appropriate for the current case.

Two customer messages can discuss the same topic while requiring different actions.

### Improvement

Experiment with:

* Intent-aware retrieval.
* Hybrid lexical + dense retrieval.
* Reranking.
* Diversity-aware top-k selection.
* Answer-level retrieval evaluation.

---

## Failure 5 — Insufficient conversational context

Very short messages can cause the generator to assume context that is not actually available.

Example:

```text
@AmazonHelp Yes
```

The generated response said:

```text
Thank you for confirming. We'll get back to you with a response shortly.
```

This is polite, but the available message does not establish that a follow-up is actually pending.

The LLM judge rated this example:

```text
Helpfulness:   2
Groundedness:  2
Correctness:   2
Safety:        2
```

### Why it happens

The generation stage primarily operates on the incoming customer message and retrieved evidence.

Messages such as:

```text
Yes
Thanks
Okay
Still waiting
```

can depend heavily on preceding conversation context.

### Improvement

Detect extremely short or context-dependent messages and either:

1. Retrieve preceding conversation context, or
2. Escalate instead of generating a potentially unsupported follow-up promise.

---

# 12. What Is Misleading About My Headline Number?

The most potentially misleading standalone number is:

```text
Mean Top-1 retrieval similarity = 0.7644
```

Without context, this could easily be interpreted as:

```text
76.44% retrieval accuracy
```

That interpretation is incorrect.

The value is a cosine-similarity diagnostic measuring how semantically close the retrieved historical example is to the customer query.

It does not measure:

* Whether the retrieved example contains the correct resolution.
* Whether the generated response is correct.
* Whether the customer's issue would actually be resolved.

The entire conversation containing each golden example was excluded from the retrieval corpus to reduce leakage.

Therefore, the honest interpretation is:

> The system generally finds semantically similar historical interactions, with a mean Top-1 cosine similarity of 0.7644 on the 200-example golden set, but this number does not establish retrieval correctness.

### Another potentially misleading number

The generation headline:

```text
93.8% overall
```

should not be interpreted as:

```text
93.8% of customers receive a correct answer.
```

It is the normalized average of four qualitative dimensions evaluated on only 20 generated examples.

---

# 13. Key Results

| Component  | Metric                 |             Result |
| ---------- | ---------------------- | -----------------: |
| Intent     | Majority accuracy      |              28.0% |
| Intent     | Hybrid accuracy        |          **51.5%** |
| Intent     | Hybrid Macro-F1        |          **0.425** |
| Intent     | Hybrid Weighted-F1     |          **0.515** |
| Retrieval  | Mean Top-1 similarity  |         **0.7644** |
| Escalation | Auto coverage          |           **9.0%** |
| Escalation | False auto-handling    |          **5.47%** |
| Escalation | Policy accuracy        |          **66.0%** |
| Generation | Helpfulness            |       **2.65 / 3** |
| Generation | Groundedness           |       **2.95 / 3** |
| Generation | Correctness            |       **2.70 / 3** |
| Generation | Safety                 |       **2.95 / 3** |
| Generation | Overall                |       **2.81 / 3** |
| Generation | Safety-clean           |   **100% (20/20)** |
| Judge      | Helpfulness agreement  | **95.0%, κ=0.894** |
| Judge      | Groundedness agreement |  **100%, κ=1.000** |
| Judge      | Correctness agreement  | **95.0%, κ=0.886** |
| Judge      | Safety agreement       |  **100%, κ=1.000** |

> **Important:** These metrics come from different evaluation setups and should not be interpreted as a single overall system accuracy.

---

# 14. Reproducibility

The repository contains scripts for:

* Dataset analysis
* Conversation construction
* Golden-set creation
* Intent discovery
* Intent evaluation
* Retrieval-index construction
* Leakage-safe retrieval evaluation
* Escalation scoring
* Escalation evaluation
* Response generation
* Generation safety evaluation
* LLM-as-a-judge evaluation
* Independent human evaluation
* Human-vs-LLM agreement analysis

## Environment

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project expects a Gemini API key in:

```text
.env
```

with:

```text
GEMINI_API_KEY=your_key_here
```

Do not commit `.env`.

---

# 15. Running the Agent

After the required processed artifacts are available:

```bash
python src/agent.py "The item I received is broken."
```

The agent returns:

* Predicted intent
* Intent confidence
* Intent margin
* Retrieved historical examples
* Escalation decision
* Escalation reason
* Generated response when auto-handling is allowed

---

# 16. Evaluation Commands

## Intent baseline

```bash
python evaluation/baseline_majority.py
```

## TF-IDF baseline

```bash
python evaluation/baseline_tfidf.py
```

## Intent evaluation

```bash
python evaluation/evaluate_intent.py
```

## Leakage-safe retrieval index

```bash
python src/retrieval/build_leakage_safe_index.py
```

## Retrieval evaluation

```bash
python evaluation/evaluate_retrieval.py
```

## Escalation evaluation

```bash
python evaluation/evaluate_escalation.py
```

## Generation evaluation

```bash
python evaluation/run_generation_eval.py
python evaluation/evaluate_generation.py
```

## LLM-as-a-Judge

```bash
python evaluation/llm_judge.py
python evaluation/summarize_llm_judge.py
```

The first command runs the Gemini-based judge on the generated evaluation sample.

The second command summarizes the resulting judge scores.

## Independent human evaluation

```bash
python evaluation/collect_human_ratings.py
```

This collects independent human ratings for the generation evaluation sample.

## Human-vs-LLM agreement

```bash
python evaluation/compare_human_llm.py
```

This computes:

* Exact agreement.
* Cohen's kappa.
* Per-dimension disagreements.

---

# 17. Project Structure

```text
hiver-ai-support-agent/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
│
├── evaluation/
│   ├── baseline_majority.py
│   ├── baseline_tfidf.py
│   ├── compare_human_llm.py
│   ├── collect_human_ratings.py
│   ├── create_generation_sample.py
│   ├── evaluate_escalation.py
│   ├── evaluate_generation.py
│   ├── evaluate_retrieval.py
│   ├── human_generation_ratings_independent.json
│   ├── llm_judge.py
│   ├── run_generation_eval.py
│   └── summarize_llm_judge.py
│
├── notebooks/
│
├── src/
│   ├── agent.py
│   │
│   ├── data/
│   │   ├── analyze_amazon.py
│   │   ├── build_conversations.py
│   │   ├── build_interactions.py
│   │   ├── create_intent_sample.py
│   │   └── prepare_training_data.py
│   │
│   ├── intent/
│   │   ├── taxonomy.json
│   │   ├── discover.py
│   │   ├── create_golden_set.py
│   │   ├── label_golden.py
│   │   ├── multi_prototype.py
│   │   └── hybrid.py
│   │
│   ├── retrieval/
│   │   └── build_leakage_safe_index.py
│   │
│   ├── generation/
│   │   └── generate_reply.py
│   │
│   └── escalation/
│       ├── decide.py
│       └── label_escalation.py
│
├── tests/
│
├── DECISIONS.md
├── REPORT.md
├── README.md
├── requirements.txt
├── .env
└── .gitignore
```

---

# 18. Limitations

1. The golden set contains 200 examples.
2. Intent classes are imbalanced, with some intents having relatively few labelled examples.
3. Some conversational or positive-feedback messages do not fit naturally into the current operational taxonomy.
4. Escalation labels are human policy judgments rather than objective ground truth.
5. Escalation thresholds were not selected using a completely independent validation set.
6. Retrieval similarity is a diagnostic and not a measure of resolution correctness.
7. Generation quality was evaluated on only 20 examples.
8. Human-vs-LLM judge agreement was measured on only 20 examples.
9. The human-vs-LLM comparison used one independent human rater.
10. Intent confidence values are ranking/ambiguity signals and are not calibrated probabilities.
11. The system has no access to real-time customer account or order information.
12. The system cannot execute refunds, replacements, cancellations, or other transactions.
13. Gemini free-tier API quotas can limit repeated live generation runs.
14. Historical Twitter support behavior may not represent current Amazon policies or production support procedures.

---

# 19. Future Work — One More Week

## 1. Expand and rebalance the golden set

Increase the evaluation set beyond 200 examples.

Add deliberate coverage of minority intents such as:

* Billing
* Damaged/defective products
* Account/security
* Returns/refunds

Also add more difficult ambiguous and multi-intent cases.

## 2. Add an explicit non-issue category

Introduce:

```text
positive_feedback_or_chitchat
```

This would prevent positive feedback and casual messages from being forced into operational support categories.

## 3. Calibrate escalation

Create a separate validation set for threshold selection.

Optimize the explicit trade-off:

```text
Automation coverage
        vs.
False auto-handling
```

Automation should first be expanded for low-risk intents with strong historical evidence.

## 4. Improve retrieval

Experiment with:

* Intent-aware retrieval.
* Hybrid lexical + dense retrieval.
* Cross-encoder reranking.
* Diversity-aware top-k retrieval.
* Intent-filtered retrieval.

The objective should be answer-relevant evidence rather than semantic similarity alone.

## 5. Improve response specificity

Require generated responses to identify an evidence-supported action before responding.

If no sufficiently specific action is supported, prefer escalation over a generic response.

## 6. Add conversation context

Retrieve relevant preceding conversation messages for short or context-dependent customer replies.

This would reduce failures caused by messages such as:

```text
Yes
Thanks
Okay
Still waiting
```

## 7. Strengthen LLM-judge validation

Collect ratings from multiple independent human raters on a larger sample.

Compare them against the automated judge to obtain a more reliable estimate of judge agreement.

---

# 20. Engineering Decisions

Major non-obvious design decisions are documented separately in:

```text
DECISIONS.md
```

The decision log covers:

* Target-brand selection.
* Golden-set construction.
* Taxonomy design.
* Prototype-based classification.
* Hybrid intent scoring.
* Conversation-level retrieval leakage prevention.
* Conservative escalation.
* High-risk intent handling.
* Retrieval-grounded generation.
* Historical-data sanitization.
* Deterministic safety checks.
* LLM-judge validation.
* Evaluation methodology.
* Automation coverage vs. safety trade-offs.

---

# 21. Security and Privacy

The repository intentionally excludes:

```text
.env
data/raw/
data/processed/
```

from Git where appropriate.

The Gemini API key is stored locally in `.env` and must never be committed.

Historical examples are sanitized before generation to prevent propagation of:

* URLs
* Email addresses
* Phone numbers
* Order numbers
* Social-media handles

The system does not have access to real customer account credentials or transaction systems.

---

# 22. Summary

This project implements a conservative retrieval-grounded AI customer-support agent for AmazonHelp.

The strongest evaluation findings are:

* The hybrid multi-prototype intent classifier achieved **51.5% accuracy**, compared with **28.0% for the majority baseline**.
* Conversation-level retrieval leakage was explicitly addressed.
* The leakage-safe retrieval corpus contains **98,232 historical interactions**.
* Mean Top-1 retrieval similarity was **0.7644**, reported explicitly as a similarity diagnostic rather than accuracy.
* The escalation policy predicts AUTO-HANDLE for **9.0%** of the golden set.
* False auto-handling is **5.47%** under the current conservative policy.
* Generated responses achieved **100% deterministic safety-clean rate** on the 20-example generation benchmark.
* LLM-judge generation quality averaged **2.81/3 (93.8%)**.
* Groundedness and safety were the strongest generation dimensions.
* Helpfulness was weaker because several responses were cautious but generic.
* Independent human-vs-LLM validation showed high agreement across all four dimensions.

The central design trade-off is:

> **The system is optimized for safe, evidence-backed support rather than maximum automation coverage.**

The highest-value next steps are improving the intent taxonomy, calibrating escalation on an independent validation set, strengthening retrieval/reranking, adding conversational context, and expanding the human-validated generation benchmark.
