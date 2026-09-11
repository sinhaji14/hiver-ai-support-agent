
````markdown
# Hiver AI Support Agent

> An AI customer-support agent for AmazonHelp that classifies customer issues, retrieves relevant historical resolutions, generates grounded responses, and decides when a human should take over.

---

## Overview

This project builds an end-to-end AI support agent using the **Kaggle Customer Support on Twitter** dataset.

For every incoming customer message, the system performs four stages:

                    Customer Message
                           │
                           ▼
              ┌──────────────────────┐
              │  Intent Classifier   │
              │  Hybrid Embeddings   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Historical Retrieval │
              │     Top-5 Cases      │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Escalation Policy    │
              └──────────┬───────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
             HUMAN           AUTO-HANDLE
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Gemini Response │
                         │   Generation    │
                         └────────┬────────┘
                                  │
                                  ▼
                         Grounded Response
````

### Design priorities

The system prioritizes:

* **Grounded responses** over creative generation
* **Safety** over maximum automation coverage
* **Historical support evidence** over invented policies
* **Conversation-level leakage prevention** during evaluation

---

# 1. Dataset

The project uses the **Customer Support on Twitter** dataset from Kaggle.

### Target brand

**AmazonHelp**

AmazonHelp was selected after comparing major support accounts in the dataset.

| Brand        | Customer Messages | Support Tweets | Unique Customers |
| ------------ | ----------------: | -------------: | ---------------: |
| AmazonHelp   |           135,160 |    **169,840** |           48,134 |
| AppleSupport |            97,896 |        106,860 |           58,578 |
| Uber_Support |            46,624 |         56,270 |           25,012 |
| SpotifyCares |            31,353 |         43,265 |           18,079 |
| Delta        |            44,838 |         42,253 |           23,782 |

AmazonHelp provided the largest support volume and enough interaction
diversity for intent discovery, retrieval, generation, and evaluation.

### AmazonHelp dataset statistics

| Statistic                           |   Value |
| ----------------------------------- | ------: |
| AmazonHelp support tweets           | 169,840 |
| Support tweets with parent IDs      | 169,287 |
| Matched customer messages           | 154,976 |
| Clean customer-support interactions | 168,814 |
| Usable conversations                |  55,733 |
| Leakage-safe retrieval interactions |  98,232 |

The raw Kaggle dataset is intentionally excluded from version control because
of its size.

---

# 2. Intent Taxonomy

A 10-intent operational taxonomy was created from the historical support
interactions.

| ID | Intent                             | Description                                                             |
| -- | ---------------------------------- | ----------------------------------------------------------------------- |
| 1  | `delivery_delay_tracking`          | Delayed shipments, missed delivery dates, and tracking issues           |
| 2  | `missing_misdelivered_package`     | Missing, lost, misdelivered, or incorrectly marked-delivered packages   |
| 3  | `damaged_defective_incorrect_item` | Damaged, defective, broken, counterfeit, or incorrect products          |
| 4  | `returns_refunds_replacements`     | Returns, refunds, replacements, and return-pickup issues                |
| 5  | `billing_payment_charges`          | Payment failures, unexpected charges, overcharging, and billing issues  |
| 6  | `pricing_promotions_cashback`      | Prices, discounts, promotions, credits, and cashback                    |
| 7  | `account_access_security`          | Login, account access, verification, fraud, and security issues         |
| 8  | `technical_digital_services`       | Website, app, Fire TV, Prime Video, and technical issues                |
| 9  | `order_subscription_preorder`      | Orders, cancellations, subscriptions, pre-orders, and stock issues      |
| 10 | `customer_support_general`         | General support questions, unresolved support, and communication issues |

A **200-example hand-labelled golden set** was created for evaluation.

---

# 3. Intent Classification

The final classifier uses:

* `all-MiniLM-L6-v2` sentence embeddings
* Multiple semantic prototypes per intent
* Taxonomy-description similarity

The hybrid classifier combines historical customer-language patterns with
the semantic definition of each intent.

## Baselines

| Model                        |  Accuracy |  Macro-F1 | Weighted-F1 |
| ---------------------------- | --------: | --------: | ----------: |
| Majority baseline            |     28.0% |         — |           — |
| TF-IDF + Logistic Regression |     21.0% |     0.167 |       0.233 |
| Class prototype              |     37.5% |     0.270 |       0.386 |
| Multi-prototype              |     49.0% |     0.376 |       0.478 |
| **Hybrid classifier**        | **51.5%** | **0.425** |   **0.515** |

The main classifier comparison uses **5-fold stratified cross-validation** on
the 200-example golden set.

### Final classifier

```text
Customer message
       │
       ├── Historical intent prototypes
       │
       └── Taxonomy descriptions
                  │
                  ▼
           Hybrid similarity
                  │
                  ▼
            Predicted intent
```

The hybrid classifier was the strongest evaluated intent approach.

---

# 4. Historical Retrieval

Historical customer-support interactions are embedded using:

```text
all-MiniLM-L6-v2
```

For a new customer message, the system retrieves the **top 5 semantically
similar historical customer-support interactions**.

Each historical interaction contains:

```text
Customer message
        +
Historical AmazonHelp response
```

This provides the generator with examples of how similar issues were
previously handled.

## Leakage prevention

The evaluation uses a strict conversation-level exclusion strategy.

For every golden-set example:

> The entire conversation containing that example is removed from the
> retrieval corpus.

This is stricter than removing only the exact golden message because another
turn from the same conversation could otherwise leak information into the
evaluation.

### Leakage-safe retrieval corpus

| Metric                       |      Value |
| ---------------------------- | ---------: |
| Historical interactions      |     98,232 |
| Embedding dimension          |        384 |
| Mean Top-1 cosine similarity | **0.7644** |
| P10                          |     0.6293 |
| P25                          |     0.7004 |
| Median                       | **0.7665** |
| P75                          |     0.8275 |
| P90                          |     0.9013 |

### Important interpretation

The **0.7644 Top-1 similarity is not retrieval accuracy**.

It is a semantic-similarity diagnostic indicating how close the retrieved
historical customer message is to the evaluation query.

It does **not** establish that the retrieved historical resolution is
correct.

---

# 5. Response Generation

When the escalation policy determines that a case is safe to automate,
the system passes the customer message and retrieved historical examples to
Gemini.

The generator is explicitly instructed to remain grounded in the available
historical evidence.

## Generation constraints

The generator must not:

* Invent refunds
* Invent replacements
* Invent compensation
* Invent dates or delivery guarantees
* Invent order information
* Invent unsupported policies
* Claim that an action has already been completed
* Expose historical customer identifiers
* Introduce unsupported URLs or contact information

## Historical-data sanitization

Historical examples are sanitized before being included in the generation
prompt.

The sanitization layer removes:

* URLs
* Email addresses
* Phone numbers
* Amazon-style order numbers
* Twitter-style usernames

Generated responses are also checked for sensitive-information leakage.

---

# 6. Escalation Policy

The system does not automatically answer every customer.

It uses a conservative escalation policy based on:

1. Historical retrieval strength
2. Intent ambiguity
3. Intent risk
4. Availability of sufficient historical precedent

### High-risk intents

The following intents are treated conservatively:

* `billing_payment_charges`
* `account_access_security`
* `missing_misdelivered_package`
* `returns_refunds_replacements`

These cases may require account-specific investigation or actions that cannot
safely be inferred from historical text alone.

## Human-labelled escalation set

The 200-example golden set contains:

| Human policy label | Examples |
| ------------------ | -------: |
| AUTO-HANDLE        |       72 |
| HUMAN              |      128 |

### Current escalation results

| Metric                   |    Result |
| ------------------------ | --------: |
| Accuracy                 | **66.0%** |
| AUTO-HANDLE coverage     |  **9.0%** |
| HUMAN escalation         | **91.0%** |
| False auto-handling rate | **5.47%** |

The current policy intentionally sacrifices automation coverage to reduce the
risk of incorrectly auto-handling cases that were labelled for human review.

### Interpretation

For a customer-support system, false auto-handling is a more important safety
signal than raw automation coverage.

The escalation labels represent **human policy judgments**, rather than an
objective ground-truth property of the customer cases.

---

# 7. Generation Evaluation

Generation evaluation was performed on a small sample of cases labelled
`AUTO-HANDLE`.

Due to the Gemini free-tier API quota, **6 generated responses were
successfully evaluated in the current run**.

This sample is reported transparently and is not presented as a large-scale
generation benchmark.

## Automated safety checks

| Check                  |           Result |
| ---------------------- | ---------------: |
| URL leakage            |            0 / 6 |
| Email leakage          |            0 / 6 |
| Phone leakage          |            0 / 6 |
| Order-number leakage   |            0 / 6 |
| Empty responses        |            0 / 6 |
| Safety-clean responses | **6 / 6 (100%)** |

## Reference evaluation

Responses were assessed using a 1–3 rubric:

* **Helpfulness:** 1 = not helpful, 3 = helpful
* **Groundedness:** 1 = unsupported, 3 = fully grounded
* **Safety:** 1 = unsafe, 3 = safe

| Dimension    |      Average |
| ------------ | -----------: |
| Helpfulness  | **2.50 / 3** |
| Groundedness | **3.00 / 3** |
| Safety       | **3.00 / 3** |
| Overall      | **2.83 / 3** |

The main observed weakness is **response specificity**.

Several responses are safe and grounded but too generic, for example:

```text
Please reach out to our support team directly so we can look into this.
```

This is safer than hallucinating a resolution, but it provides limited
actionable information.

An LLM-judge prompt has been prepared in:

```text
data/processed/llm_judge_prompt.txt
```

LLM-judge scores and human-agreement statistics are intentionally **not
reported until an actual LLM-judge run is completed**.

---

# 8. Failure Analysis

## Failure 1 — Generic responses

The generator sometimes falls back to a generic support handoff even when
historical examples contain more specific resolution language.

### Why it happens

The generation prompt prioritizes avoiding unsupported claims.

### Trade-off

```text
More conservative
       ↓
Lower hallucination risk
       ↓
Potentially less helpful response
```

### Next improvement

Use intent-aware retrieval and require the generator to identify a concrete
supported action from the historical evidence before producing a resolution.

---

## Failure 2 — Intent confusion

Conversational and positive-feedback messages can be difficult to map into
the operational taxonomy.

One evaluated example thanked AmazonHelp for a successful delivery.

Human label:

```text
customer_support_general
```

Predicted label:

```text
delivery_delay_tracking
```

The classifier correctly recognized the delivery topic but missed the fact
that the customer was providing positive feedback rather than reporting a
delivery problem.

### Next improvement

Introduce a dedicated:

```text
positive_feedback_or_chitchat
```

category or add a separate non-issue detection stage before operational
intent classification.

---

## Failure 3 — Conservative automation

The current escalation policy auto-handles only **9%** of the golden set.

This is deliberately conservative.

The benefit is a much lower false auto-handling rate:

```text
5.47%
```

### Next improvement

Use a separate validation set to calibrate escalation thresholds and gradually
increase automation coverage for low-risk intents.

---

## Failure 4 — Retrieval similarity is not correctness

A high semantic similarity score does not guarantee that the historical
response is the correct resolution for the current customer.

Retrieval therefore needs to be evaluated separately from response quality.

---

# 9. Misleading Headline Number

The most potentially misleading standalone number is:

```text
Mean Top-1 retrieval similarity = 0.7644
```

It would be incorrect to interpret this as:

```text
76.44% retrieval accuracy
```

The number is simply a cosine-similarity diagnostic.

The evaluation deliberately reports it as a **retrieval-quality diagnostic**
rather than a correctness metric.

---

# 10. Reproduction

## Requirements

* Python 3.12+
* Gemini API key for response generation
* Internet access for downloading the embedding model
* Kaggle Customer Support on Twitter dataset

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/hiver-ai-support-agent.git
cd hiver-ai-support-agent
```

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Gemini

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from version control.

---

# 11. Run the Agent

From the project root:

```bash
python src/agent.py "The item I received is broken."
```

Example:

```text
============================================================
HIVER AI SUPPORT AGENT
============================================================

Intent: damaged_defective_incorrect_item
Intent confidence: 0.268
Intent similarity: 0.481
Intent margin: 0.055

Top historical matches:
...

Decision: AUTO-HANDLE
Reason: Strong historical match ...

Generated customer response:
That's not what we like to hear! We'd like to help.
Please reach out to our support team to start the return process.
```

The terminal output sanitizes historical customer identifiers before
displaying retrieved examples.

---

# 12. Evaluation Commands

## Intent baseline

```bash
python evaluation/baseline_majority.py
```

```bash
python evaluation/baseline_tfidf.py
```

## Intent evaluation

```bash
python evaluation/evaluate_supervised_intent.py
```

## Retrieval evaluation

```bash
python evaluation/evaluate_retrieval_safe.py
```

## Escalation evaluation

```bash
python evaluation/score_escalation.py
```

```bash
python evaluation/evaluate_escalation.py
```

## Generation safety evaluation

```bash
python evaluation/evaluate_generation.py
```

## Human/reference generation evaluation

```bash
python evaluation/summarize_human_ratings.py
```

## LLM-judge preparation

```bash
python evaluation/llm_judge.py
```

The LLM-judge preparation script creates the batched judge prompt without
requiring an API call.

---

# 13. Project Structure

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
│   ├── evaluate_generation.py
│   ├── evaluate_intent.py
│   ├── evaluate_retrieval_safe.py
│   ├── evaluate_escalation.py
│   ├── evaluate_supervised_intent.py
│   ├── score_escalation.py
│   ├── run_generation_eval.py
│   ├── create_generation_sample.py
│   ├── summarize_human_ratings.py
│   ├── llm_judge.py
│   └── human_generation_ratings.json
│
├── src/
│   ├── agent.py
│   ├── data/
│   ├── intent/
│   ├── retrieval/
│   ├── generation/
│   └── escalation/
│
├── tests/
│
├── DECISIONS.md
├── REPORT.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 14. Key Results at a Glance

| Component  | Metric                 |         Result |
| ---------- | ---------------------- | -------------: |
| Intent     | Accuracy               |      **51.5%** |
| Intent     | Macro-F1               |      **0.425** |
| Intent     | Weighted-F1            |      **0.515** |
| Retrieval  | Mean Top-1 similarity  |     **0.7644** |
| Escalation | Accuracy               |      **66.0%** |
| Escalation | AUTO-HANDLE coverage   |       **9.0%** |
| Escalation | False auto-handling    |      **5.47%** |
| Generation | Safety-clean responses | **100% (6/6)** |
| Generation | Helpfulness            |     **2.50/3** |
| Generation | Groundedness           |     **3.00/3** |
| Generation | Safety                 |     **3.00/3** |

> **Important:** These metrics come from different evaluation setups and
> should not be interpreted as a single overall system accuracy.

---

# 15. Limitations

1. The golden set contains 200 examples.
2. Intent classes are imbalanced, with some intents having relatively few
   labelled examples.
3. Some conversational or positive-feedback messages do not fit naturally
   into the current operational taxonomy.
4. Escalation labels are human policy judgments rather than objective
   ground truth.
5. Generation evaluation currently contains only six successfully generated
   examples because of Gemini free-tier quota limits.
6. The LLM-judge agreement analysis has not yet been completed.
7. Intent confidence values are used as ranking/ambiguity signals and are not
   calibrated probabilities.
8. Retrieval similarity is a diagnostic and not a measure of resolution
   correctness.

---

# 16. Future Work

### 1. Improve intent taxonomy

Add explicit handling for:

* Positive feedback
* Chitchat
* General appreciation
* Non-problem messages

### 2. Improve retrieval

Experiment with:

* Intent-aware retrieval
* Cross-encoder reranking
* Diversity-aware top-k retrieval
* Intent-filtered retrieval

### 3. Improve escalation calibration

Create a separate validation set for threshold selection and optimize the
trade-off between:

```text
Automation coverage
        vs.
False auto-handling
```

### 4. Improve response specificity

Require generated responses to be supported by an identifiable historical
resolution pattern.

### 5. Expand generation evaluation

Evaluate a larger, stratified generation sample once sufficient API quota is
available.

### 6. Complete LLM-judge agreement

Run the prepared LLM judge once sufficient API quota is available, then
compare its ratings against independently collected human ratings using
agreement metrics such as:

* Percent agreement
* Cohen's kappa

Human–LLM agreement is **not reported in this submission** because independent
human ratings have not yet been collected.

---

# 17. Engineering Decisions

Major design decisions are documented separately in:

```text
DECISIONS.md
```

The decision log covers:

* AmazonHelp selection
* Intent taxonomy design
* Golden-set construction
* Annotation policy
* Cross-validation
* Embedding model
* Multi-prototype classification
* Leakage prevention
* Top-k retrieval
* Generation grounding
* PII sanitization
* Escalation policy
* Evaluation methodology

---

# 18. Security and Data Handling

The Gemini API key is loaded from:

```text
.env
```

and `.env` is excluded from version control.

Historical support examples are sanitized before being exposed to the
generation model.

The system removes obvious:

* Order numbers
* URLs
* Email addresses
* Phone numbers
* Twitter handles

from historical examples before generation.

The raw dataset is also excluded from version control because of its size.

---

# 19. Summary

This project implements a complete support-agent pipeline:

```text
Historical Support Data
          │
          ▼
     Intent Discovery
          │
          ▼
    Golden Evaluation Set
          │
          ▼
 Hybrid Intent Classifier
          │
          ▼
 Leakage-Safe Retrieval
          │
          ▼
  Conservative Escalation
       │          │
       │          └──────────────┐
       ▼                         ▼
     HUMAN                 AUTO-HANDLE
                                 │
                                 ▼
                           Gemini Generation
                                 │
                                 ▼
                         Grounded Response
```

The current system demonstrates that historical support interactions can be
used to build a grounded customer-support agent while explicitly controlling
automation risk and preventing obvious sensitive-information leakage.

The main remaining improvements are better handling of conversational
messages, more specific responses, calibrated escalation, and a larger
generation evaluation set.

````
