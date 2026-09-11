# Hiver AI Support Agent — Evaluation Report

## 1. Problem Framing

This project builds an AI customer-support agent for AmazonHelp using
historical customer-support interactions from the Kaggle Customer Support
on Twitter dataset.

For each incoming customer message, the system performs three tasks:

1. Classifies the customer's primary support intent.
2. Retrieves semantically similar historical customer-support interactions.
3. Decides whether the case can be automatically handled or should be
   escalated to a human.

For cases that are safe to automate, the system generates a concise response
grounded in historical support resolutions.

The design prioritizes safety over maximum automation coverage. In
particular, financial, account-security, missing-package, and return/refund
cases are treated conservatively.

---

## 2. Dataset and Target Brand

The Kaggle dataset contains customer and brand-support tweets.

AmazonHelp was selected because it had the largest support volume among the
candidate brands examined.

Key dataset statistics:

- AmazonHelp support tweets: 169,840
- Support tweets with parent IDs: 169,287
- Matched customer parent tweets: 154,976
- Clean customer-support interactions: 168,814
- Usable conversations: 55,733

The final retrieval corpus contains 98,232 historical interactions after
removing entire conversations containing golden-set examples.

This conversation-level exclusion is used to reduce retrieval leakage.

---

## 3. Intent Taxonomy

The system uses 10 operational intents:

1. Delivery Delay & Tracking
2. Missing, Misdelivered & Lost Package
3. Damaged, Defective & Incorrect Item
4. Returns, Refunds & Replacements
5. Billing, Payment & Unauthorized Charges
6. Pricing, Promotions & Cashback
7. Account Access & Security
8. Technical & Digital Services
9. Order, Subscription & Pre-order Issues
10. Customer Support & General Inquiry

A 200-example golden set was manually labelled.

The annotation rule was to assign the customer's primary support need,
based on the issue most directly addressed by the customer request and
historical support response.

---

## 4. Intent Classification

### Baselines

Several approaches were evaluated:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Majority baseline | 28.0% | — | — |
| TF-IDF + Logistic Regression | 21.0% | 0.167 | 0.233 |
| Class prototype | 37.5% | 0.270 | 0.386 |
| Multi-prototype | 49.0% | 0.376 | 0.478 |
| Hybrid multi-prototype + taxonomy | **51.5%** | **0.425** | **0.515** |

The main supervised comparisons use 5-fold stratified cross-validation on
the 200-example golden set.

The final hybrid model combines semantic similarity to multiple historical
prototypes for each intent with similarity to the taxonomy description.

The hybrid approach performed best among the evaluated classifiers.

---

## 5. Historical Retrieval

The system uses `all-MiniLM-L6-v2` embeddings to retrieve historical
customer-support interactions.

To reduce evaluation leakage, any entire conversation containing a golden
example is excluded from the retrieval corpus.

The resulting retrieval corpus contains:

- 98,232 historical interactions
- 384-dimensional embeddings

### Leakage-safe retrieval diagnostic

| Metric | Value |
|---|---:|
| Mean Top-1 cosine similarity | **0.7644** |
| P10 | 0.6293 |
| P25 | 0.7004 |
| Median | 0.7665 |
| P75 | 0.8275 |
| P90 | 0.9013 |

These numbers measure semantic similarity, not retrieval correctness or
answer accuracy.

---

## 6. Response Generation

For cases classified as safe to automate, the system retrieves the top five
historical interactions and provides them as evidence to Gemini.

The generation prompt explicitly prohibits:

- Inventing refunds or replacements
- Inventing compensation
- Inventing dates or guarantees
- Inventing order information
- Claiming that an action has already been performed
- Introducing unsupported policies
- Exposing historical URLs or identifiers

Historical examples are sanitized before being passed to the generator.

The sanitization layer removes:

- URLs
- Email addresses
- Phone numbers
- Amazon-style order numbers
- Twitter-style handles

Generated responses are also checked for sensitive information.

---

## 7. Escalation Policy

The escalation policy is intentionally conservative.

A case is escalated when:

- Historical retrieval evidence is weak.
- Intent classification is ambiguous.
- The intent is considered high risk.
- Evidence is insufficient for safe automatic handling.

High-risk intents include:

- Billing, Payment & Unauthorized Charges
- Account Access & Security
- Missing, Misdelivered & Lost Package
- Returns, Refunds & Replacements

### Evaluation

The 200 golden examples include human escalation-policy labels:

- AUTO-HANDLE: 72
- HUMAN: 128

Current policy results:

| Metric | Result |
|---|---:|
| Accuracy | **66.0%** |
| AUTO-HANDLE coverage | **9.0%** |
| HUMAN escalation | **91.0%** |
| False auto-handling | **5.47%** |

The system therefore prioritizes avoiding unsafe automation over maximizing
automatic handling coverage.

The escalation labels represent human policy judgments rather than an
objective ground-truth property of the cases.

---

## 8. Generation Evaluation

A small generation evaluation sample was created from human-labelled
AUTO-HANDLE examples.

Due to the Gemini free-tier API request quota, six examples were successfully
generated and evaluated in the current run.

### Automated safety evaluation

| Metric | Result |
|---|---:|
| URL leakage | 0 / 6 |
| Email leakage | 0 / 6 |
| Phone leakage | 0 / 6 |
| Order-number leakage | 0 / 6 |
| Empty responses | 0 / 6 |
| Safety-clean responses | **6 / 6 (100%)** |

### Reference evaluation

Responses were evaluated on a 1–3 reference scale.

| Dimension | Average |
|---|---:|
| Helpfulness | **2.50 / 3** |
| Groundedness | **3.00 / 3** |
| Safety | **3.00 / 3** |
| Overall | **2.83 / 3** |

The main observed weakness was helpfulness: some generated responses were
safe and grounded but too generic.

These scores are reference ratings for the six generated examples, not
independently collected human ratings. Human–LLM agreement is therefore not
reported.

An LLM-judge prompt has been prepared to evaluate the same dimensions.
LLM-judge results and human-agreement statistics are not reported until the
judge has actually been run.

---

## 9. Failure Analysis

### Failure 1 — Generic responses

Several responses use a safe but generic support handoff such as asking the
customer to contact support.

This minimizes hallucination risk but can reduce helpfulness.

**Improvement:** Retrieve more specific historical resolutions and condition
generation on the actual action supported by those examples.

---

### Failure 2 — Intent confusion

The golden set contains ambiguous and conversational customer messages.

For example, one generation-sample case thanked AmazonHelp for a successful
delivery. The model predicted `delivery_delay_tracking`, while the human
label was `customer_support_general`.

This demonstrates that semantic similarity can overemphasize the topic of
delivery even when the customer's actual intent is appreciation or general
support feedback.

**Improvement:** Add a dedicated positive-feedback/chitchat category or
introduce a separate intent-detection stage for non-problem messages.

---

### Failure 3 — Conservative escalation

The current policy only auto-handles 9% of the golden examples.

This limits automation coverage but keeps false auto-handling relatively
low at 5.47%.

**Improvement:** Calibrate escalation thresholds using a separate validation
set and gradually expand automation coverage for low-risk intents.

---

### Failure 4 — Retrieval similarity is not correctness

A high cosine similarity does not guarantee that the retrieved historical
resolution is appropriate for the current case.

The retrieval metric should therefore not be interpreted as answer accuracy.

---

## 10. Misleading Headline Number

A potentially misleading number is:

> **0.7644 mean Top-1 retrieval similarity**

Without context, this could be interpreted as "76.44% retrieval accuracy."

That interpretation is incorrect.

The value is a cosine-similarity diagnostic measuring how semantically close
the retrieved historical example is to the query. It does not measure whether
the retrieved example contains the correct resolution.

The entire conversation containing each golden example was excluded from the
retrieval corpus to reduce leakage.

---

## 11. What I Would Improve Next Week

### 1. Expand the golden set

Increase the evaluation set beyond 200 examples, especially for minority
intents such as billing and damaged/defective items.

### 2. Add an explicit non-issue category

Positive feedback, thanks, and casual messages are currently forced into the
10 operational intents.

Adding a `positive_feedback_or_chitchat` category would reduce this source
of confusion.

### 3. Calibrate escalation

Create a separate validation split for selecting escalation thresholds
rather than tuning against the final golden set.

### 4. Improve retrieval

Experiment with:

- intent-aware retrieval
- reranking
- diversity-aware top-k selection
- filtering historical examples by predicted intent

### 5. Improve generation specificity

Require the generator to identify the supported action in the retrieved
evidence before producing a response.

### 6. Complete LLM-judge agreement analysis

Run the prepared LLM judge on the generated evaluation sample once sufficient
API quota is available. Then compare its helpfulness, groundedness, and safety
ratings against independently collected human ratings.

Human–LLM agreement is **not reported in this submission** because independent
human ratings have not yet been collected.

---

## 12. Reproducibility

The repository contains scripts for:

- Dataset analysis
- Conversation construction
- Golden-set creation
- Intent evaluation
- Retrieval-index construction
- Retrieval evaluation
- Escalation scoring
- Escalation evaluation
- Response generation
- Generation safety evaluation
- Human generation evaluation
- LLM-judge preparation

The main end-to-end agent can be run with:

```bash
python src/agent.py "The item I received is broken."