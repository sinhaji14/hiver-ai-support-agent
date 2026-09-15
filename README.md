````markdown
# Hiver AI Support Agent

An AI customer-support agent for **AmazonHelp** that classifies customer issues, retrieves relevant historical support interactions, generates grounded responses, and decides when a human should take over.

The system is designed as a **conservative, retrieval-grounded support agent**: when evidence is weak, the intent is ambiguous, or the issue is high-risk, it prefers human escalation over unsupported automation.

## 1. Overview

For an incoming customer message, the pipeline performs:
flowchart LR
    A[Customer Message] --> B[Intent Classification]
    B --> C[Historical Retrieval]
    C --> D[Escalation Decision]
    D -->|AUTO-HANDLE| E[Response Generation]
    E --> F[Safety Checks]
    F --> G[Final Response]
    D -->|HUMAN| H[Human Agent]
````

### Pipeline

1. **Intent classification** — predicts the customer's primary support need across 10 intents.
2. **Historical retrieval** — finds semantically similar AmazonHelp customer-support interactions.
3. **Escalation decision** — determines whether the case is safe to automate.
4. **Response generation** — generates a concise response grounded in historical support behavior.
5. **Safety checks** — removes potentially sensitive information and prevents unsupported claims.
6. **Final response** — returns either an automated response or routes the case to a human.

### What good means

For this project, a good support agent should:

* correctly understand the customer's primary issue
* use relevant historical evidence
* generate concise and useful responses
* avoid inventing policies, refunds, dates, or actions
* avoid leaking sensitive information
* escalate uncertain or high-risk cases
* prefer safe automation over maximum automation coverage

### What is intentionally not built

This prototype does **not**:

* access live Amazon orders or customer accounts
* perform refunds, cancellations, replacements, or other transactions
* verify real-time delivery status
* call external customer-support systems
* make account-specific decisions
* guarantee that generated responses represent Amazon's current policy
* optimize purely for maximum automation coverage

---

# 2. Dataset

The project uses the public **Customer Support on Twitter** dataset from Kaggle.

Dataset:

[https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)

The dataset contains customer tweets and support-agent responses, including conversation relationships.

For this project, **AmazonHelp** was selected because it has the largest support-response volume among the evaluated brands.

### AmazonHelp dataset statistics

| Metric                              |   Value |
| ----------------------------------- | ------: |
| AmazonHelp support tweets           | 169,840 |
| Support tweets with parent ID       | 169,287 |
| Matched customer parent tweets      | 154,976 |
| Clean customer-support interactions | 168,814 |
| Conversations                       |  55,733 |
| Customer messages                   | 134,997 |
| Agent messages                      | 132,718 |
| Average turns                       |    4.80 |
| Median turns                        |       3 |

The conversation dataset is used for analysis and leakage-safe retrieval.

---

# 3. Golden Evaluation Set

A hand-labelled **200-example golden set** was created from AmazonHelp interactions.

Sampling:

* random sample
* seed: `42`
* size: `200`

Each example contains:

* customer message
* historical support response
* human-labelled intent
* human escalation decision

### Intent annotation guideline

The primary intent was chosen based on the customer's main support need, with the historical support response used as additional context.

### Escalation annotation guideline

An example was labelled **HUMAN** when it involved:

* account-specific actions
* financial or security-sensitive handling
* investigation requiring information unavailable to the system
* insufficient evidence for a safe response

An example was labelled **AUTO-HANDLE** when the historical response provided a clear, low-risk answer or instruction that could reasonably be reproduced.

### Golden set distribution

| Intent                                  | Examples |
| --------------------------------------- | -------: |
| Delivery Delay & Tracking               |       56 |
| Customer Support & General Inquiry      |       52 |
| Technical & Digital Services            |       22 |
| Order, Subscription & Pre-order Issues  |       17 |
| Missing, Misdelivered & Lost Package    |       11 |
| Pricing, Promotions & Cashback          |       11 |
| Account Access & Security               |        9 |
| Returns, Refunds & Replacements         |        9 |
| Damaged, Defective & Incorrect Item     |        8 |
| Billing, Payment & Unauthorized Charges |        5 |
| **Total**                               |  **200** |

The distribution is intentionally reported because the headline accuracy is affected by the concentration of examples in the two largest classes.

### Escalation labels

| Label       | Examples |
| ----------- | -------: |
| AUTO-HANDLE |       72 |
| HUMAN       |      128 |

These escalation labels are **policy annotations**, not objective ground truth.

---

# 4. Intent Taxonomy

The system uses 10 manually consolidated support intents.

| Intent ID                          | Intent                                  | Description                                                                            |
| ---------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------- |
| `delivery_delay_tracking`          | Delivery Delay & Tracking               | Delayed shipments, missed delivery dates, unreliable estimates, or inaccurate tracking |
| `missing_misdelivered_package`     | Missing, Misdelivered & Lost Package    | Missing, lost, misdelivered, or incorrectly marked-delivered packages                  |
| `damaged_defective_incorrect_item` | Damaged, Defective & Incorrect Item     | Damaged, broken, defective, counterfeit, poor-quality, or incorrect products           |
| `returns_refunds_replacements`     | Returns, Refunds & Replacements         | Returns, pickups, refunds, replacements, or delayed refund processing                  |
| `billing_payment_charges`          | Billing, Payment & Unauthorized Charges | Payment failures, billing errors, unexpected or unauthorized charges                   |
| `pricing_promotions_cashback`      | Pricing, Promotions & Cashback          | Prices, discounts, promotions, promotional credits, or cashback                        |
| `account_access_security`          | Account Access & Security               | Login, verification, lockout, fraud, phishing, or account-security problems            |
| `technical_digital_services`       | Technical & Digital Services            | Website, mobile app, Fire TV, Prime Video, or other technical issues                   |
| `order_subscription_preorder`      | Order, Subscription & Pre-order Issues  | Cancellations, subscriptions, pre-orders, out-of-stock and order-management issues     |
| `customer_support_general`         | Customer Support & General Inquiry      | Unresolved support, communication problems, ineffective support, and general questions |

The taxonomy is stored in:

```text
src/intent/taxonomy.json
```

---

# 5. Intent Classification

Several increasingly strong baselines were evaluated using the 200-example golden set.

For the final comparison, the supervised models use **5-fold stratified cross-validation** where applicable.

| Model                                         |  Accuracy |  Macro-F1 | Weighted-F1 |
| --------------------------------------------- | --------: | --------: | ----------: |
| Majority baseline                             | **28.0%** |         — |           — |
| TF-IDF + Logistic Regression                  | **21.0%** |     0.167 |       0.233 |
| Class-level mean embeddings                   | **37.5%** |     0.270 |       0.386 |
| Multi-prototype embeddings                    | **49.0%** |     0.376 |       0.478 |
| Hybrid multi-prototype + taxonomy description | **51.5%** | **0.425** |   **0.515** |

The final classifier combines:

* multiple semantic prototypes per intent
* taxonomy descriptions
* cosine similarity
* intent margin between the best and second-best candidates

### Why multi-prototypes?

A single embedding prototype can hide substantial variation within a support intent.

For example, delivery issues can contain:

* "Where is my package?"
* "Tracking hasn't updated"
* "My delivery is late"
* "It was supposed to arrive yesterday"

Multiple prototypes better capture these linguistic variations.

### Final intent model

The production implementation is:

```text
src/intent/multi_prototype.py
src/intent/hybrid.py
```

The confidence score is used as a decision signal, **not as a calibrated probability**.

---

# 6. Historical Retrieval

Historical support interactions are indexed using sentence-transformers:

```text
all-MiniLM-L6-v2
```

The model produces 384-dimensional embeddings.

### Leakage-safe retrieval

A major evaluation concern is conversation leakage.

If one customer message from a conversation is placed in the golden set while another message from the same conversation remains in the retrieval corpus, retrieval can become artificially easy.

To prevent this:

1. identify every conversation containing a golden-set customer message
2. exclude the entire conversation from the retrieval corpus
3. build the retrieval index from the remaining interactions

Results:

| Metric                    |  Value |
| ------------------------- | -----: |
| Total conversations       | 55,733 |
| Excluded conversations    |    528 |
| Leakage-safe interactions | 98,232 |
| Embedding dimensions      |    384 |

### Leakage-safe retrieval diagnostic

On the 200-example golden set:

| Metric     | Cosine Similarity |
| ---------- | ----------------: |
| Mean Top-1 |        **0.7644** |
| Mean Top-3 |            0.7515 |
| Mean Top-5 |            0.7434 |
| P10        |            0.6293 |
| P25        |            0.7004 |
| Median     |            0.7665 |
| P75        |            0.8275 |
| P90        |            0.9013 |

**Important:** these are retrieval-quality diagnostics, not retrieval accuracy.

A high cosine similarity does not prove that the retrieved response is correct or appropriate for the current customer.

---

# 7. Response Generation

The response generator uses retrieved historical support interactions as evidence.

The generator is instructed to:

* use only historical support evidence
* avoid inventing Amazon policies
* avoid inventing refunds, replacements, credits, or compensation
* avoid inventing dates or delivery guarantees
* avoid claiming that an action was performed
* avoid inventing order information
* avoid URLs, phone numbers, or email addresses
* remain cautious when evidence is insufficient
* produce concise, professional responses

The generation implementation is:

```text
src/generation/generate_reply.py
```

The final agent pipeline is:

```text
src/agent.py
```

### Sensitive-information sanitization

Generated and retrieved text is sanitized for:

* URLs
* Amazon order numbers
* email addresses
* phone numbers
* Twitter handles

This prevents historical customer-specific information from being copied directly into a new response.

---

# 8. Escalation Policy

The system intentionally uses a **conservative escalation policy**.

The current policy escalates when:

1. no relevant historical examples are retrieved
2. retrieval similarity is too weak
3. intent classification is ambiguous
4. the intent is considered high-risk
5. the evidence is not strong enough for automatic handling

High-risk intents currently include:

```text
billing_payment_charges
account_access_security
missing_misdelivered_package
returns_refunds_replacements
```

The final policy was evaluated on all 200 golden examples.

### Escalation results

| Metric                |              Result |
| --------------------- | ------------------: |
| Accuracy              |           **66.0%** |
| Predicted AUTO-HANDLE |   **18/200 (9.0%)** |
| Predicted HUMAN       | **182/200 (91.0%)** |
| False AUTO-HANDLE     |   **7/128 (5.47%)** |

The low automation coverage is intentional: the system prioritizes avoiding unsafe automatic responses over maximizing the number of cases it handles automatically.

The policy implementation is:

```text
src/escalation/decide.py
```

---

# 9. End-to-End Generation Evaluation

A 20-example subset of the human-labelled `AUTO-HANDLE` examples was evaluated end-to-end.

The sample was selected using seed `42`.

## Automated safety checks

| Check                   |           Result |
| ----------------------- | ---------------: |
| URL violations          |         **0/20** |
| Email violations        |         **0/20** |
| Phone violations        |         **0/20** |
| Order-number violations |         **0/20** |
| Empty responses         |         **0/20** |
| Safety-clean responses  | **20/20 (100%)** |

## Intent consistency

The generated-response evaluation sample had:

**17/20 correct intent predictions = 85.0%**

Three examples had an intent mismatch.

---

# 10. LLM-as-a-Judge

Generated responses were evaluated using Gemini with a 1–3 rubric.

### Rubric

**Helpfulness**

* 1 = not useful
* 2 = partially useful
* 3 = directly useful

**Groundedness**

* 1 = unsupported or fabricated
* 2 = mostly grounded
* 3 = clearly grounded in available evidence

**Correctness**

* 1 = incorrect
* 2 = partially correct
* 3 = correct

**Safety**

* 1 = unsafe
* 2 = minor concern
* 3 = safe

### Results

| Dimension    |              Average |
| ------------ | -------------------: |
| Helpfulness  |         **2.65 / 3** |
| Groundedness |         **2.95 / 3** |
| Correctness  |         **2.70 / 3** |
| Safety       |         **2.95 / 3** |
| **Overall**  | **2.81 / 3 (93.8%)** |

The strongest dimensions were groundedness and safety.

The main weakness was helpfulness: several responses were safe but too generic or simply directed the customer to support instead of resolving the question directly.

---

# 11. Human vs LLM-Judge Validation

The same 20 responses were independently rated by a human using the same rubric.

### Agreement

| Dimension    |    Exact Agreement | Cohen's κ |
| ------------ | -----------------: | --------: |
| Helpfulness  |  **95.0% (19/20)** | **0.894** |
| Groundedness | **100.0% (20/20)** | **1.000** |
| Correctness  |  **95.0% (19/20)** | **0.886** |
| Safety       | **100.0% (20/20)** | **1.000** |

There was only one sample-level disagreement: **Sample 8**.

The human rated its helpfulness and correctness as 2, while the LLM judge rated both as 3. The response acknowledged the customer's suggestion but did not directly address the underlying pricing/payment concern, so the human assessment was stricter.

The agreement result is encouraging, but the validation set contains only 20 examples and should not be interpreted as definitive evidence of judge reliability across all possible support cases.

The judge implementation is:

```text
evaluation/llm_judge.py
```

The summary script is:

```text
evaluation/summarize_llm_judge.py
```

Independent human ratings are stored in:

```text
evaluation/human_generation_ratings_independent.json
```

---

# 12. Top Failure Modes

## Failure 1 — Safe but overly generic responses

Example:

> "Kindly connect with us, we'd like to help."

The response is safe and grounded but does not directly address the customer's pricing question.

### Hypothesis

The generator is overly conservative when retrieved examples do not provide a clear actionable resolution.

### Improvement

Use a structured response strategy:

```text
Acknowledge → Answer what can be answered → State limitation → Next action
```

---

## Failure 2 — Conversational messages are difficult to classify

Short messages such as:

* "Yes"
* "Thanks"
* "You're welcome"
* positive feedback

contain little explicit intent information.

These can be incorrectly mapped to a substantive support category.

### Hypothesis

The taxonomy is designed around support issues, while some Twitter conversations contain social or conversational turns.

### Improvement

Introduce additional classes such as:

```text
chitchat
positive_feedback
acknowledgement
other
```

or detect conversational turns before intent classification.

---

## Failure 3 — Conservative escalation produces low automation coverage

The current policy automatically handles only:

**9.0% of the golden set.**

This is a deliberate safety trade-off.

### Hypothesis

The combination of high-risk intent rules and strict retrieval/intent thresholds rejects many cases that could potentially be automated safely.

### Improvement

Collect a larger policy-labelled validation set and tune thresholds against explicit business costs:

```text
false auto-handle cost
vs
unnecessary human escalation cost
```

---

## Failure 4 — Retrieval similarity is not answer correctness

The leakage-safe retrieval diagnostic has a mean Top-1 similarity of:

**0.7644**

However, similarity alone does not guarantee that the retrieved response is appropriate.

A semantically similar example can still contain:

* outdated policy
* different customer context
* account-specific instructions
* an action that cannot safely be reproduced

### Improvement

Add a retrieval relevance judge or cross-encoder reranker that evaluates whether the retrieved example actually supports the intended response.

---

## Failure 5 — Historical context can be insufficient

Some customer requests require information unavailable in the historical interaction:

* current order status
* account state
* payment state
* current policy
* transaction-specific information

In such cases, generating a confident answer from historical examples would be unsafe.

### Improvement

Explicitly detect missing required context and escalate instead of generating.

---

# 13. What Is Misleading About My Headline Number?

The strongest-looking number in this project is the **0.7644 mean Top-1 retrieval similarity**.

It is misleading if interpreted as:

> "The retrieval system is 76.44% accurate."

That is **not** what the number means.

It only measures the cosine similarity between each golden customer message and its nearest historical interaction after conversation-level leakage prevention.

A high similarity does not guarantee:

* the retrieved answer is correct
* the historical policy is still valid
* the response is appropriate
* the customer can actually receive the suggested action

Similarly, the **93.8% LLM-judge score** should not be interpreted as 93.8% of customers receiving a correct answer. It is an average rubric score over only 20 generated examples.

The most meaningful headline result is therefore the combination of:

* **51.5% intent accuracy**
* **66.0% escalation-policy accuracy**
* **5.47% false AUTO-HANDLE rate**
* **100% safety-clean responses on the 20-example generation evaluation**
* **strong human/LLM judge agreement on the same 20 examples**

rather than any single metric.

---

# 14. Key Results

| Component        | Metric                             |             Result |
| ---------------- | ---------------------------------- | -----------------: |
| Intent           | Majority baseline accuracy         |          **28.0%** |
| Intent           | Hybrid model accuracy              |          **51.5%** |
| Intent           | Hybrid Macro-F1                    |          **0.425** |
| Intent           | Hybrid Weighted-F1                 |          **0.515** |
| Retrieval        | Leakage-safe mean Top-1 similarity |         **0.7644** |
| Escalation       | Policy accuracy                    |          **66.0%** |
| Escalation       | False AUTO-HANDLE                  |          **5.47%** |
| Escalation       | Predicted AUTO-HANDLE              |           **9.0%** |
| Generation       | Safety-clean responses             |   **100% (20/20)** |
| Generation       | Intent consistency                 |  **85.0% (17/20)** |
| Generation       | Helpfulness                        |         **2.65/3** |
| Generation       | Groundedness                       |         **2.95/3** |
| Generation       | Correctness                        |         **2.70/3** |
| Generation       | Safety                             |         **2.95/3** |
| Generation       | Overall LLM-judge score            | **2.81/3 (93.8%)** |
| Judge validation | Helpfulness κ                      |          **0.894** |
| Judge validation | Groundedness κ                     |          **1.000** |
| Judge validation | Correctness κ                      |          **0.886** |
| Judge validation | Safety κ                           |          **1.000** |

---

# 15. Reproducibility

## Requirements

* Python 3.10+
* Kaggle dataset
* Gemini API key for LLM generation/judge evaluation

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit `.env`.

---

## Dataset preparation

Download the Kaggle dataset and place it under:

```text
data/raw/
```

Expected structure:

```text
data/raw/
└── twcs/
    └── twcs.csv
```

Build the AmazonHelp conversation dataset:

```bash
python src/data/analyze_amazon.py
python src/data/build_conversations.py
python src/data/build_interactions.py
```

Create the intent sample:

```bash
python src/data/create_intent_sample.py
```

---

## Golden set

Create the golden set:

```bash
python src/intent/create_golden_set.py
```

Label it interactively:

```bash
python src/intent/label_golden.py
```

Label escalation decisions:

```bash
python src/escalation/label_escalation.py
```

---

## Intent evaluation

Run the baseline:

```bash
python evaluation/baseline_majority.py
```

Run the intent evaluation:

```bash
python evaluation/evaluate_intent.py
```

Run the multi-prototype evaluation:

```bash
python evaluation/evaluate_multi_prototype.py
```

Run the hybrid evaluation:

```bash
python evaluation/evaluate_hybrid.py
```

---

## Leakage-safe retrieval

Build the leakage-safe retrieval index:

```bash
python src/retrieval/build_leakage_safe_index.py
```

The index is created under:

```text
data/processed/retrieval_index_safe/
```

---

## Run the agent

Example:

```bash
python src/agent.py
```

The pipeline performs:

```text
Customer message
      ↓
Intent classification
      ↓
Historical retrieval
      ↓
Escalation decision
      ↓
 ┌───────────────┐
 │               │
HUMAN        AUTO-HANDLE
                 ↓
        Response generation
                 ↓
           Safety checks
                 ↓
          Final response
```

---

## Generation evaluation

Create the evaluation sample:

```bash
python evaluation/create_generation_sample.py
```

Run response generation:

```bash
python evaluation/run_generation_eval.py
```

Evaluate safety and intent consistency:

```bash
python evaluation/evaluate_generation.py
```

Run the LLM judge:

```bash
python evaluation/llm_judge.py
```

Summarize judge results:

```bash
python evaluation/summarize_llm_judge.py
```

Collect independent human ratings:

```bash
python evaluation/collect_human_ratings.py
```

Compare human and LLM ratings:

```bash
python evaluation/compare_human_llm.py
```

---

# 16. Project Structure

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
│   ├── compare_human_llm.py
│   ├── collect_human_ratings.py
│   ├── create_generation_sample.py
│   ├── evaluate_generation.py
│   ├── evaluate_intent.py
│   ├── llm_judge.py
│   ├── summarize_llm_judge.py
│   └── human_generation_ratings_independent.json
│
├── src/
│   ├── agent.py
│   │
│   ├── data/
│   │   ├── analyze_amazon.py
│   │   ├── build_conversations.py
│   │   ├── build_interactions.py
│   │   └── create_intent_sample.py
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
│       ├── label_escalation.py
│       └── decide.py
│
├── tests/
├── DECISIONS.md
├── README.md
├── REPORT.md
├── requirements.txt
└── .gitignore
```

---

# 17. Engineering Decisions

The major non-obvious decisions are documented in:

```text
DECISIONS.md
```

Important decisions include:

1. Selecting AmazonHelp because of its large support volume.
2. Using conversation-level leakage prevention.
3. Manually consolidating the initial intent candidates into 10 operational intents.
4. Using multiple semantic prototypes rather than one prototype per class.
5. Combining prototypes with taxonomy descriptions.
6. Treating confidence as a decision signal rather than a calibrated probability.
7. Using historical support interactions as the grounding source.
8. Preventing invented policies and unsupported actions.
9. Sanitizing sensitive information from retrieved/generated text.
10. Treating financial, security, account-specific, delivery-investigation, and refund cases conservatively.
11. Optimizing the escalation policy for low false-auto risk rather than maximum automation coverage.
12. Validating the LLM judge against independent human ratings.
13. Reporting retrieval similarity as a diagnostic rather than calling it retrieval accuracy.
14. Explicitly documenting limitations of the small generation evaluation set.

---

# 18. Limitations

### Dataset limitations

The source dataset consists of historical Twitter interactions and may not represent modern customer-support traffic.

### Taxonomy limitations

The 10-intent taxonomy is manually consolidated and does not contain explicit classes for conversational or positive-feedback messages.

### Golden-set limitations

The golden set contains 200 examples and is not perfectly balanced across intents.

### Retrieval limitations

Historical similarity does not guarantee that the retrieved response is correct for the current case.

### Generation limitations

The generation evaluation contains only 20 examples.

### Judge limitations

Human-vs-LLM judge agreement was measured on only 20 examples and therefore provides limited evidence about judge reliability.

### Policy-label limitations

Escalation labels represent a designed safety policy rather than objective customer-support truth.

### Live-system limitations

The prototype does not access live customer accounts, orders, payments, delivery systems, or current Amazon policies.

---

# 19. What I Would Improve With One More Week

## 1. Expand the golden set

Increase the evaluation set from 200 to approximately 1,000 examples with better class balance.

Add:

```text
Other
Chitchat
Positive feedback
Ambiguous
```

This would reduce forced classification of conversational messages.

## 2. Build a calibrated confidence model

Instead of relying directly on cosine similarity and prototype margins, calibrate confidence using a held-out validation set.

Potential approaches:

* temperature scaling
* isotonic regression
* logistic calibration

## 3. Improve retrieval

Add a reranking stage:

```text
Embedding retrieval
        ↓
Top 20 candidates
        ↓
Cross-encoder / LLM reranker
        ↓
Top 3 evidence examples
```

## 4. Improve generation quality

Use structured generation:

```text
Customer issue
      ↓
What the evidence supports
      ↓
What the system cannot know
      ↓
Safe next action
```

This should reduce generic responses.

## 5. Evaluate automation as a business decision

Create a proper cost-sensitive evaluation:

```text
False automation
        vs
Unnecessary escalation
```

The threshold should be chosen using business costs rather than only classification accuracy.

## 6. Validate the judge at larger scale

Use 100+ independently human-rated examples spanning:

* easy cases
* ambiguous cases
* high-risk cases
* poor retrieval
* conversational messages
* generated responses

Then measure agreement and judge stability across categories.

---

# 20. Security and Privacy

The project follows these principles:

* API keys are stored in `.env`
* `.env` is excluded through `.gitignore`
* raw and processed datasets are excluded from Git
* customer-specific order information is sanitized
* URLs, emails, phone numbers, and Twitter handles are removed from generated output
* the agent does not claim to perform account-specific actions
* high-risk cases are routed to humans

The system is therefore designed as a **prototype for safe support automation**, not as a production system with access to customer accounts.

---

# 21. Summary

The final system combines:

```text
Intent Classification
        +
Leakage-Safe Historical Retrieval
        +
Grounded Response Generation
        +
Conservative Escalation
        +
Safety Sanitization
        +
LLM-as-a-Judge
        +
Human Judge Validation
```

The strongest result is not maximum automation.

Instead, the system demonstrates a conservative approach to AI support:

* **51.5% intent accuracy**
* **66.0% escalation-policy accuracy**
* **5.47% false AUTO-HANDLE rate**
* **9.0% automatic handling coverage**
* **100% safety-clean generated responses in the 20-example evaluation**
* **93.8% overall LLM-judge score**
* **strong human/LLM agreement across all four quality dimensions**

The main remaining challenge is improving **usefulness and automation coverage without increasing unsafe automatic responses**.

That is the primary direction for further development.
