````markdown
# Hiver AI Support Agent

> A conservative, retrieval-grounded AI customer-support agent for AmazonHelp.

The system classifies customer issues, retrieves relevant historical support interactions, generates grounded responses, and decides whether a request can be safely automated or should be escalated to a human agent.

The primary design goal is **safe automation rather than maximum automation coverage**.

---

## Overview

For every incoming customer message, the system follows this pipeline:

```mermaid
flowchart TD
    A["Customer Message"] --> B["Intent Classification"]
    B --> C["Historical Retrieval"]
    C --> D["Escalation Decision"]

    D -->|"AUTO-HANDLE"| E["Response Generation"]
    E --> F["Safety Checks"]
    F --> G["Final Response"]

    D -->|"HUMAN"| H["Human Agent"]
````

### Pipeline stages

| Stage                     | Purpose                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| **Customer Message**      | Receive the incoming support request                               |
| **Intent Classification** | Identify the customer's primary support intent                     |
| **Historical Retrieval**  | Find semantically similar historical customer-support interactions |
| **Escalation Decision**   | Decide whether the case is safe to automate                        |
| **Response Generation**   | Generate a concise response grounded in historical evidence        |
| **Safety Checks**         | Detect and remove sensitive information and unsupported content    |
| **Final Response**        | Return the response or route the case to a human agent             |

### Design principle

The agent deliberately prefers:

**safe human escalation > unsupported automatic response**

Cases with weak retrieval evidence, ambiguous intent, or high-risk requirements are therefore routed to a human.

---

# 1. Problem 

The goal is to build a customer-support agent that can handle common support requests while minimizing unsafe or unsupported responses.

A good system should:

* understand the customer's primary issue
* retrieve relevant historical support evidence
* generate useful and concise responses
* avoid hallucinating policies or actions
* avoid leaking customer-specific information
* recognize when it does not have enough evidence
* escalate high-risk or ambiguous cases to a human

## What this prototype does

* classifies incoming customer messages into 10 support intents
* retrieves similar historical AmazonHelp interactions
* uses historical responses as grounding evidence
* generates concise support responses
* applies deterministic safety sanitization
* makes a conservative human-vs-automation decision
* evaluates intent classification, retrieval, escalation, generation, and judge agreement

## What this prototype does not do

This is not a production customer-support system.

It does **not**:

* access live Amazon customer accounts
* access live order information
* perform refunds or replacements
* cancel orders
* verify real-time delivery status
* modify customer data
* make account-specific decisions
* guarantee current Amazon policy compliance
* connect to production customer-support systems

---

# 2. Dataset

The project uses the public **Customer Support on Twitter** dataset from Kaggle.

The dataset contains customer tweets, support-agent responses, and conversation relationships.

For this project, **AmazonHelp** was selected because it had the largest support-response volume among the evaluated brands.

### Dataset statistics

| Metric                              |   Value |
| ----------------------------------- | ------: |
| AmazonHelp support tweets           | 169,840 |
| Support tweets with parent ID       | 169,287 |
| Matched customer parent tweets      | 154,976 |
| Clean customer-support interactions | 168,814 |
| Conversations                       |  55,733 |
| Customer messages                   | 134,997 |
| Agent messages                      | 132,718 |
| Average conversation turns          |    4.80 |
| Median conversation turns           |       3 |

The raw dataset is intentionally excluded from Git because of its size.

Expected raw-data location:

```text
data/raw/twcs/twcs.csv
```

---

# 3. Evaluation Dataset

A hand-labelled **200-example golden evaluation set** was created from AmazonHelp interactions.

### Sampling

* random sampling
* sample size: **200**
* random seed: **42**

Each example contains:

* customer message
* historical support response
* human-labelled intent
* human-labelled escalation decision

The golden set is stored in:

```text
data/golden/golden_set.jsonl
```

## Annotation guideline

The primary intent is the issue most directly associated with the customer's support need, using the historical support response as additional context.

For escalation:

### AUTO-HANDLE

Used when the historical response provides a clear, low-risk answer or instruction that can reasonably be reproduced.

### HUMAN

Used when the case involves:

* account-specific actions
* financial or security-sensitive handling
* investigation requiring unavailable information
* insufficient evidence for a safe response

The escalation labels represent a **designed safety policy**, rather than objective ground truth.

---

# 4. Golden Set Distribution

The 200-example evaluation set contains the following intent distribution:

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

The distribution is important when interpreting headline accuracy because **54% of the evaluation set belongs to the two largest intents**.

### Escalation distribution

| Decision    | Examples |
| ----------- | -------: |
| AUTO-HANDLE |       72 |
| HUMAN       |      128 |
| **Total**   |  **200** |

---

# 5. Intent Taxonomy

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

# 6. Intent Classification

Several increasingly strong approaches were evaluated.

The final supervised comparisons use **5-fold stratified cross-validation** where applicable.

| Model                                             |  Accuracy |  Macro-F1 | Weighted-F1 |
| ------------------------------------------------- | --------: | --------: | ----------: |
| Majority baseline                                 | **28.0%** |         — |           — |
| TF-IDF + Logistic Regression                      | **21.0%** |     0.167 |       0.233 |
| Class-level mean embeddings                       | **37.5%** |     0.270 |       0.386 |
| Multi-prototype embeddings                        | **49.0%** |     0.376 |       0.478 |
| **Hybrid multi-prototype + taxonomy description** | **51.5%** | **0.425** |   **0.515** |

### Final approach

The production classifier combines:

1. multiple semantic prototypes per intent
2. taxonomy descriptions
3. cosine similarity
4. the margin between the top intent candidates

The final implementation is:

```text
src/intent/multi_prototype.py
src/intent/hybrid.py
```

### Why multiple prototypes?

A single embedding prototype can fail to represent the linguistic diversity within an intent.

For example, delivery issues may appear as:

```text
"Where is my package?"
"My tracking hasn't updated."
"My delivery is late."
"It was supposed to arrive yesterday."
```

Multiple prototypes allow the classifier to represent several semantic patterns within the same intent.

### Important limitation

The model's confidence score is a **decision signal**, not a calibrated probability.

---

# 7. Historical Retrieval

Historical support interactions are embedded using:

```text
all-MiniLM-L6-v2
```

Each interaction is represented using a 384-dimensional embedding.

The retrieval system searches historical customer-support interactions for semantically similar examples.

## Retrieval corpus

The initial historical corpus contained:

```text
168,313 interactions
```

However, using random interaction-level splitting can create conversation leakage.

For example:

```text
Golden example:
"My package is late."

Historical corpus:
"Yes, the same package is still delayed."
```

If both belong to the same conversation, retrieval can appear much stronger than it actually is.

---

# 8. Leakage-Safe Retrieval

To prevent conversation leakage, the final retrieval evaluation uses **conversation-level exclusion**.

For every golden example:

1. identify the conversation containing the golden customer message
2. exclude that entire conversation
3. build the retrieval corpus from the remaining conversations
4. evaluate retrieval against the excluded golden examples

### Leakage-safe index

| Metric                    |  Value |
| ------------------------- | -----: |
| Total conversations       | 55,733 |
| Excluded conversations    |    528 |
| Leakage-safe interactions | 98,232 |
| Embedding dimensions      |    384 |

Implementation:

```text
src/retrieval/build_leakage_safe_index.py
```

## Retrieval diagnostic

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

### Important interpretation

**0.7644 is not retrieval accuracy.**

It is the mean cosine similarity between a golden customer message and its nearest historical interaction.

A high similarity score does not guarantee that:

* the retrieved answer is correct
* the historical policy is still valid
* the response applies to the current customer
* the customer can actually perform the suggested action

---

# 9. Response Generation

The response generator uses historical support interactions as grounding evidence.

Implementation:

```text
src/generation/generate_reply.py
```

The complete agent pipeline is:

```text
src/agent.py
```

## Generation principles

The generator is instructed to:

* rely on historical support evidence
* avoid inventing Amazon policies
* avoid inventing refunds
* avoid inventing replacements
* avoid inventing credits or compensation
* avoid inventing dates or delivery guarantees
* avoid claiming that an action was performed
* avoid inventing order information
* avoid copying historical customer-specific information
* avoid unnecessary URLs, emails, or phone numbers
* remain cautious when evidence is insufficient
* keep responses concise and professional

---

# 10. Safety and Sanitization

Generated and retrieved content is sanitized before being returned.

The sanitizer removes or replaces:

* URLs
* Amazon order numbers
* email addresses
* phone numbers
* Twitter handles

This prevents historical customer-specific information from being accidentally copied into a new response.

The system therefore treats historical responses as **evidence**, not as text to blindly copy.

---

# 11. Escalation Policy

The escalation policy is intentionally conservative.

A request is routed to a human when:

* no relevant historical evidence is found
* retrieval similarity is too weak
* intent classification is ambiguous
* the intent is considered high-risk
* evidence is insufficient for safe automation

## High-risk intents

The current policy treats the following intents conservatively:

```text
billing_payment_charges
account_access_security
missing_misdelivered_package
returns_refunds_replacements
```

Implementation:

```text
src/escalation/decide.py
```

## Escalation evaluation

| Metric                |              Result |
| --------------------- | ------------------: |
| Policy accuracy       |           **66.0%** |
| Predicted AUTO-HANDLE |   **18/200 (9.0%)** |
| Predicted HUMAN       | **182/200 (91.0%)** |
| False AUTO-HANDLE     |   **7/128 (5.47%)** |

The policy intentionally sacrifices automation coverage to reduce unsafe automatic handling.

### Why not maximize automation?

A false automatic response can be more costly than an unnecessary human escalation.

For example:

```text
Customer has a payment problem
        ↓
Weak evidence
        ↓
Confident automated response
        ↓
Potentially incorrect financial guidance
```

The current system instead prefers:

```text
Customer has a payment problem
        ↓
High-risk intent
        ↓
HUMAN
```

---

# 12. End-to-End Generation Evaluation

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

The predicted intent matched the human-labelled intent for:

**17/20 examples = 85.0%**

Three examples contained an intent mismatch.

---

# 13. LLM-as-a-Judge

Generated responses were evaluated using Gemini with a 1–3 rubric.

## Evaluation rubric

### Helpfulness

| Score | Meaning          |
| ----- | ---------------- |
| 1     | Not useful       |
| 2     | Partially useful |
| 3     | Directly useful  |

### Groundedness

| Score | Meaning                                |
| ----- | -------------------------------------- |
| 1     | Unsupported or fabricated              |
| 2     | Mostly grounded                        |
| 3     | Clearly grounded in available evidence |

### Correctness

| Score | Meaning           |
| ----- | ----------------- |
| 1     | Incorrect         |
| 2     | Partially correct |
| 3     | Correct           |

### Safety

| Score | Meaning       |
| ----- | ------------- |
| 1     | Unsafe        |
| 2     | Minor concern |
| 3     | Safe          |

## Results

| Dimension    |              Average |
| ------------ | -------------------: |
| Helpfulness  |         **2.65 / 3** |
| Groundedness |         **2.95 / 3** |
| Correctness  |         **2.70 / 3** |
| Safety       |         **2.95 / 3** |
| **Overall**  | **2.81 / 3 (93.8%)** |

The strongest dimensions were **groundedness** and **safety**.

The main weakness was **helpfulness**. Several responses were safe and grounded but too generic or simply redirected the customer to support.

---

# 14. Human vs LLM-Judge Validation

To validate the LLM judge, the same 20 responses were independently rated by a human using the same rubric.

## Agreement results

| Dimension    |    Exact Agreement | Cohen's κ |
| ------------ | -----------------: | --------: |
| Helpfulness  |  **95.0% (19/20)** | **0.894** |
| Groundedness | **100.0% (20/20)** | **1.000** |
| Correctness  |  **95.0% (19/20)** | **0.886** |
| Safety       | **100.0% (20/20)** | **1.000** |

There was only one sample-level disagreement: **Sample 8**.

The human rated helpfulness and correctness as `2`, while the LLM judge rated both as `3`.

The response acknowledged the customer's suggestion but did not directly address the underlying pricing/payment concern. The human assessment was therefore stricter.

### Interpretation

The agreement results provide evidence that the LLM judge is reasonably aligned with human assessment on this evaluation sample.

However, the validation set contains only **20 examples**, so these results should not be treated as definitive evidence of judge reliability across the full support distribution.

Implementation:

```text
evaluation/llm_judge.py
evaluation/summarize_llm_judge.py
evaluation/compare_human_llm.py
```

Independent human ratings:

```text
evaluation/human_generation_ratings_independent.json
```

---

# 15. Top Failure Modes

## 1. Safe but overly generic responses

Example:

```text
"Kindly connect with us, we'd like to help."
```

The response is safe but does not directly answer the customer's pricing question.

### Hypothesis

The generator becomes overly conservative when retrieved examples do not contain a sufficiently clear actionable resolution.

### Improvement

Use a structured response strategy:

```text
Acknowledge
    ↓
Answer what the evidence supports
    ↓
State limitations
    ↓
Provide a safe next action
```

---

## 2. Conversational messages are difficult to classify

Very short messages such as:

```text
"Yes"
"Thanks"
"You're welcome"
```

contain little explicit intent information.

They can therefore be incorrectly mapped to substantive support categories.

### Hypothesis

The taxonomy focuses on support issues, while Twitter conversations also contain conversational turns.

### Improvement

Add dedicated categories such as:

```text
chitchat
positive_feedback
acknowledgement
other
```

or detect conversational messages before intent classification.

---

## 3. Conservative escalation produces low automation coverage

The current policy automatically handles only:

**9.0% of the golden set.**

This is intentional.

### Hypothesis

The combination of:

* high-risk intent rules
* retrieval thresholds
* intent-margin thresholds

rejects many cases that could potentially be automated safely.

### Improvement

Collect a larger policy-labelled validation set and optimize thresholds using explicit business costs:

```text
Cost of unsafe automation
        vs
Cost of unnecessary escalation
```

---

## 4. Retrieval similarity does not equal answer correctness

The leakage-safe retrieval diagnostic has a mean Top-1 similarity of:

**0.7644**

However, a semantically similar historical interaction can still contain:

* outdated information
* different customer context
* account-specific instructions
* actions that cannot safely be reproduced

### Improvement

Add a reranking stage:

```text
Embedding Retrieval
        ↓
Top-K Candidates
        ↓
Cross-Encoder / LLM Reranker
        ↓
Relevant Evidence
        ↓
Generation
```

---

## 5. Historical context can be insufficient

Some requests require information that is unavailable in historical interactions.

Examples include:

* current order status
* account state
* current payment state
* current policy
* transaction-specific information

Generating a confident answer in these cases would be unsafe.

### Improvement

Detect missing required context explicitly and escalate rather than generating unsupported answers.

---

# 16. What Is Misleading About My Headline Number?

The most tempting headline number is the:

**0.7644 mean Top-1 retrieval similarity.**

It would be misleading to describe this as:

> "76.44% retrieval accuracy."

That is not what the metric measures.

It is simply the average cosine similarity between each golden customer message and its nearest historical interaction after conversation-level leakage prevention.

Similarly, the:

**93.8% LLM-judge score**

does not mean that 93.8% of customers would receive a correct answer.

It is an average rubric score over only **20 generated examples**.

The results should therefore be interpreted together rather than relying on one headline number.

---

# 17. Key Results

| Component        | Metric                             |             Result |
| ---------------- | ---------------------------------- | -----------------: |
| Intent           | Majority baseline accuracy         |          **28.0%** |
| Intent           | Hybrid accuracy                    |          **51.5%** |
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

# 18. Reproducibility

## Requirements

* Python 3.10+
* Kaggle Customer Support on Twitter dataset
* Gemini API key for generation and LLM-judge evaluation

## Setup

```bash
git clone <repository-url>
cd hiver-ai-support-agent

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit the `.env` file.

---

## Dataset preparation

Place the Kaggle dataset at:

```text
data/raw/twcs/twcs.csv
```

Run:

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

## Golden evaluation set

Create the golden set:

```bash
python src/intent/create_golden_set.py
```

Label the examples:

```bash
python src/intent/label_golden.py
```

Label escalation decisions:

```bash
python src/escalation/label_escalation.py
```

---

## Intent evaluation

Run the majority baseline:

```bash
python evaluation/baseline_majority.py
```

Run the intent evaluation:

```bash
python evaluation/evaluate_intent.py
```

---

## Leakage-safe retrieval

Build the leakage-safe retrieval index:

```bash
python src/retrieval/build_leakage_safe_index.py
```

The resulting index is stored under:

```text
data/processed/retrieval_index_safe/
```

---

## Run the agent

```bash
python src/agent.py
```

The agent performs:

```text
Customer Message
       ↓
Intent Classification
       ↓
Historical Retrieval
       ↓
Escalation Decision
       ↓
 ┌───────────────┐
 │               │
HUMAN        AUTO-HANDLE
                 ↓
        Response Generation
                 ↓
           Safety Checks
                 ↓
          Final Response
```

---

## Generation evaluation

Create the generation evaluation sample:

```bash
python evaluation/create_generation_sample.py
```

Run generation evaluation:

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

Compare the independent human ratings with the LLM judge:

```bash
python evaluation/compare_human_llm.py
```

---

# 19. Project Structure

```text
hiver-ai-support-agent/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── golden/
│       └── golden_set.jsonl
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
│       ├── decide.py
│       └── label_escalation.py
│
├── tests/
├── DECISIONS.md
├── README.md
├── REPORT.md
├── requirements.txt
└── .gitignore
```

---

# 20. Engineering Decisions

The major non-obvious decisions are documented in:

```text
DECISIONS.md
```

Key decisions include:

1. Selecting AmazonHelp because of its high support volume.
2. Using conversation-level leakage prevention.
3. Consolidating the initial intent candidates into 10 operational intents.
4. Using multiple semantic prototypes instead of a single prototype per intent.
5. Combining semantic prototypes with taxonomy descriptions.
6. Treating classifier confidence as a decision signal rather than a calibrated probability.
7. Using historical support responses as grounding evidence.
8. Preventing invented policies, refunds, replacements, credits, dates, and actions.
9. Sanitizing customer-specific information from generated responses.
10. Treating financial, security, account-specific, delivery-investigation, and refund cases conservatively.
11. Optimizing escalation for low false-auto risk instead of maximum automation coverage.
12. Using conversation-level exclusion to reduce retrieval leakage.
13. Validating the LLM judge against independent human ratings.
14. Reporting retrieval similarity as a diagnostic rather than retrieval accuracy.

---

# 21. Limitations

## Dataset limitations

The source dataset consists of historical Twitter interactions and may not represent modern customer-support traffic.

## Taxonomy limitations

The taxonomy is manually consolidated and does not explicitly model conversational messages such as acknowledgements, thanks, or positive feedback.

## Golden-set limitations

The golden set contains 200 examples and is not perfectly balanced across intents.

## Retrieval limitations

Semantic similarity does not guarantee that the retrieved support response is appropriate for the current customer.

## Generation limitations

The generation evaluation contains only 20 examples.

## Judge limitations

Human-vs-LLM judge agreement was measured on only 20 examples.

## Policy-label limitations

Escalation labels represent a safety policy designed for this prototype rather than objective business ground truth.

## Live-system limitations

The system has no access to live Amazon accounts, orders, payments, delivery systems, or current policies.

---

# 22. What I Would Improve With One More Week

## 1. Expand the golden set

Increase the evaluation set from 200 to approximately 1,000 examples with better class balance.

Add explicit categories for:

```text
Other
Chitchat
Positive Feedback
Acknowledgement
Ambiguous
```

This would reduce forced classification of conversational messages.

---

## 2. Calibrate intent confidence

Instead of using raw similarity and intent margins, create a held-out validation set and calibrate confidence.

Potential approaches:

* temperature scaling
* isotonic regression
* logistic calibration

---

## 3. Improve retrieval

Introduce a reranking stage:

```text
Embedding Retrieval
        ↓
Top 20 Candidates
        ↓
Cross-Encoder / LLM Reranker
        ↓
Top 3 Evidence Examples
        ↓
Response Generation
```

---

## 4. Improve response usefulness

Use structured response generation:

```text
Customer Issue
      ↓
Evidence-supported answer
      ↓
Known limitation
      ↓
Safe next action
```

This should reduce generic responses while preserving the current safety constraints.

---

## 5. Optimize escalation using business costs

Instead of optimizing only classification accuracy, define explicit costs:

```text
False AUTO-HANDLE
        vs
Unnecessary HUMAN escalation
```

Then tune the escalation thresholds against those costs.

---

## 6. Expand judge validation

Increase human-vs-LLM validation to 100+ examples covering:

* easy cases
* ambiguous cases
* high-risk cases
* poor retrieval
* conversational messages
* incorrect intents
* generic responses

This would provide stronger evidence for evaluator reliability.

---

# 23. Security

The project follows these principles:

* API keys are stored in `.env`
* `.env` is excluded from Git
* raw and processed datasets are excluded from Git
* customer-specific order information is sanitized
* URLs, emails, phone numbers, and Twitter handles are removed from generated output
* the system does not claim to perform account-specific actions
* high-risk cases are escalated to human agents

This makes the system suitable as a **safe-support prototype**, not as a production system with direct access to customer accounts.

---

# 24. Final Takeaway

The project demonstrates a conservative approach to AI-assisted customer support:

```text
Intent Classification
        +
Leakage-Safe Retrieval
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

### Final headline results

* **51.5% intent accuracy**
* **0.425 Macro-F1**
* **66.0% escalation-policy accuracy**
* **5.47% false AUTO-HANDLE rate**
* **9.0% automatic handling coverage**
* **100% safety-clean responses on the 20-example generation evaluation**
* **85.0% intent consistency on the generation sample**
* **93.8% overall LLM-judge score**
* **Strong human/LLM judge agreement across all four evaluation dimensions**

The central trade-off is clear:

> **The system currently prioritizes safe, evidence-backed support over maximum automation coverage.**

```
```
