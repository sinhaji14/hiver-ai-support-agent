# Hiver AI Support Agent — Evaluation Report

## 1. Problem Framing

This project builds an AI customer-support agent for AmazonHelp using historical customer-support interactions from the Kaggle Customer Support on Twitter dataset.

For each incoming customer message, the system performs four stages:

1. Classifies the customer's primary support intent.
2. Retrieves semantically similar historical customer-support interactions.
3. Decides whether the case can be automatically handled or should be escalated to a human.
4. For cases that are safe to automate, generates a concise response grounded in historical support resolutions.

The design prioritizes safety over maximum automation coverage. In particular, financial, account-security, missing-package, and return/refund cases are treated conservatively.

### What "good" means

A good system should:

- Correctly identify the customer's primary support need.
- Retrieve historical examples that provide useful evidence for the current request.
- Avoid automatically handling cases requiring account-specific or high-risk actions.
- Produce concise responses grounded in observed support behavior.
- Avoid inventing policies, refunds, compensation, dates, guarantees, or completed actions.
- Avoid exposing historical customer information or identifiers.

### What is not built

This is an offline support-agent prototype rather than a production customer-support system.

It does not include:

- Live CRM or ticketing-system integration.
- Access to customer accounts or order systems.
- Real-time order-status lookup.
- Execution of refunds, replacements, cancellations, or other transactions.
- A production policy engine.
- Authentication or authorization infrastructure.
- Human-agent workflow integration.

The generated responses are therefore deliberately limited to actions supported by historical evidence.

---

## 2. Dataset and Target Brand

The Kaggle Customer Support on Twitter dataset contains customer messages and brand-support responses.

AmazonHelp was selected because it had the largest support volume among the candidate brands examined.

Key dataset statistics:

- AmazonHelp support tweets: 169,840
- Support tweets with parent IDs: 169,287
- Matched customer parent tweets: 154,976
- Clean customer-support interactions: 168,814
- Usable conversations: 55,733

The conversation builder produced 55,733 usable conversations containing approximately 135k customer messages and 133k support messages.

A 200-example golden evaluation set was sampled from the AmazonHelp interactions using a fixed random seed.

### Sampling and labeling

The golden set was randomly sampled using seed 42.

Each example was manually assigned one primary intent from the operational taxonomy. The annotation guideline was:

> Assign the primary intent corresponding to the issue most directly addressed by the customer's request and historical support response.

The same 200 examples were also manually assigned an escalation-policy label:

- AUTO-HANDLE: 72
- HUMAN: 128

These escalation labels represent human policy judgments about whether the case is safe to automate. They are not objective ground truth in the same sense as a directly observable property of the data.

### Golden-set distribution

| Intent | Count |
|---|---:|
| Delivery Delay & Tracking | 56 |
| Missing, Misdelivered & Lost Package | 11 |
| Damaged, Defective & Incorrect Item | 8 |
| Returns, Refunds & Replacements | 9 |
| Billing, Payment & Unauthorized Charges | 5 |
| Pricing, Promotions & Cashback | 11 |
| Account Access & Security | 9 |
| Technical & Digital Services | 22 |
| Order, Subscription & Pre-order Issues | 17 |
| Customer Support & General Inquiry | 52 |
| **Total** | **200** |

The set is therefore not class-balanced. Delivery and general-support examples account for more than half of the evaluation set. This matters when interpreting aggregate accuracy.

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

The taxonomy was created by reviewing candidate categories discovered from a 2,000-example sample and consolidating them into a smaller operational taxonomy.

The goal was not to maximize the number of categories. The goal was to create categories that are sufficiently distinct to support routing and escalation decisions.

One limitation is that positive feedback, thanks, and casual messages are currently forced into the operational taxonomy rather than having a dedicated non-issue category.

---

## 4. Intent Classification

### Baselines and evaluated approaches

Several classification strategies were evaluated.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Majority baseline | 28.0% | — | — |
| TF-IDF + Logistic Regression | 21.0% | 0.167 | 0.233 |
| Class mean prototype | 37.5% | 0.270 | 0.386 |
| Multi-prototype | 49.0% | 0.376 | 0.478 |
| Hybrid multi-prototype + taxonomy | **51.5%** | **0.425** | **0.515** |

The majority baseline predicts `delivery_delay_tracking` for every example because it is the largest class.

The main supervised comparisons use 5-fold stratified cross-validation on the 200-example golden set.

The TF-IDF baseline uses the same evaluation framework but performed worse than the majority baseline on this small, heterogeneous dataset.

### Final classifier

The final classifier uses multiple semantic prototypes for each intent rather than representing an intent with a single centroid.

The hybrid score combines:

- Similarity to multiple historical examples representing the intent.
- Similarity to the written taxonomy description.

This approach performed best among the evaluated classifiers.

The hybrid model achieved:

- Accuracy: 51.5%
- Macro-F1: 0.425
- Weighted-F1: 0.515

### Interpretation

The 51.5% result is meaningfully above the 28.0% majority baseline, but it should not be interpreted as production-level intent classification accuracy.

The golden set is small and contains many ambiguous conversational messages. Several classes also have very few examples, particularly billing and damaged/defective items.

The result is primarily evidence that the multi-prototype + taxonomy approach was the strongest of the tested approaches on this benchmark.

---

## 5. Historical Retrieval

The system uses `all-MiniLM-L6-v2` embeddings to retrieve historical customer-support interactions.

The original retrieval corpus contained 168,313 interactions after removing the 200 golden examples themselves.

However, simply removing the exact golden records is insufficient to prevent leakage because other messages from the same conversation could remain in the retrieval corpus.

### Leakage-safe evaluation

For the final retrieval diagnostic, the entire conversation containing each golden example was excluded from the retrieval corpus.

This produced:

- 98,232 historical interactions
- 384-dimensional embeddings
- 528 conversations excluded

This is a conversation-level leakage control rather than simple row-level filtering.

### Leakage-safe retrieval diagnostic

| Metric | Value |
|---|---:|
| Mean Top-1 cosine similarity | **0.7644** |
| P10 | 0.6293 |
| P25 | 0.7004 |
| Median | 0.7665 |
| P75 | 0.8275 |
| P90 | 0.9013 |

These numbers measure semantic similarity between the query and retrieved historical examples.

They do **not** measure retrieval correctness, resolution correctness, or answer accuracy.

---

## 6. Response Generation

For cases classified as safe to automate, the system retrieves the top five historical customer-support interactions and provides them as evidence to Gemini.

The generation prompt explicitly prohibits:

- Inventing refunds or replacements.
- Inventing compensation.
- Inventing dates or guarantees.
- Inventing order information.
- Claiming that an action has already been performed.
- Introducing unsupported policies.
- Exposing historical URLs or identifiers.
- Making unsupported commitments.

Historical examples are sanitized before being passed to the generator.

The sanitization layer removes:

- URLs
- Email addresses
- Phone numbers
- Amazon-style order numbers
- Twitter-style handles

Generated responses are also checked for sensitive-information leakage.

---

## 7. Escalation Policy

The escalation policy is intentionally conservative.

A case is escalated when:

- Historical retrieval evidence is weak.
- Intent classification is ambiguous.
- The intent is considered high risk.
- There is insufficient evidence for safe automatic handling.

High-risk intents include:

- Billing, Payment & Unauthorized Charges
- Account Access & Security
- Missing, Misdelivered & Lost Package
- Returns, Refunds & Replacements

The current policy uses both retrieval similarity and intent-separation thresholds.

### Evaluation

The 200 golden examples contain:

- AUTO-HANDLE: 72
- HUMAN: 128

Current policy results:

| Metric | Result |
|---|---:|
| Accuracy | **66.0%** |
| Predicted AUTO-HANDLE | **9.0%** |
| Predicted HUMAN | **91.0%** |
| False auto-handling | **5.47%** |

The current policy therefore sacrifices automation coverage to reduce unsafe automatic handling.

Of the 128 human-labelled HUMAN cases, 7 were incorrectly predicted as AUTO-HANDLE.

### Important limitation

The escalation labels are policy judgments created for this evaluation rather than objective ground truth.

The thresholds were also not selected using an independent validation set, so the reported policy performance should be treated as an evaluation of the current policy rather than a fully calibrated estimate of production performance.

A separate validation set should be used before making threshold changes based on these results.

---

## 8. Generation Evaluation

A 20-example generation evaluation sample was created from examples that had been manually labeled AUTO-HANDLE.

The generated responses were evaluated for:

- Helpfulness
- Groundedness
- Correctness
- Safety

Each dimension uses a 1–3 scale:

- 1 = Poor
- 2 = Partially acceptable
- 3 = Good

### Automated safety checks

All 20 generated responses passed the deterministic safety checks.

| Metric | Result |
|---|---:|
| URL leakage | 0 / 20 |
| Email leakage | 0 / 20 |
| Phone leakage | 0 / 20 |
| Order-number leakage | 0 / 20 |
| Empty responses | 0 / 20 |
| Safety-clean responses | **20 / 20 (100%)** |

### LLM-as-a-Judge results

The LLM judge evaluated all 20 generated responses.

| Dimension | Average |
|---|---:|
| Helpfulness | **2.65 / 3** |
| Groundedness | **2.95 / 3** |
| Correctness | **2.70 / 3** |
| Safety | **2.95 / 3** |
| **Overall** | **2.81 / 3 (93.8%)** |

The strongest dimensions were groundedness and safety.

The main weakness was helpfulness. Several responses were appropriately cautious but too generic to directly resolve the customer's issue.

Seven of the 20 examples received at least one score below 3.

Notable examples included:

- A price-reduction question receiving only a generic offer to help.
- A delivered-but-not-received package being redirected to customer support without a concrete troubleshooting step.
- A cancellation question receiving a support handoff rather than an explanation.
- A one-word "Yes" response causing the generator to promise a future response despite insufficient context.
- A late-delivery complaint receiving a request for tracking information rather than a more useful next step.

### Independent human validation of the LLM judge

To validate the automated judge, the same 20 generated responses were independently rated by a human using the same four dimensions and 1–3 scale.

The human ratings were collected separately from the LLM judge ratings.

| Dimension | Exact Agreement | Cohen's κ |
|---|---:|---:|
| Helpfulness | **95.0% (19/20)** | **0.894** |
| Groundedness | **100.0% (20/20)** | **1.000** |
| Correctness | **95.0% (19/20)** | **0.886** |
| Safety | **100.0% (20/20)** | **1.000** |

The only disagreement was Sample 8.

The human evaluator rated both helpfulness and correctness as 2, while the LLM judge rated both as 3.

The response acknowledged a customer's suggestion to support Sodexo coupons but did not directly answer whether the payment method was supported. This suggests that the judge can be somewhat more lenient toward polite acknowledgments when a response is not fully actionable.

The agreement results provide evidence that the judge is reasonably aligned with the independent human ratings on this small benchmark. However, the sample size of 20 is small, so these agreement estimates should be treated as directional rather than definitive.

---

## 9. Failure Analysis

### Failure 1 — Generic responses

Several responses use a safe but generic support handoff such as asking the customer to contact support.

This minimizes hallucination risk but can reduce helpfulness.

For example, a customer reporting a missing package received a response that essentially redirected them to customer service without providing a concrete diagnostic step.

**Hypothesis:** The generator is strongly constrained by the grounding rules and therefore prefers a safe handoff when the retrieved evidence does not provide a sufficiently specific action.

**Improvement:** Retrieve more specific historical resolutions and require the generator to identify an evidence-supported action before producing a response.

---

### Failure 2 — Intent confusion on conversational messages

The golden set contains ambiguous and conversational customer messages.

Positive feedback and thanks can be incorrectly classified into operational issue categories.

For example, a customer thanking AmazonHelp for a successful delivery can be classified as `delivery_delay_tracking` because the embedding representation focuses on the delivery topic even though the actual intent is positive feedback.

**Hypothesis:** The current taxonomy lacks a dedicated representation for non-problem messages.

**Improvement:** Add a dedicated `positive_feedback_or_chitchat` category or introduce a separate first-stage detector for problem vs. non-problem messages.

---

### Failure 3 — Conservative escalation

The current policy auto-handles only 9.0% of the golden set.

This limits automation coverage.

However, the policy also reduces false automatic handling to 5.47%.

**Hypothesis:** The current similarity and intent-margin thresholds are intentionally conservative and are especially restrictive for high-risk intents.

**Improvement:** Create a separate validation set and calibrate the thresholds against an explicit risk/coverage objective. Expand automation first within low-risk intents where historical evidence is strong.

---

### Failure 4 — Retrieval similarity is not correctness

A high cosine similarity does not guarantee that the retrieved historical resolution is appropriate for the current case.

For example, two customer messages can discuss the same topic while requiring different actions.

**Hypothesis:** Embedding similarity captures semantic/topic similarity better than resolution equivalence.

**Improvement:**

- Add intent-aware retrieval.
- Filter historical examples using the predicted intent.
- Add a reranking stage.
- Select diverse evidence rather than relying only on the highest similarity.
- Evaluate retrieval using answer-level relevance rather than cosine similarity alone.

---

### Failure 5 — Insufficient context can cause unsupported conversational behavior

A particularly clear example was a customer message consisting only of:

> "Yes"

The generated response said that the support team would get back to the customer shortly.

The response was polite, but the available message alone did not establish that a follow-up was actually pending.

The LLM judge consequently gave this example 2/3 for helpfulness, groundedness, correctness, and safety.

**Hypothesis:** The generation stage assumes conversational context that is not always available in the single-message evaluation setup.

**Improvement:** Detect extremely short or context-dependent messages and automatically escalate them or retrieve the preceding conversation context before generating a response.

---

## 10. What Is Misleading About My Headline Number?

A potentially misleading headline number is:

> **0.7644 mean Top-1 retrieval similarity**

Without context, this could easily be interpreted as "76.44% retrieval accuracy."

That interpretation is incorrect.

The value is a cosine-similarity diagnostic measuring how semantically close the retrieved historical example is to the customer query.

It does not measure:

- Whether the retrieved example contains the correct resolution.
- Whether the generated response is correct.
- Whether the customer issue would actually be resolved.

The retrieval evaluation is also intentionally leakage-safe: the entire conversation containing each golden example was excluded from the retrieval corpus.

Therefore, the honest interpretation is:

> The system generally finds semantically similar historical interactions, with a mean Top-1 cosine similarity of 0.7644 on the 200-example golden set, but this number does not establish retrieval correctness.

A second potentially misleading number is the 93.8% overall LLM-judge generation score.

This should not be interpreted as 93.8% of customers receiving a correct answer. It is the average normalized score across four qualitative dimensions on only 20 generated examples.

---

## 11. What I Would Improve With One More Week

### 1. Expand and rebalance the golden set

Increase the evaluation set beyond 200 examples, with deliberate coverage of minority intents such as:

- Billing
- Damaged/defective products
- Account/security
- Returns/refunds

Also include more difficult ambiguous and multi-intent examples.

---

### 2. Add an explicit non-issue category

Positive feedback, thanks, and casual messages are currently forced into the 10 operational intents.

Adding:

`positive_feedback_or_chitchat`

would reduce intent confusion and make routing more realistic.

---

### 3. Calibrate escalation

Create a separate validation set for selecting escalation thresholds.

Optimize for a measurable trade-off between:

- False auto-handling
- Auto-handling coverage
- Escalation recall

This would make the policy evaluation less dependent on the final golden set.

---

### 4. Improve retrieval

Experiment with:

- Intent-aware retrieval
- Hybrid lexical + dense retrieval
- Reranking
- Diversity-aware top-k selection
- Filtering historical examples by predicted intent

The objective should be answer-relevant evidence rather than semantic similarity alone.

---

### 5. Improve generation specificity

Require the generator to identify a supported action from the retrieved evidence before producing a response.

If no sufficiently specific action is supported, the system should prefer escalation over a generic response.

This should improve helpfulness while maintaining the current safety constraints.

---

### 6. Add conversation context

The current agent primarily evaluates the incoming customer message.

A production support system should retrieve or pass relevant preceding conversation context, especially for short messages such as:

- "Yes"
- "Thanks"
- "Okay"
- "Still waiting"
- "That's fine"

This would reduce context-dependent generation failures.

---

### 7. Strengthen judge validation

The current independent human-vs-LLM comparison contains 20 examples.

With another week, I would expand this to a larger independently human-rated set and include multiple human raters where possible.

This would provide a more reliable estimate of judge agreement and reduce the uncertainty associated with the small sample.

---

## 12. Reproducibility

The repository contains scripts for:

- Dataset analysis
- Conversation construction
- Golden-set creation
- Intent discovery
- Intent evaluation
- Retrieval-index construction
- Leakage-safe retrieval evaluation
- Escalation scoring
- Escalation evaluation
- Response generation
- Generation safety evaluation
- LLM-as-a-judge evaluation
- Independent human evaluation
- Human-vs-LLM agreement analysis

### Main agent

```bash
python src/agent.py "The item I received is broken."
````

The agent returns:

* Predicted intent
* Intent confidence
* Intent margin
* Retrieved historical examples
* Escalation decision
* Escalation reason
* Generated response when auto-handling is allowed

### Intent evaluation

```bash
python evaluation/baseline_majority.py
python evaluation/evaluate_intent.py
```

The multi-prototype and hybrid approaches are evaluated using the scripts under:

```text
src/intent/
```

### Leakage-safe retrieval

```bash
python src/retrieval/build_leakage_safe_index.py
```

This excludes entire conversations containing golden-set examples before constructing the retrieval corpus.

### Escalation evaluation

```bash
python evaluation/evaluate_escalation.py
```

### Generation evaluation

```bash
python evaluation/run_generation_eval.py
python evaluation/evaluate_generation.py
```

### LLM-as-a-Judge

```bash
python evaluation/llm_judge.py
python evaluation/summarize_llm_judge.py
```

### Human-vs-LLM agreement

```bash
python evaluation/compare_human_llm.py
```

The agreement script computes exact agreement and Cohen's kappa for each evaluation dimension.

---

## 13. Key Results

The current system demonstrates the following:

### Intent classification

* Majority baseline: 28.0% accuracy
* Final hybrid classifier: **51.5% accuracy**
* Final hybrid macro-F1: **0.425**
* Final hybrid weighted-F1: **0.515**

### Retrieval

* Leakage-safe retrieval corpus: **98,232 interactions**
* Mean Top-1 cosine similarity: **0.7644**

This is a semantic-similarity diagnostic, not retrieval accuracy.

### Escalation

* AUTO-HANDLE predicted: **9.0%**
* HUMAN predicted: **91.0%**
* False auto-handling: **5.47%**
* Policy accuracy: **66.0%**

The policy deliberately favors escalation over aggressive automation.

### Generation

On 20 generated AUTO-HANDLE examples:

* Helpfulness: **2.65 / 3**
* Groundedness: **2.95 / 3**
* Correctness: **2.70 / 3**
* Safety: **2.95 / 3**
* Overall: **2.81 / 3 (93.8%)**
* Deterministic safety-clean rate: **100%**

### LLM judge validation

On the same 20 examples independently rated by a human:

* Helpfulness agreement: **95.0%, κ = 0.894**
* Groundedness agreement: **100.0%, κ = 1.000**
* Correctness agreement: **95.0%, κ = 0.886**
* Safety agreement: **100.0%, κ = 1.000**

---

## 14. Limitations

The main limitations are:

1. The golden set contains only 200 examples.
2. The golden set is imbalanced toward delivery and general-support messages.
3. Some examples are inherently ambiguous.
4. Positive feedback and chitchat do not have their own intent.
5. Escalation labels represent policy judgments rather than objective ground truth.
6. Escalation thresholds were not selected using an independent validation set.
7. Retrieval similarity is only a diagnostic and does not measure resolution correctness.
8. Generation quality was evaluated on only 20 examples.
9. Human-vs-LLM judge agreement was also measured on only 20 examples.
10. The human evaluation used one independent human rater.
11. The system has no access to real-time customer account or order information.
12. The system cannot execute refunds, replacements, cancellations, or other transactions.
13. The Gemini free-tier API quota can limit full live reproduction of generation experiments.
14. Historical Twitter support behavior may not represent current Amazon policies or production support procedures.

These limitations mean the reported metrics should be interpreted as evidence about this offline prototype and benchmark rather than production performance.

---

## 15. Decision Summary

The main non-obvious design decisions were:

1. **Choose AmazonHelp as the target brand**
   It had the largest support volume among the candidate brands examined.

2. **Use 10 operational intents rather than the full discovered category set**
   A smaller taxonomy is easier to label consistently and more useful for routing.

3. **Use conversation-level leakage prevention**
   Removing only exact golden examples could leave related messages from the same conversation in retrieval.

4. **Use multiple prototypes per intent**
   A single class centroid can erase important modes within an intent.

5. **Combine prototypes with taxonomy descriptions**
   Historical examples provide empirical language patterns while taxonomy descriptions provide semantic structure.

6. **Treat high-risk intents conservatively**
   Financial, security, missing-package, and refund cases can require account-specific actions.

7. **Use both retrieval similarity and intent margin for escalation**
   Strong retrieval alone does not guarantee correct intent classification.

8. **Prefer escalation when evidence is insufficient**
   The system should not fabricate a resolution simply to increase automation coverage.

9. **Ground generation in historical support examples**
   The model should reproduce supported support behavior rather than inventing policy.

10. **Sanitize retrieved historical examples**
    Historical URLs, identifiers, phone numbers, emails, and handles should not propagate into generated responses.

11. **Add deterministic output safety checks**
    Prompt constraints alone are not sufficient to guarantee that sensitive information will never appear.

12. **Evaluate retrieval separately from generation**
    Retrieval similarity and answer quality measure different properties.

13. **Use an LLM judge but validate it against human ratings**
    Automated evaluation is scalable, but its reliability should be checked against independent human judgments.

14. **Prioritize false-auto reduction over coverage**
    In customer support, an unsafe automatic response can be more damaging than escalating an unnecessary case.

---

## 16. Summary

The resulting system is a conservative retrieval-grounded customer-support agent for AmazonHelp.

The strongest evidence from the evaluation is:

* The hybrid multi-prototype intent classifier outperformed the majority baseline and other tested approaches.
* Conversation-level retrieval leakage was explicitly addressed.
* The escalation policy reduced predicted automation to 9.0% while keeping false auto-handling at 5.47%.
* Generated responses achieved a 100% deterministic safety-clean rate on the 20-example generation sample.
* LLM-judge evaluation showed strong groundedness and safety, with weaker helpfulness caused primarily by generic responses.
* The LLM judge showed high agreement with an independent human rating on the same 20 examples.

The central trade-off is clear:

> **The system is currently optimized for safe, evidence-backed support rather than maximum automation coverage.**

The next highest-value improvements are a larger and better-balanced evaluation set, explicit handling of non-issue messages, calibrated escalation thresholds, stronger retrieval/reranking, and more specific evidence-backed generation.


