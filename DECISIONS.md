# Decision Log

## 1. Selected AmazonHelp as the target brand

**Decision:** Build the AI support agent for AmazonHelp.

**Why:**
- AmazonHelp has the largest support volume in the dataset.
- 169,840 support tweets are associated with AmazonHelp.
- 169,287 (99.67%) of these have a parent tweet.
- 154,976 parent tweets can be matched to inbound customer messages.
- The volume provides enough historical examples for intent discovery,
  retrieval, and evaluation.

**Alternatives considered:**
- AppleSupport
- Uber_Support
- SpotifyCares
- Delta

---

## 2. Use customer-to-support interactions as the retrieval unit

**Decision:** Represent historical support experience as customer message
paired with the corresponding AmazonHelp response.

**Why:** The generation system needs examples of both the customer problem
and how support historically responded.

---

## 3. Use a 10-intent taxonomy

**Decision:** Consolidate the discovered support themes into 10 operational
intents.

**Why:** A smaller taxonomy is easier to evaluate and reduces excessive
overlap between closely related categories.

---

## 4. Create a 200-example hand-labelled golden set

**Decision:** Label 200 interactions manually.

**Why:** The assignment requires 150–250 hand-labelled examples. 200 gives
enough coverage while keeping annotation manageable.

---

## 5. Define intent using the customer's primary support need

**Decision:** Label the intent based on the issue most directly addressed
by the customer's request and historical support response.

**Why:** Some tweets contain multiple topics. A primary-intent rule makes
annotation more reproducible.

---

## 6. Split intent evaluation by stratified cross-validation

**Decision:** Use 5-fold stratified cross-validation for the main supervised
intent comparisons.

**Why:** It reduces dependence on a single train/test split and provides a
more stable estimate on the small golden set.

---

## 7. Use sentence-transformer embeddings

**Decision:** Use `all-MiniLM-L6-v2` embeddings for semantic similarity.

**Why:** The system needs to match semantically similar customer issues even
when their wording differs.

---

## 8. Use multiple prototypes per intent

**Decision:** Represent each intent using up to three semantic prototypes
rather than one class centroid.

**Why:** A single centroid can collapse several distinct ways customers
describe the same issue. Multiple prototypes preserve more variation.

---

## 9. Combine historical prototypes with taxonomy descriptions

**Decision:** Use a hybrid classifier combining multi-prototype similarity
with taxonomy-description similarity.

**Why:** Historical examples capture real customer language while taxonomy
descriptions provide a semantic prior for the intended category.

---

## 10. Exclude complete conversations containing golden examples

**Decision:** Remove entire conversations containing golden examples from
the historical retrieval corpus.

**Why:** Removing only the exact golden message could still allow a nearly
identical turn from the same conversation to appear in retrieval and cause
evaluation leakage.

---

## 11. Use Top-5 historical examples for generation

**Decision:** Retrieve the five most similar historical support cases.

**Why:** Top-5 provides multiple precedents without making the generation
prompt unnecessarily large.

---

## 12. Apply conservative grounding rules

**Decision:** The generator must not invent refunds, replacements,
compensation, dates, order information, policies, or completed actions.

**Why:** Customer-support hallucinations can cause operational and trust
problems. The system should prefer a cautious support handoff when evidence
is insufficient.

---

## 13. Sanitize historical examples before generation

**Decision:** Remove URLs, order numbers, email addresses, phone numbers,
and Twitter-style handles from historical examples.

**Why:** Historical support data can contain customer-specific information.
Sanitization reduces the risk of reproducing sensitive identifiers.

---

## 14. Use conservative escalation for high-risk cases

**Decision:** Financial, account-security, missing-package, and
return/refund cases are treated as high risk and generally escalated.

**Why:** These cases may require account-specific investigation or actions
that cannot safely be inferred from historical text alone.

---

## 15. Evaluate generation with both automated checks and human review

**Decision:** Use automated sensitive-information checks plus human ratings
of helpfulness, groundedness, and safety.

**Why:** Automated checks can detect concrete safety failures, but they cannot
reliably determine whether a response is useful or adequately grounded.
A human reference is therefore required for qualitative evaluation.

---

## Evaluation note

The escalation labels are human policy judgments rather than objective
ground truth. Therefore, escalation metrics should be interpreted as
agreement with the human escalation policy.

The generation evaluation is currently based on a small sample because the
Gemini free-tier API quota limited the number of generation requests.