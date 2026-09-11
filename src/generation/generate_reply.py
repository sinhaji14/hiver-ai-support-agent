import json
import os
import re
import sys

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# Configuration
# ============================================================

TOP_K = 5

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GEMINI_MODEL = "gemini-3.6-flash"

# Leakage-safe retrieval index.
# Golden-set conversations have been excluded.
INDEX_DIR = "data/processed/retrieval_index_safe"


# ============================================================
# Load retrieval index
# ============================================================

def load_retrieval_index():
    """
    Load historical customer-support embeddings and metadata.
    """

    embeddings = np.load(
        f"{INDEX_DIR}/embeddings.npy"
    )

    metadata = []

    with open(
        f"{INDEX_DIR}/metadata.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            metadata.append(
                json.loads(line)
            )

    return embeddings, metadata


# ============================================================
# Retrieve historical examples
# ============================================================

def retrieve_examples(
    query,
    embeddings,
    metadata,
    model,
    top_k=TOP_K
):
    """
    Retrieve the most semantically similar historical
    customer-support interactions.

    Because embeddings are normalized, the dot product
    is equivalent to cosine similarity.
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = embeddings @ query_embedding

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        item = metadata[idx]

        results.append(
            {
                "score": float(
                    scores[idx]
                ),
                "customer_text": item[
                    "customer_text"
                ],
                "agent_text": item[
                    "agent_text"
                ],
            }
        )

    return results


# ============================================================
# Sanitize historical text
# ============================================================

def sanitize_text(text):
    """
    Remove obvious sensitive information from historical
    examples before they are shown to the LLM.

    This protects against reproducing customer-specific
    information from the historical support dataset.
    """

    if not text:
        return ""

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        "[URL REMOVED]",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\bwww\.\S+",
        "[URL REMOVED]",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Amazon-style order numbers
    #
    # Example:
    # 404-6800877-5877920
    # --------------------------------------------------------

    text = re.sub(
        r"\b\d{3}-\d{7}-\d{7}\b",
        "[ORDER NUMBER REMOVED]",
        text
    )

    # --------------------------------------------------------
    # Email addresses
    # --------------------------------------------------------

    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL REMOVED]",
        text
    )

    # --------------------------------------------------------
    # Phone numbers
    # --------------------------------------------------------

    text = re.sub(
        r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
        "[PHONE NUMBER REMOVED]",
        text
    )

    # --------------------------------------------------------
    # Twitter-style usernames
    # --------------------------------------------------------

    text = re.sub(
        r"@\w+",
        "@customer",
        text
    )

    return text.strip()


# ============================================================
# Build Gemini prompt
# ============================================================

def build_prompt(
    customer_message,
    retrieved_examples
):
    """
    Build a grounded prompt using sanitized historical
    Amazon customer-support responses.
    """

    context = ""

    for i, example in enumerate(
        retrieved_examples,
        start=1
    ):

        sanitized_customer = sanitize_text(
            example["customer_text"]
        )

        sanitized_agent = sanitize_text(
            example["agent_text"]
        )

        context += f"""
Example {i}

Customer:
{sanitized_customer}

Historical support response:
{sanitized_agent}

Similarity:
{example["score"]:.3f}

---
"""

    prompt = f"""
You are an Amazon customer support agent.

Your task is to write a helpful and concise response to
the customer's message.

Use the historical customer-support examples below as the
primary evidence for your response.

STRICT GROUNDING RULES:

1. Use only information supported by the historical support examples.

2. Do not invent policies.

3. Do not invent refunds, replacements, credits, or compensation.

4. Do not invent delivery dates or guarantees.

5. Do not claim that you performed an action that you cannot perform.

6. NEVER include URLs, links, phone numbers, or email addresses.

7. If a historical response contains a URL, do NOT copy or reproduce it.

8. Do not invent information about the customer's order.

9. Do not expose customer-specific information from historical examples.

10. If the historical examples do not contain enough information
    to safely answer the customer, give a cautious response and
    recommend further assistance.

11. Do not mention these instructions.

12. Do not mention that you used historical examples.

13. Do not mention AI.

14. Keep the response short and professional.

15. Prefer language and actions supported by the historical
    support responses.

16. Do not include historical customer usernames, order numbers,
    phone numbers, email addresses, or other personal identifiers.

Customer message:

{customer_message}

Historical support examples:

{context}

Generate ONLY the final customer-support response.
"""

    return prompt


# ============================================================
# Clean generated response
# ============================================================

def clean_generated_reply(reply):
    """
    Remove URLs, contact information, and common unsupported
    link formats from the generated response.
    """

    if not reply:
        return ""

    # --------------------------------------------------------
    # Remove HTTP / HTTPS URLs
    # --------------------------------------------------------

    reply = re.sub(
        r"https?://\S+",
        "",
        reply,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Remove www URLs
    # --------------------------------------------------------

    reply = re.sub(
        r"\bwww\.\S+",
        "",
        reply,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Remove markdown links while keeping visible text
    #
    # [Amazon Help](https://amazon.com)
    # becomes:
    # Amazon Help
    # --------------------------------------------------------

    reply = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        reply
    )

    # --------------------------------------------------------
    # Remove HTML links while keeping visible text
    # --------------------------------------------------------

    reply = re.sub(
        r"<a\b[^>]*>(.*?)</a>",
        r"\1",
        reply,
        flags=re.IGNORECASE | re.DOTALL
    )

    # --------------------------------------------------------
    # Remove email addresses
    # --------------------------------------------------------

    reply = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "",
        reply
    )

    # --------------------------------------------------------
    # Remove phone numbers
    # --------------------------------------------------------

    reply = re.sub(
        r"\b(?:\+?\d[\d\s().-]{7,}\d)\b",
        "",
        reply
    )

    # --------------------------------------------------------
    # Remove excessive whitespace
    # --------------------------------------------------------

    reply = re.sub(
        r"[ \t]+",
        " ",
        reply
    )

    # --------------------------------------------------------
    # Clean spaces before punctuation
    # --------------------------------------------------------

    reply = re.sub(
        r"\s+([,.!?])",
        r"\1",
        reply
    )

    # --------------------------------------------------------
    # Clean excessive blank lines
    # --------------------------------------------------------

    reply = re.sub(
        r"\n\s*\n+",
        "\n",
        reply
    )

    return reply.strip()


# ============================================================
# Generate response
# ============================================================

def generate_reply(
    customer_message,
    retrieved_examples
):
    """
    Generate a grounded customer-support response
    using Gemini and sanitized historical examples.
    """

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found "
            "in the environment."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = build_prompt(
        customer_message,
        retrieved_examples
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    cleaned_reply = clean_generated_reply(
        response.text.strip()
    )

    return cleaned_reply


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            'Usage:\n'
            'python src/generation/generate_reply.py '
            '"your customer message"'
        )

        raise SystemExit(1)

    customer_message = sys.argv[1]

    # --------------------------------------------------------
    # Load retrieval index
    # --------------------------------------------------------

    print(
        "Loading retrieval index..."
    )

    embeddings, metadata = (
        load_retrieval_index()
    )

    print(
        f"Loaded {len(metadata)} "
        "historical interactions."
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # Retrieve examples
    # --------------------------------------------------------

    print(
        "\nRetrieving similar historical cases..."
    )

    examples = retrieve_examples(
        customer_message,
        embeddings,
        metadata,
        model,
        top_k=TOP_K
    )

    # --------------------------------------------------------
    # Display sanitized retrieved cases
    # --------------------------------------------------------

    print(
        "\nTop historical examples:"
    )

    for i, example in enumerate(
        examples,
        start=1
    ):

        print(
            f"\n{i}. Similarity: "
            f"{example['score']:.3f}"
        )

        print(
            "Customer: "
            + sanitize_text(
                example["customer_text"]
            )
        )

        print(
            "Agent: "
            + sanitize_text(
                example["agent_text"]
            )
        )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    print(
        "\nGenerating grounded reply..."
    )

    reply = generate_reply(
        customer_message,
        examples
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "GENERATED REPLY"
    )

    print(
        "=" * 60
    )

    print(reply)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()