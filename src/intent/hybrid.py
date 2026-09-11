import json
import numpy as np

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


GOLDEN_FILE = "data/golden/golden_set.jsonl"
TAXONOMY_FILE = "src/intent/taxonomy.json"

MODEL_NAME = "all-MiniLM-L6-v2"

N_PROTOTYPES = 3
RANDOM_SEED = 42

EXAMPLE_WEIGHT = 0.7
DESCRIPTION_WEIGHT = 0.3


class HybridIntentClassifier:

    def __init__(
        self,
        golden_file=GOLDEN_FILE,
        taxonomy_file=TAXONOMY_FILE,
        model_name=MODEL_NAME,
        n_prototypes=N_PROTOTYPES,
    ):
        self.golden_file = golden_file
        self.taxonomy_file = taxonomy_file
        self.model_name = model_name
        self.n_prototypes = n_prototypes

        self.model = None
        self.prototypes = {}
        self.intent_ids = []
        self.description_embeddings = None

    def load_jsonl(self, path):

        data = []

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                data.append(
                    json.loads(line)
                )

        return data

    def load_taxonomy(self):

        with open(
            self.taxonomy_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    def fit(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            self.model_name
        )

        golden = self.load_jsonl(
            self.golden_file
        )

        taxonomy = self.load_taxonomy()

        # -----------------------------
        # Taxonomy embeddings
        # -----------------------------

        self.intent_ids = [
            item["intent_id"]
            for item in taxonomy
        ]

        descriptions = [
            item["description"]
            for item in taxonomy
        ]

        self.description_embeddings = (
            self.model.encode(
                descriptions,
                normalize_embeddings=True
            )
        )

        # -----------------------------
        # Group labelled examples
        # -----------------------------

        grouped = {}

        for item in golden:

            intent = item["intent"]

            if intent not in grouped:
                grouped[intent] = []

            grouped[intent].append(
                item["customer_text"]
            )

        # -----------------------------
        # Build multiple prototypes
        # -----------------------------

        print(
            f"Building prototypes for "
            f"{len(grouped)} intents..."
        )

        for intent, texts in grouped.items():

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            k = min(
                self.n_prototypes,
                len(embeddings)
            )

            if k == 1:

                prototype = embeddings.mean(
                    axis=0
                )

                prototype /= np.linalg.norm(
                    prototype
                )

                self.prototypes[intent] = [
                    prototype
                ]

                continue

            clustering = KMeans(
                n_clusters=k,
                random_state=RANDOM_SEED,
                n_init=10
            )

            clustering.fit(
                embeddings
            )

            intent_prototypes = []

            for cluster_id in range(k):

                cluster_embeddings = embeddings[
                    clustering.labels_ == cluster_id
                ]

                prototype = cluster_embeddings.mean(
                    axis=0
                )

                prototype /= np.linalg.norm(
                    prototype
                )

                intent_prototypes.append(
                    prototype
                )

            self.prototypes[intent] = (
                intent_prototypes
            )

        print("Hybrid intent classifier ready.")

        return self

    def predict(self, message):

        if self.model is None:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        message_embedding = self.model.encode(
            message,
            normalize_embeddings=True
        )

        message_embedding = np.asarray(
            message_embedding,
            dtype=np.float32
        )

        # -----------------------------
        # Example similarity
        # -----------------------------

        example_scores = {}

        for intent, prototypes in (
            self.prototypes.items()
        ):

            prototype_scores = [
                float(
                    prototype @ message_embedding
                )
                for prototype in prototypes
            ]

            example_scores[intent] = max(
                prototype_scores
            )

        # -----------------------------
        # Taxonomy similarity
        # -----------------------------

        description_scores = (
            self.description_embeddings
            @ message_embedding
        )

        description_score_map = {
            self.intent_ids[i]:
                float(description_scores[i])
            for i in range(
                len(self.intent_ids)
            )
        }

        # -----------------------------
        # Hybrid score
        # -----------------------------

        combined_scores = {}

        for intent in self.intent_ids:

            example_score = example_scores.get(
                intent,
                0.0
            )

            description_score = (
                description_score_map[intent]
            )

            combined_scores[intent] = (
                EXAMPLE_WEIGHT * example_score
                +
                DESCRIPTION_WEIGHT * description_score
            )

        sorted_scores = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_intent = sorted_scores[0][0]
        best_score = sorted_scores[0][1]

        second_score = (
            sorted_scores[1][1]
            if len(sorted_scores) > 1
            else 0.0
        )

        margin = (
            best_score -
            second_score
        )

        # IMPORTANT:
        # This is a ranking signal, not
        # a calibrated probability.
        confidence = (
            0.5 * best_score
            +
            0.5 * margin
        )

        return {
            "intent": best_intent,
            "confidence": float(confidence),
            "best_similarity": float(best_score),
            "margin": float(margin),
            "scores": combined_scores,
        }


if __name__ == "__main__":

    classifier = HybridIntentClassifier()

    classifier.fit()

    while True:

        message = input(
            "\nCustomer message "
            "(q to quit): "
        ).strip()

        if message.lower() == "q":
            break

        result = classifier.predict(
            message
        )

        print(
            f"\nIntent: "
            f"{result['intent']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"Similarity: "
            f"{result['best_similarity']:.3f}"
        )

        print(
            f"Margin: "
            f"{result['margin']:.3f}"
        )