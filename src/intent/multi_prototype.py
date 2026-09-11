import json
import numpy as np

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


GOLDEN_FILE = "data/golden/golden_set.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

N_PROTOTYPES = 3
RANDOM_SEED = 42


class MultiPrototypeClassifier:

    def __init__(
        self,
        golden_file=GOLDEN_FILE,
        model_name=MODEL_NAME,
        n_prototypes=N_PROTOTYPES,
    ):
        self.golden_file = golden_file
        self.model_name = model_name
        self.n_prototypes = n_prototypes

        self.model = None
        self.prototypes = {}

    def load_data(self):
        data = []

        with open(
            self.golden_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                data.append(
                    json.loads(line)
                )

        return data

    def fit(self):
        """
        Build multiple semantic prototypes for each intent
        using the complete labelled golden set.

        This method is intended for inference after the
        evaluation has already been completed using cross-validation.
        """

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            self.model_name
        )

        data = self.load_data()

        grouped = {}

        for item in data:

            intent = item["intent"]

            if intent not in grouped:
                grouped[intent] = []

            grouped[intent].append(
                item["customer_text"]
            )

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

                prototype = prototype / np.linalg.norm(
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

                prototype = prototype / np.linalg.norm(
                    prototype
                )

                intent_prototypes.append(
                    prototype
                )

            self.prototypes[intent] = (
                intent_prototypes
            )

        print("Prototype model ready.")

        return self

    def predict(self, message):
        """
        Predict the customer's intent.

        Returns:
            intent
            confidence
            best_similarity
            margin
        """

        if self.model is None:
            raise RuntimeError(
                "Classifier has not been fitted. "
                "Call fit() first."
            )

        message_embedding = self.model.encode(
            message,
            normalize_embeddings=True
        )

        scores = {}

        for intent, prototypes in (
            self.prototypes.items()
        ):

            prototype_scores = [
                float(
                    prototype @ message_embedding
                )
                for prototype in prototypes
            ]

            scores[intent] = max(
                prototype_scores
            )

        sorted_scores = sorted(
            scores.items(),
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

        # This is a heuristic confidence signal,
        # NOT a calibrated probability.
        confidence = (
            0.5 * best_score +
            0.5 * margin
        )

        return {
            "intent": best_intent,
            "confidence": float(confidence),
            "best_similarity": float(best_score),
            "margin": float(margin),
        }


if __name__ == "__main__":

    classifier = MultiPrototypeClassifier()

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