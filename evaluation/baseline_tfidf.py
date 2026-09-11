import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PATH = "data/golden/golden_set.jsonl"

# Load golden set
with open(PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

texts = [x["customer_text"] for x in data]
labels = [x["intent"] for x in data]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifier
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
)

model.fit(X_train_tfidf, y_train)

# Predict
predictions = model.predict(X_test_tfidf)

# Metrics
accuracy = accuracy_score(y_test, predictions)

print("TF-IDF + Logistic Regression")
print("============================")
print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))