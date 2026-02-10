"""
Example: Using EqProp with Scikit-Learn

This script demonstrates how to use the EqPropClassifier with the standard
Scikit-Learn API (fit/predict) on a simple dataset.
"""

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from bioplausible.sklearn_interface import EqPropClassifier


def main():
    print("Loading Digits dataset...")
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # Normalize to [0, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training EqPropClassifier on {len(X_train)} samples...")
    clf = EqPropClassifier(
        hidden_dim=128,
        epochs=15,
        steps=20,
        learning_rate=0.005,
        random_state=42,
        device="cpu",  # Use CPU for this small example
    )

    clf.fit(X_train, y_train)
    print("Training complete.")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
