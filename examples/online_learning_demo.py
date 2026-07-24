"""
Online Learning Demo with EqPropClassifier

Demonstrates how to use the EqPropClassifier in an online learning setting
using the `partial_fit` method. This allows processing data streams or
datasets too large to fit in memory.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from bioplausible.datasets import get_vision_dataset
from bioplausible.sklearn_interface import EqPropClassifier


def main():
    print("Loading Digits dataset (sklearn)...")
    # Load dataset as TensorDataset
    train_dataset = get_vision_dataset("digits", train=True, flatten=True)
    test_dataset = get_vision_dataset("digits", train=False, flatten=True)

    # Extract tensors for sklearn compatibility
    X_train = train_dataset.tensors[0].numpy()
    y_train = train_dataset.tensors[1].numpy()
    X_test = test_dataset.tensors[0].numpy()
    y_test = test_dataset.tensors[1].numpy()

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Initialize Classifier
    # We choose "EqProp MLP" from the registry.
    clf = EqPropClassifier(
        model_name="EqProp MLP",
        hidden_dim=128,
        steps=20,  # Fewer steps for speed
        learning_rate=0.01,
        device="cpu",  # Force CPU for small dataset
        random_state=42,
    )

    # Online Learning Simulation
    batch_size = 32
    n_batches = len(X_train) // batch_size
    classes = np.unique(y_train)

    print("\nStarting Online Learning Stream...")
    print("-" * 50)
    print(f"{'Batch':<10} | {'Test Acc':<10}")
    print("-" * 50)

    accuracies = []

    # Shuffle for streaming simulation
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_stream = X_train[indices]
    y_stream = y_train[indices]

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_stream[start:end]
        y_batch = y_stream[start:end]

        # Incremental update
        # classes arg is only needed for the first call, but safe to pass always
        clf.partial_fit(X_batch, y_batch, classes=classes)

        # Monitor progress every 5 batches
        if (i + 1) % 5 == 0:
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            print(f"{i + 1:<10} | {acc:.4f}")

    print("-" * 50)
    print(f"Final Test Accuracy: {accuracies[-1]:.4f}")

    # Plot Learning Curve
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(5, n_batches + 1, 5), accuracies, marker="o")
        plt.title("Online Learning Curve (EqProp MLP on Digits)")
        plt.xlabel("Batches Processed")
        plt.ylabel("Test Accuracy")
        plt.grid(True)
        plt.savefig("online_learning_curve.png")
        print("Learning curve saved to 'online_learning_curve.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()
