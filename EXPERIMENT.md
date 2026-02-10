# Bioplausible Discovery Protocol: A Guide to Finite-Patience Research

> *"Fail fast, fail cheap, and scale only what works."*

This document defines the methodology for harvesting scientific knowledge about bioplausible learning algorithms. It is explicitly designed for researchers with **limited time and compute**, ensuring that every minute of waiting yields maximum insight.

## 🔬 The Core Philosophy: Return on Compute (RoC)

In this project, we treat **Researcher Attention** and **Compute Cycles** as our most valuable resources. We strictly forbid "blind scaling"—running expensive overnight jobs before verifying basic functionality.

Instead, we use a **Tiered Discovery Funnel**. An algorithm must earn the right to consume more compute by passing specific success criteria at cheaper tiers.

## 📊 The Tiered Discovery Funnel

We have implemented this protocol directly into the **Bioplausible Studio** UI. You will see a **"Discovery Tier"** selector in the Experiment tab.

### 🧱 Tier 1: SMOKE (The Sanity Check)
*Goal: Does the code run? Does it learn anything at all?*
-   **Cost**: ~1 minute per trial.
-   **Config**: 2 Epochs, Tiny Model (64 units), Tiny Subset (500 samples).
-   **Use Case**: Debugging new algorithms, checking for NaNs, verifying pipeline integrity.
-   **Success Criteria**: Accuracy > Random Chance + epsilon.
-   **UI Action**: Select **"Smoke (1 min)"** in Search Tab.

### ⛵ Tier 2: SHALLOW (The Breadth Search)
*Goal: Is this idea worth my time?*
-   **Cost**: ~5-10 minutes per trial.
-   **Config**: 5 Epochs, Small Model (128 units), Subset (2000 samples).
-   **Use Case**: Sweeping wide hyperparameter ranges (e.g., learning rates from $10^{-5}$ to $10^{-1}$).
-   **Why it works**: If an algorithm is robust, it should show *signs of life* (learning curve slope) even in this shallow regime.
-   **Success Criteria**: Achieves >80% of the Backprop baseline's performance on the same config.
-   **UI Action**: Select **"Shallow (10 min)"** in Search Tab.

### ⚖️ Tier 3: STANDARD (The Fair Comparison)
*Goal: Can it compete with Backprop?*
-   **Cost**: ~30-60 minutes per trial.
-   **Config**: 30 Epochs, Medium Model (256 units), Full Dataset.
-   **Use Case**: Getting the "official" number for the leaderboard.
-   **Why it works**: This is the standard "Academic Benchmark" setting.
-   **Success Criteria**: **Gap to Baseline < 5%**.
-   **UI Action**: Select **"Standard (1 hr)"** in Search Tab.

### 🚢 Tier 4: DEEP (The Scientific Proof)
*Goal: What is the absolute limit?*
-   **Cost**: Overnight (4+ hours).
-   **Config**: 100+ Epochs, Large Model, Full Dataset + Augmentation.
-   **Use Case**: Producing the final paper / report figures.
-   **Why it works**: Once we know the optimal hyperparameters from Standard tier, we invest heavy compute to maximize performance.
-   **Success Criteria**: State-of-the-Art parity or biological advantage (e.g. high energy efficiency).
-   **UI Action**: Select **"Deep (Overnight)"** in Search Tab.

---

## 🔄 The Experimenter's Loop

1.  **Hypothesize**: "Maybe Hebbian learning needs a higher learning rate."
2.  **Probe (Shallow)**: Go to **Experiments Tab**. Select `EqProp`, select `Shallow`, and run 10 trials.
    *   *Result*: Fast feedback. You see results in minutes.
3.  **Analyze (Radar)**: Go to **Radar View**. Do the high-LR points crash? Or do they form a high-accuracy cluster?
    *   *Insight*: "Ah, it works, but only if momentum is low."
4.  **Invest (Standard)**: Go back to **Experiments Tab**. Select `Standard`. Narrow the search space based on your Shallow insights. Run 20 trials.
5.  **Verify (Leaderboard)**: Go to **Leaderboard**. Set **Tier Filter** to "Standard". Compare `EqProp` vs `Backprop`.
    *   *Check*: Is the gap < 5%?
6.  **Scale (Production)**: Click the best trial card -> **"🚀 Train This Configuration"**. Monitor biological metrics in the Train Tab.

## 🚫 Preventing Unfair Comparisons

The Leaderboard now includes a **Tier Filter**.
-   **Rule**: NEVER compare a "Smoke" run to a "Standard" run.
-   **Why**: A 2-epoch run will always have lower accuracy than a 30-epoch run. Mixing them ruins the rankings.
-   **Tooling**: The UI defaults to showing all, but you should always select a specific Tier in the dropdown when analyzing competition.

## 🧠 Continuous Knowledge Accumulation

We store every trial, no matter the tier. This builds a "Map of the Landscape".
-   **Smoke Trials** map the "Dead Zones" (where models crash).
-   **Shallow Trials** map the "Promising Regions".
-   **Standard Trials** map the "Peak Performance".

By visualizing *all* of these in Radar View (filtered by Tier), you gain deep intuition about the algorithm's behavior without wasting weeks of compute.
