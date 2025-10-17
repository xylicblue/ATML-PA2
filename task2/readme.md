# 🌍 Domain Generalization with Invariant & Robust Learning

This project explores **Domain Generalization (DG)** techniques to train models that are **robust to domain shifts** — i.e., capable of generalizing to unseen domains without adaptation at test time.

The experiments compare four DG strategies using the **PACS dataset**, with the following setup:

- **Source Domains:** `art_painting`, `cartoon`, `photo`
- **Unseen Target Domain:** `sketch`

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Setup and Data Preparation](#1-setup-and-data-preparation)
3. [Model Architecture](#2-model-architecture)
4. [Experiment 1: Empirical Risk Minimization (ERM)](#3-experiment-1-empirical-risk-minimization-erm)
5. [Experiment 2: Invariant Risk Minimization (IRM)](#4-experiment-2-invariant-risk-minimization-irm)
6. [Experiment 3: Group Distributionally Robust Optimization (Group-DRO)](#5-experiment-3-group-distributionally-robust-optimization-group-dro)
7. [Experiment 4: Sharpness-Aware Minimization (SAM)](#6-experiment-4-sharpness-aware-minimization-sam)
8. [Analysis and Visualization](#7-analysis-and-visualization)
9. [Results Summary](#results-summary)

---

## 🧠 Overview

This notebook implements and compares **four Domain Generalization strategies**:

| Method        | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| **ERM**       | Standard baseline minimizing empirical risk.                       |
| **IRM**       | Encourages domain-invariant feature representations.               |
| **Group DRO** | Optimizes for worst-case performance across domains.               |
| **SAM**       | Seeks flat minima in the loss landscape for better generalization. |

All experiments are conducted using **ResNet-50** (pre-trained on ImageNet) as the backbone.

---

## 1. ⚙️ Setup and Data Preparation

### Dataset

- **PACS** dataset (loaded via Hugging Face Datasets library).
- Contains four domains: `art_painting`, `cartoon`, `photo`, and `sketch`.

### Data Transforms

- **train_transform:** Includes `RandomResizedCrop`, `RandomHorizontalFlip`, and `ColorJitter`.
- **eval_transform:** Deterministic resizing and normalization for the target domain.

### Domain Splitting

Each domain is filtered into a separate dataset.

- Source domains (`art_painting`, `cartoon`, `photo`) → Training
- Target domain (`sketch`) → Evaluation only

### DataLoaders

- **ERM & SAM:** Combined all source domains using `ConcatDataset`.
- **IRM & Group DRO:** Created separate per-domain loaders.
- **Target (sketch):** Dedicated `test_loader` for evaluation.

---

## 2. 🧩 Model Architecture

- **Backbone:** `ResNet-50` pre-trained on ImageNet (`torchvision.models.resnet50`).
- **Classifier Head:** Replaced with `nn.Linear(2048, 7)` for 7 PACS classes.
- **Training:** End-to-end fine-tuning for all experiments.

---

## 3. 🧪 Experiment 1: Empirical Risk Minimization (ERM)

**ERM** serves as the baseline DG approach.

- **Training:** Model trained on the combined source domains.
- **Loss:** Standard Cross-Entropy loss.
- **Evaluation Metrics:**
  - Target domain accuracy (on `sketch`)
  - Per-source domain accuracy
  - Average and worst-case source domain accuracy

---

## 4. 🧬 Experiment 2: Invariant Risk Minimization (IRM)

**IRMv1** aims to learn domain-invariant representations by penalizing domain-specific variations.

### IRM Penalty

Implemented via:
\[
\text{IRM penalty} = \| \nabla\_{\text{dummy scale}} \text{loss}(\text{scale} \cdot f(x), y) \|^2
\]

### Training

- Batches are drawn from each source domain.
- Final loss:
  \[
  \text{Total Loss} = \text{mean(ERM losses)} + \lambda \times \text{mean(IRM penalties)}
  \]
- Models are trained with varying **penalty weights** (e.g., 2, 5, 20, 100).

### Evaluation

Same as ERM — accuracy breakdown per domain + unseen target accuracy.

---

## 5. ⚖️ Experiment 3: Group Distributionally Robust Optimization (Group DRO)

**Group DRO** explicitly optimizes for worst-case performance across domains.

### Training

1. Compute per-domain losses.
2. Apply softmax weighting to emphasize higher-loss domains.
3. Compute weighted loss:
   \[
   \text{Weighted Loss} = \sum w_i \times \text{Loss}\_i
   \]
4. Backpropagate using the weighted loss.

### Evaluation

Consistent with the ERM and IRM evaluation protocols.

---

## 6. 🌄 Experiment 4: Sharpness-Aware Minimization (SAM)

**SAM** improves generalization by finding flatter minima in the loss landscape.

### Optimizer

Custom `SAM` optimizer wrapping a base optimizer (e.g., AdamW).

### Training Process

1. Compute gradient and perform an _ascent_ step (`optimizer.first_step()`).
2. Compute new gradient at perturbed weights.
3. Perform the final update (`optimizer.second_step()`).

### Evaluation

Same as other methods — per-domain and target-domain accuracy.

---

## 7. 📊 Analysis and Visualization

### 🪨 Loss Landscape Visualization

- Compares **ERM** vs **SAM** loss basins.
- Loss values plotted over a 2D plane in parameter space using filter-wise normalized directions.
- **Goal:** Demonstrate SAM’s flatter, wider minima.

### 🌈 Feature Space Visualization (t-SNE)

- Extracted features from the penultimate layer (`avgpool`) of each model.
- Used **t-SNE** to project to 2D.
- Plots colored by class, shaped by domain → reveals class separability & domain alignment.

### 📈 Quantitative Flatness (Gradient Similarity)

- Computed gradient vectors for each source domain.
- Measured **cosine similarity** between gradients.
- Higher similarity ⇒ flatter, domain-general loss basin.

---
