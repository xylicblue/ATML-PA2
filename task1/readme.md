# 🌐 Domain Adaptation on the PACS Dataset

This project investigates multiple **Domain Adaptation (DA)** strategies using the **PACS dataset** — a widely used benchmark for evaluating visual recognition models under **distributional shifts** across distinct visual domains.

The goal is to assess how effectively various adaptation techniques improve **generalization from a labeled source domain** to an **unlabeled target domain**.

---

## 📘 Table of Contents

1. [Overview](#overview)  
2. [Methodology](#methodology)  
   - [1. Source-Only Baseline](#1-source-only-baseline)  
   - [2. Domain-Adversarial Neural Network (DANN)](#2-domain-adversarial-neural-network-dann)  
   - [3. Deep Adaptation Network (DAN)](#3-deep-adaptation-network-dan)  
   - [4. Conditional Domain-Adversarial Network (CDAN)](#4-conditional-domain-adversarial-network-cdan)  
   - [5. Self-Training via Pseudo-Labeling](#5-self-training-via-pseudo-labeling)  
   - [6. Concept Shift Analysis](#6-concept-shift-analysis)  
   - [7. Visualizations](#7-visualizations)  
3. [Key Experimental Parameters](#key-experimental-parameters)  
4. [Summary](#summary)  
5. [Setup & Requirements](#setup--requirements)  
6. [Citation](#citation)  

---

## 🧠 Overview

The notebook implements and compares **five domain adaptation methods** — ranging from simple baselines to advanced adversarial and semi-supervised techniques.  
Each method is trained, evaluated, and analyzed independently on both **source** and **target** domains.

All experiments use **ResNet-50** as the base architecture, pretrained on ImageNet.

---

## 🧪 Methodology

### 1. 🧩 Source-Only Baseline

**Description:**  
A standard ResNet-50 model is trained *only* on labeled source domain data, with no adaptation.

**Purpose:**  
Serves as a baseline to measure the magnitude of domain shift — the gap between source and target performance.

**Outputs:**
- Source and target test accuracies  
- Model checkpoint: `source_only_model.pth`

---

### 2. ⚔️ Domain-Adversarial Neural Network (DANN)

**Description:**  
Implements the **DANN** architecture featuring:
- A **domain discriminator** alongside the label predictor  
- A **Gradient Reversal Layer (GRL)** that enforces domain invariance

**Purpose:**  
To align source and target feature distributions *adversarially*.  
The GRL reverses gradients from the discriminator, encouraging domain-invariant features.

**Outputs:**
- Label and domain loss logs  
- Source and target domain accuracies  
- Model checkpoint: `dann_model_finetuned.pth`

---

### 3. 🔗 Deep Adaptation Network (DAN)

**Description:**  
Implements **DAN**, which aligns source and target representations using the **Maximum Mean Discrepancy (MMD)** criterion.

**Purpose:**  
To minimize the statistical distance between source and target features in **Reproducing Kernel Hilbert Space (RKHS)**.

**Outputs:**
- Training logs (label + MMD losses)  
- Source and target accuracies  
- Model checkpoint: `dan_model_finetuned_lambda_X.pth`

---

### 4. 🌀 Conditional Domain-Adversarial Network (CDAN)

**Description:**  
An extension of DANN that **conditions the domain discriminator** on both:
- The **feature representations**, and  
- The **class predictions**

**Purpose:**  
To enable *class-aware domain alignment*, ensuring that semantically similar classes remain well-aligned across domains.

**Outputs:**
- Training logs and evaluation metrics  
- Source and target domain accuracies  
- Model checkpoint: `cdan_model_finetuned_lambda_X.pth`

---

### 5. 🧩 Self-Training via Pseudo-Labeling

**Description:**  
A **semi-supervised** adaptation strategy:
- The model generates **pseudo-labels** for target samples with high prediction confidence.
- Confident pseudo-labeled samples are mixed with source data for fine-tuning.

**Purpose:**  
To exploit **unlabeled target data** for improved adaptation without explicit target supervision.

**Outputs:**
- Number of pseudo-labels generated  
- Final target accuracy  
- Updated model weights

---

### 6. 🔍 Concept Shift Analysis

**Description:**  
Tests model robustness under *simulated distribution shifts* (e.g., label or class imbalance).

**Purpose:**  
Evaluate generalization under more realistic conditions:
- **Missing Class (Label Shift):** Excludes one class from the target domain.  
- **Rare Class (Imbalanced Data):** Reduces the occurrence of a specific target class.

**Outputs:**
- Per-model accuracy scores on modified target datasets  
- Confusion matrix visualizations for rare class scenarios

---

### 7. 🎨 Visualizations

**Description:**  
t-SNE visualizations of learned features across domains.

**Purpose:**
- **Domain Alignment:** Visual overlap between source (blue) and target (red) feature spaces  
- **Class Separation:** Cluster tightness of same-class samples across domains

**Outputs:**
- t-SNE scatter plots for each model (Source vs Target feature distributions)

---

## ⚙️ Key Experimental Parameters

| Parameter | Description |
|------------|-------------|
| **SOURCE_DOMAIN / TARGET_DOMAIN** | Source and target domains (e.g., `art_painting`, `cartoon`) |
| **NUM_EPOCHS** | Total number of training epochs |
| **LEARNING_RATE** | Learning rate for optimization |
| **LAMBDA_WEIGHT** | Balances classification and adaptation losses (DAN/CDAN) |
| **CONFIDENCE_THRESHOLD** | Minimum confidence for pseudo-labeling |

---

## 📈 Summary

This project delivers a **comprehensive empirical study** of key **Domain Adaptation** methods:

| Method | Adaptation Type | Key Idea | Strength |
|---------|----------------|-----------|-----------|
| **Source-Only** | None | Baseline for domain shift | Simple, no adaptation |
| **DANN** | Adversarial | Gradient reversal to align domains | Robust invariance |
| **DAN** | Distribution-based | MMD alignment in feature space | Explicit feature matching |
| **CDAN** | Conditional adversarial | Class-aware domain alignment | Fine-grained adaptation |
| **Pseudo-Labeling** | Semi-supervised | Confidence-based self-training | Leverages unlabeled data |

**Outcome:**  
Through quantitative metrics and t-SNE visualization, the notebook highlights how adversarial, distributional, and semi-supervised techniques mitigate domain shift in the **PACS** dataset.

---

## 🛠️ Setup & Requirements

### Dependencies
- Python 3.9+
- PyTorch ≥ 2.0
- torchvision
- numpy
- scikit-learn
- matplotlib
- tqdm
- seaborn
- datasets (Hugging Face)

### Installation
```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm seaborn datasets
