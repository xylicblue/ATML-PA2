# Probing CLIP for Domain Adaptation: Prompts, Gradients, and Open-Set Generalization

## Overview

This project presents a **research-oriented investigation** into leveraging **pre-trained vision–language models**, specifically **OpenAI’s CLIP**, for **Domain Adaptation (DA)** and **Domain Generalization (DG)**.
The central focus is on **prompt learning** — a parameter-efficient tuning technique — as a mechanism to control and adapt CLIP’s representations across diverse visual domains.

Experiments are conducted on the **PACS dataset**, which contains four visually distinct domains — *Photo, Art Painting, Cartoon,* and *Sketch* — representing the same object categories. This makes PACS a suitable benchmark for studying cross-domain generalization.

---

## Research Objectives

This repository contains four sequential analyses, each exploring a different dimension of CLIP’s adaptability and generalization:

1. **Baseline Evaluation** — Assessing CLIP’s zero-shot and fine-tuned performance.
2. **Prompt Learning for Unsupervised Domain Adaptation (UDA)** — Adapting CLIP using learnable text prompts.
3. **Gradient Conflict Analysis** — Examining conflicts during multi-domain optimization.
4. **Open-Set Generalization** — Evaluating how prompt tuning affects CLIP’s out-of-distribution recognition capabilities.

---

## Notebook's Overview

### 1. CLIP Zero-Shot vs. Fine-Tuning Analysis

**Objective:**
Establish baseline performance metrics for CLIP on the PACS dataset to evaluate its intrinsic generalization versus fine-tuned specialization.

**Methodology:**

* **Zero-Shot Evaluation:** Measures CLIP’s accuracy across all four domains using domain-specific text prompts (e.g., *“a sketch of a …”*).
* **Linear Probing:** Uses CLIP’s frozen image encoder to extract features. A linear classifier is trained on three source domains (*photo, art, cartoon*) and tested on the unseen *sketch* domain.

**Expected Output:**

* Zero-shot accuracy for each domain
* Linear probe accuracy on the target domain
* Comparative analysis of generalization vs. adaptation trade-off

---

### 2. Unsupervised Domain Adaptation via Prompt Learning


**Objective:**
Implement and analyze **prompt tuning** for **Unsupervised Domain Adaptation**, utilizing labeled source data and unlabeled target data.

**Methodology:**

* **PromptTunedCLIP Model:** Defines learnable text prompts (context tokens), inspired by the CoOp framework.
* **Source-Only Training:** Trains prompts using labeled source data (*photo*) and evaluates on *sketch*.
* **UDA with Pseudo-Labeling:** Incorporates unlabeled target data by generating pseudo-labels for high-confidence samples, optimizing a combined supervised + unsupervised loss.

**Expected Output:**
A summary table comparing:

* Zero-Shot CLIP
* Source-Only Prompt Tuning (CoOp)
* UDA with Pseudo-Labels

This demonstrates whether unlabeled target data improves adaptation performance.

---

### 3. Gradient Conflict and Alignment Analysis

**Objective:**
Investigate **gradient conflicts** that occur when training on multiple domains simultaneously — a key challenge in multi-domain adaptation.

**Methodology:**

* Train a prompt-tunable CLIP jointly on two domains (*photo* and *cartoon*).
* Periodically compute gradients of the prompt parameters for each domain independently.
* Calculate cosine similarity between gradients:

  * **Positive:** aligned objectives
  * **Negative:** conflicting objectives

**Expected Output:**

* A plot of gradient cosine similarity over training steps
* A red dashed reference line (`y=0`) marking the conflict threshold
* Insights into when and how domains diverge during learning

---

### 4. Open-Set and Generalization Analysis


**Objective:**
Evaluate the effect of prompt tuning on CLIP’s **open-set recognition** — its ability to identify unseen classes or out-of-distribution samples.

**Methodology:**

* **Open-Set Experiment:** Split PACS into *seen* and *unseen* classes. Train prompts only on *seen* classes.
* **Evaluation Metrics:** Compute **AUROC** and **FPR** for distinguishing seen vs. unseen data.
* **Prompt Embedding Similarity:** Fine-tune prompts on different domains (*photo* and *sketch*) and measure cosine similarity between their learned embeddings.

**Expected Output:**

* Comparative open-set detection metrics for Zero-Shot and Prompt-Tuned CLIP
* Cosine similarity scores between prompts from different domains, revealing the domain gap in CLIP’s text encoder space

---

## Dependencies

Before running the notebooks, ensure the following dependencies are installed:

* Python **3.8+**
* Jupyter Notebook or JupyterLab
* PyTorch and Torchvision
* [OpenAI CLIP](https://github.com/openai/CLIP)
* Scikit-learn
* NumPy
* Pillow (PIL)
* Matplotlib
* tqdm


> **Note:** Some notebook cells use hardcoded paths (e.g., `../task1/pacs_data/pacs_data/`).
> Adjust the paths if your dataset is located elsewhere.

---

## Summary

This repository provides a **systematic exploration of CLIP’s adaptability** across domains, with emphasis on prompt learning as a control mechanism for domain shift.
By analyzing **unsupervised adaptation**, **gradient conflicts**, and **open-set behavior**, the project highlights the trade-offs between **adaptation performance** and **generalization robustness** in large vision–language models.


