# 🧭 Domain Adaptation and Generalization Experiments

This repository contains the implementation and analysis of **Unsupervised Domain Adaptation (UDA)**, **Domain Generalization (DG)**, and **Prompt-based Adaptation** experiments using both traditional and CLIP-based approaches.  
The project explores how models can be trained to generalize across domains under various settings, following three structured tasks.

---

## 📘 Overview

Modern deep learning models often struggle when faced with **domain shifts** — differences in data distribution between training and testing environments.  
This project systematically investigates three aspects of this challenge:

1. **Unsupervised Domain Adaptation (UDA)** – adapting a model from a labeled source domain to an unlabeled target domain.  
2. **Domain Generalization (DG)** – learning from multiple source domains to perform well on an unseen target domain.  
3. **Prompt-based Domain Adaptation and Generalization with CLIP** – leveraging large pre-trained vision-language models for cross-domain transfer using prompt learning and gradient alignment.

---

## 🧩 Task 1: Unsupervised Domain Adaptation (UDA)

**Goal:** Adapt a model trained on a labeled source domain to perform well on an unlabeled target domain.

### Experiments

1. **Source-Only Baseline**  
   - Trained using standard empirical risk minimization (ERM) on the source domain.  
   - Evaluated on both source and target test sets to measure domain shift.  

2. **Domain Alignment Methods**  
   - Implemented or used:
     - **DAN (Deep Alignment Network)** – MMD-based feature distribution matching.  
     - **DANN (Domain-Adversarial Neural Network)** – adversarial alignment using a Gradient Reversal Layer.  
     - **CDAN (Conditional DANN)** – class-aware alignment for better semantic consistency.  
   - Compared target accuracy, class-wise F1 scores, and analyzed negative transfer effects.  

3. **Self-Training (Pseudo-Labeling)**  
   - Generated pseudo-labels for target samples using the source-trained model.  
   - Fine-tuned on high-confidence pseudo-labeled samples to boost target performance.  

4. **Concept & Label Shift Experiments**  
   - Simulated rare-class and label imbalance scenarios.  
   - Measured the robustness of each method under varying domain shifts.  

### Visualizations
- **t-SNE embeddings** of source vs. target features.  
- **Confusion matrices** to analyze misclassifications.  
- **Heatmaps** showing class distribution shift and per-class accuracy.

---

## 🌍 Task 2: Domain Generalization (DG)

**Goal:** Generalize to a completely unseen target domain without using any target data during training.

### Dataset
- Multi-domain dataset such as **PACS** (Photo, Art, Cartoon, Sketch).  
- Trained on multiple source domains and evaluated on the held-out domain.

### Methods

1. **ERM (Empirical Risk Minimization)**  
   - Baseline model trained on merged source domains.  
   - Evaluated on both source and unseen target domain.  

2. **IRM (Invariant Risk Minimization)**  
   - Learned invariant representations by penalizing gradient variance across source domains.  
   - Reported both target accuracy and IRM penalty to check for trivial solutions.  

3. **Group DRO (Distributionally Robust Optimization)**  
   - Optimized for the worst-performing source domain to enhance robustness.  
   - Balanced performance across domains.  

4. **SAM (Sharpness-Aware Minimization)**  
   - Enforced flat minima for improved generalization to unseen domains.  
   - Compared flatness and robustness with ERM-trained models.

### Analysis
- Compared **target-domain accuracy**, **worst-case source accuracy**, and **flatness of loss surfaces**.  
- Discussed trade-offs between **invariance** and **discriminability**.  
- Analyzed how **flat minima** improve cross-domain robustness.  

---

## 🧠 Task 3: Prompt Learning & CLIP-based Domain Adaptation

**Goal:** Investigate prompt-based domain adaptation and generalization using **CLIP (Contrastive Language–Image Pre-training)**.

### Experiments

1. **Zero-Shot vs Fine-Tuned CLIP**  
   - Evaluated CLIP zero-shot classification using prompts like  
     `"a photo of a {class}"`, `"a sketch of a {class}"`, etc.  
   - Compared zero-shot results with fine-tuned and linear-probed CLIP models.  

2. **Prompt Learning (CoOp & CoCoOp)**  
   - Learned domain-specific text prompts while keeping CLIP frozen.  
   - Combined supervised (source) and unsupervised (target) objectives to adapt prompts.  

3. **Gradient Conflict & Alignment Analysis**  
   - Computed cosine similarity between gradients of different domains.  
   - Explored gradient alignment ideas from **PCGrad** and **GradCos** to mitigate domain interference.  

4. **Open-Set Generalization**  
   - Evaluated how prompt tuning affects CLIP’s ability to recognize unseen classes.  
   - Analyzed calibration metrics and prompt embedding similarities across domains.

---

## 🔍 Key Insights

- **Reducing domain divergence** improves adaptation but may reduce class discriminability if overdone.  
- **Self-training** can rival complex methods when pseudo-labels are reliable.  
- **IRM** is theoretically elegant but practically difficult to optimize.  
- **Group DRO** enhances robustness by focusing on the hardest domains.  
- **SAM** finds flatter minima that generalize well across unseen distributions.  
- **Prompt tuning** effectively adapts CLIP to new domains, but can be brittle.  
- **Gradient alignment** helps stabilize learning across domains by avoiding conflicting updates.

---

## ⚙️ Setup & Requirements

```bash
# Clone the repository
git clone https://github.com/xylicblue/ATML-PA2
cd ATML-PA2

