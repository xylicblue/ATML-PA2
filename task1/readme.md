Domain Adaptation on the PACS Dataset
Overview

This project investigates multiple domain adaptation strategies using the PACS dataset, a benchmark for evaluating visual recognition models under distributional shifts across different domains.
The objective is to assess how effectively various adaptation techniques can improve generalization from a labeled source domain to an unlabeled target domain.

Methodology

The notebook is organized into several sections, each corresponding to a specific domain adaptation approach. Every method is independently trained, evaluated, and analyzed based on its performance on both the source and target domains.

1. Source-Only Baseline

Description:
A standard ResNet-50 model is trained exclusively on labeled source domain data without any domain adaptation mechanism.

Purpose:
Establishes a baseline for comparison. The difference between source and target test accuracies quantifies the severity of the domain shift.

Output:

Accuracy on source and target domains

Trained model saved as source_only_model.pth

2. Domain-Adversarial Neural Network (DANN)

Description:
Implements the DANN architecture, which includes a domain discriminator alongside the label predictor.
A Gradient Reversal Layer (GRL) is employed to promote domain-invariant feature learning.

Purpose:
To align source and target feature distributions adversarially.
The GRL reverses gradients from the domain discriminator, forcing the feature extractor to produce features that are indistinguishable across domains.

Output:

Label and domain loss logs

Source and target accuracies

Model saved as dann_model_finetuned.pth

3. Deep Adaptation Network (DAN)

Description:
Implements the DAN framework, which aligns source and target features using the Maximum Mean Discrepancy (MMD) criterion.

Purpose:
To explicitly minimize the statistical distance between source and target feature distributions in a reproducing kernel Hilbert space (RKHS).

Output:

Training logs (label and MMD losses)

Source and target accuracies

Model saved as dan_model_finetuned_lambda_X.pth

4. Conditional Domain-Adversarial Network (CDAN)

Description:
An extension of DANN that conditions the domain discriminator on both feature representations and class predictions.

Purpose:
To achieve fine-grained domain alignment by incorporating class information into the adaptation process, ensuring that similar classes across domains remain aligned.

Output:

Training logs and final accuracies

Model saved as cdan_model_finetuned_lambda_X.pth

5. Self-Training via Pseudo-Labeling

Description:
A semi-supervised approach where the model generates pseudo-labels for target domain samples with high prediction confidence.

Purpose:
To leverage unlabeled target data.
Samples exceeding the confidence threshold are combined with source-labeled data to fine-tune the model, improving adaptation without explicit target supervision.

Output:

Number of pseudo-labels generated

Final target domain accuracy

Updated model weights

6. Concept Shift Analysis

Description:
Evaluates model robustness under simulated distribution shifts, testing generalization under more realistic conditions.

Purpose:
Two key scenarios are examined:

Missing Class (Label Shift): Evaluates performance when a class present during training is absent from the target domain.

Rare Class (Imbalanced Data): Examines model behavior when a target class is severely underrepresented.

Output:

Accuracy scores for each model on modified target datasets

Confusion matrix plots for the rare class scenario

7. Visualizations

Description:
Generates t-SNE visualizations to explore the learned feature representations of each model.

Purpose:
To qualitatively assess domain alignment and class separation:

Domain Alignment: Visualizes overlap between source (blue) and target (red) features.

Class Separation: Displays class-specific clustering across domains.

Output:

t-SNE plots illustrating domain and class alignment for each model

Key Experimental Parameters
Parameter	Description
SOURCE_DOMAIN / TARGET_DOMAIN	Specifies the source and target domains (e.g., art_painting, cartoon).
NUM_EPOCHS	Number of training epochs.
LEARNING_RATE	Learning rate for optimization.
LAMBDA_WEIGHT	Balances classification and adaptation losses (used in DAN and CDAN).
CONFIDENCE_THRESHOLD	Minimum confidence required to generate pseudo-labels in self-training.
Summary

This project provides a comprehensive empirical comparison of contemporary domain adaptation techniques, including adversarial (DANN, CDAN), distribution-based (DAN), and semi-supervised (pseudo-labeling) methods.
Through quantitative evaluation and qualitative visualization, the notebook highlights the strengths and limitations of each approach in mitigating domain shift across the diverse visual domains of the PACS dataset.
