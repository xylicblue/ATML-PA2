Domain Generalization with Invariant & Robust Learning

This project explores various Domain Generalization (DG) techniques to train a model that is robust to domain shifts. The goal is to train a model on several source domains and have it generalize effectively to a completely unseen target domain without any form of adaptation at test time.

This notebook implements and compares four distinct DG strategies:

Empirical Risk Minimization (ERM): The standard baseline.

Invariant Risk Minimization (IRM): A method that seeks domain-invariant feature representations.

Group Distributionally Robust Optimization (Group DRO): An approach that optimizes for worst-case performance across source domains.

Sharpness-Aware Minimization (SAM): An optimizer that seeks flat minima in the loss landscape to improve generalization.

The experiments are conducted on the PACS dataset.

Source Domains: art_painting, cartoon, photo

Unseen Target Domain: sketch

Table of Contents

Setup and Data Preparation

Model Architecture

Experiment 1: Empirical Risk Minimization (ERM) Baseline

Experiment 2: Invariant Risk Minimization (IRM)

Experiment 3: Group Distributionally Robust Optimization (Group DRO)

Experiment 4: Sharpness-Aware Minimization (SAM)

Analysis and Visualization

Loss Landscape Visualization

Feature Space Visualization (t-SNE)

Quantitative Flatness (Gradient Similarity)

1. Setup and Data Preparation

This phase prepares the PACS dataset for the DG experiments.

Loading Data: The script begins by loading the PACS dataset from the Hugging Face datasets library.

Defining Transforms: Two sets of image transformations are defined:

train_transform: Includes data augmentation techniques like RandomResizedCrop, RandomHorizontalFlip, and ColorJitter. This is applied to all source domains.

eval_transform: A deterministic transform that only resizes and normalizes images. This is applied to the target domain.

Domain Splitting: The script filters the master dataset to create separate datasets for each of the four domains (art_painting, cartoon, photo, sketch).

Applying Transforms: The apply_transforms function sets the appropriate transform for each domain dataset.

Creating DataLoaders:

For ERM and SAM, all source domain datasets are combined into a single ConcatDataset (train_dataset_erm) to create a unified train_loader_erm.

For IRM and Group DRO, which require per-domain loss calculations, separate DataLoaders are created for each source domain and stored in a dictionary (source_train_loader).

A test_loader is created for the unseen sketch target domain, which is used exclusively for evaluation.

2. Model Architecture

The notebook initially defines a custom Vision Transformer (ViT) from scratch. However, the executed experiments and final analyses are performed using a pre-trained ResNet-50 from torchvision.

Backbone: A resnet50 model pre-trained on ImageNet is used.

Classifier Head: The final fully connected layer (model.fc) is replaced with a new nn.Linear layer to match the 7 classes of the PACS dataset.

Training Strategy: The entire model (backbone and head) is fine-tuned during training for all experiments.

3. Experiment 1: Empirical Risk Minimization (ERM) Baseline

This section implements the standard DG baseline.

Training: The ResNet-50 model is trained on the train_loader_erm, which contains a shuffled mix of all source domain data. The training objective is to minimize the standard Cross-Entropy loss. Checkpoints of the model are saved at various epochs.

Evaluation: A dedicated evaluation script is used to provide a granular performance breakdown. For each saved checkpoint, the script measures and reports:

Target Domain Accuracy: The primary metric, calculated on the unseen 'sketch' domain.

Per-Source Domain Accuracy: Accuracy is calculated individually for art_painting, cartoon, and photo.

Summary Metrics: The script also reports the average and worst-case accuracy across the source domains.

4. Experiment 2: Invariant Risk Minimization (IRM)

This experiment implements IRMv1 to encourage the model to learn domain-invariant features.

IRM Penalty: The compute_irm_penalty function implements the IRMv1 penalty. It calculates the squared norm of the gradient of the loss with respect to a dummy classifier (a scalar scale variable), which penalizes representations where the optimal classifier differs across domains.

Training: The training loop is modified to handle multiple source domains simultaneously. In each step:

A batch is drawn from each source domain's DataLoader.

The standard ERM loss and the IRM penalty are computed for each domain.

The final loss is calculated as: mean(ERM_losses) + penalty_weight \* mean(IRM_penalties).

The model is updated based on this combined loss.

Ablation: The script is run multiple times with different values for the PENALTY_WEIGHT hyperparameter (e.g., 2, 5, 20, 100) to analyze its effect.

Evaluation: Each saved IRM model checkpoint is evaluated using the same granular script as the ERM baseline.

5. Experiment 3: Group Distributionally Robust Optimization (Group DRO)

This section implements Group DRO to explicitly optimize for the worst-performing source domain.

Training: The training loop is structured similarly to IRM. In each step:

The loss is computed for each source domain individually.

A vector of these per-domain losses is created.

A softmax function is applied to the (detached) loss vector to create a set of weights. This dynamically assigns a higher weight to the domain with the highest loss in the current step.

A final weighted_loss is computed by taking the dot product of the weights and the loss vector.

The model is updated by backpropagating this weighted loss.

Evaluation: The saved Group DRO model checkpoints are evaluated using the same granular per-domain script.

6. Experiment 4: Sharpness-Aware Minimization (SAM)

This experiment uses the SAM optimizer to find flatter minima in the loss landscape, which is theorized to improve generalization.

Optimizer: The script defines a SAM optimizer class that wraps a base optimizer (e.g., AdamW).

Training: The model is trained on the combined source train_loader_erm. The update step is a two-part process:

A standard forward and backward pass calculates the initial gradient.

optimizer.first_step() performs an "ascent" step, perturbing the model's weights in the direction of the gradient to find a point of high loss in the local neighborhood.

A second forward and backward pass is performed at this perturbed position.

optimizer.second_step() resets the weights to their original position and then performs the actual update using the gradient from the perturbed position.

Evaluation: The saved SAM model checkpoints are evaluated using the same granular per-domain script.

7. Analysis and Visualization

The final part of the notebook is dedicated to analyzing and visualizing the results from the different DG methods.

Loss Landscape Visualization

Goal: To visually compare the "flatness" of the solutions found by ERM and the best-performing SAM model.

Process:

The best ERM and SAM model checkpoints are loaded.

Two random direction vectors are generated with filter-wise normalization to create a meaningful 2D plane in the high-dimensional weight space.

The script calculates the loss of each model on the unseen target domain at various points along this 2D plane.

The results are plotted as two contour maps, visually demonstrating whether the SAM solution occupies a wider, flatter basin of low loss compared to the ERM solution.

Feature Space Visualization (t-SNE)

Goal: To understand how each DG method organizes the feature space for different classes and domains.

Process:

The best-performing model from each method (ERM, IRM, DRO, SAM) is loaded.

The script extracts feature vectors from a penultimate layer (avgpool) for a balanced set of images from all four domains (including the target).

t-SNE is used to project these high-dimensional features into a 2D space.

The 2D embeddings are plotted as a scatter plot, with points colored by class and shaped by domain, allowing for a qualitative analysis of class separability and domain alignment.

Quantitative Flatness (Gradient Similarity)

Goal: To provide a numerical proxy for the "cross-domain flatness" of the ERM and SAM solutions.

Process:

The best ERM and SAM models are loaded.

The script iterates through the source domain datasets. For each batch, it computes the gradient of the loss with respect to the model parameters for each of the three source domains.

It then calculates the pairwise cosine similarity between these three gradient vectors.

The average similarity is reported. A higher average similarity suggests that the gradients from different domains are more aligned, indicating that a single update step benefits all domains simultaneously—a characteristic of a flat, domain-general loss basin.
