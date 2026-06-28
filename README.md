# Project 13: Style Shift Robustness in Visual Recognition

**Course:** DSC02 Machine Learning 60 — Summer 2026  
**Group:** ML_DS_60_03

## Team Members
- Bhazil Bakruthin Ali Ahamed (68819399)
- Yuvaraj Rengaraj (47229423)

## Problem Statement
Image classifiers often learn style-specific cues (colour, texture) instead of object shape, so a model trained on some visual styles can fail on an unseen style. This project measures that style-shift drop and compares methods that improve robustness.

## Dataset
PACS (Photo, Art Painting, Cartoon, Sketch) — 7 object classes, ~9,991 images. Specified as "PACS via DomainBed"; loaded here from a public parquet mirror (identical data).

## Methodology
- **Leave-One-Domain-Out:** train on 3 styles, test on the held-out 4th, rotate through all 4.
- Per-domain accuracy reported separately (not just one average), as required.

## Models / Methods Compared
- **Baseline (ERM):** ResNet-18 pretrained on ImageNet, standard training.
- **+ Augmentation:** heavy style augmentation (grayscale, colour jitter, blur) to force shape-based learning.
- **+ BN-Adapt (Test-Time Adaptation):** recompute BatchNorm statistics on the unseen domain, no retraining.

## Evaluation
- Accuracy on each unseen domain (reported separately) + overall average.
- Per-class accuracy on the hardest domain (Sketch) for error analysis.

## Key Results
- Baseline collapses on unseen Sketch (~62%); dog-sketches misclassified as elephants (~15%).
- Augmentation lifts Sketch by ~+11; average rises from ~76% to ~84% with all methods.

## Tools
- Google Colab (T4 GPU)
- PyTorch / torchvision (ResNet-18)
- pandas, matplotlib

## How to Run
Open the notebook in Google Colab, set Runtime to GPU, and run all cells top to bottom.
