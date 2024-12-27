# Adaptive Entropy-Guided Universal Domain Adaptation (AEG-UDA)

## Overview

Adaptive Entropy-Guided Universal Domain Adaptation (AEG-UDA) introduces a novel approach to address universal domain adaptation challenges, focusing on four specific scenarios:

1. **Open-Set Domain Adaptation (ODA)**  
2. **Open Partial Domain Adaptation (OPDA)**  
3. **Closed Domain Adaptation (CDA)**  
4. **Partial Domain Adaptation (PDA)**  

This framework is designed to compare our model, AEG-UDA, with other state-of-the-art models using:
- A unique **Dynamic Adaptive Threshold**.
- An **Entropy-Guided Refinement Pseudo-Labeling Strategy**.
- A novel **Dynamic Rejection Loss** to handle domain overlaps and unknown classes effectively.

AEG-UDA is tested against leading benchmarks in domain adaptation, leveraging its robust mechanisms to balance performance across diverse tasks.

---

## Results

Results for the emulated DANCE baseline and our AEG-UDA approach can be found in the **`results`** folder. This folder contains:
1. Results for DANCE emulated runs categorized by adaptation scenario (ODA, OPDA, CDA, PDA).
2. Results for AEG-UDA, categorized similarly for direct comparisons.

Additionally, the **`Testing Other Adaptation Models`** folder includes implementations and testing for other models such as:
- SF-UDA  
- DANN  
- UAN  

These models serve as benchmarks for evaluating AEG-UDA in diverse adaptation scenarios.

---

## Dataset Preparation

To begin, download the **Office-31 dataset**, which is required for all experiments. The dataset can be obtained from the following link:  
[Office-31 Dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/).

Prepare the dataset in the following directory structure:
