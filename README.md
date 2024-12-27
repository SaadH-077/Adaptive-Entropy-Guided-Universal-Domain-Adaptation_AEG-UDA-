# Adaptive Entropy-Guided Universal Domain Adaptation (AEG-UDA)

## Overview

Adaptive Entropy-Guided Universal Domain Adaptation (AEG-UDA) introduces a novel approach to address universal domain adaptation challenges, focusing on four specific scenarios:

![Implementation Diagram](Full_Pipeline.jpeg "Architectural Pipeline")

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
[Office-31 Dataset](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md).

Prepare the dataset in the following directory structure:

data/
├── amazon/Images/
├── dslr/Images
├── webcam/Images

---

Once downloaded:
1. Place the zipped dataset in your Google Drive.  
2. Ensure it is accessible during training by mounting your Google Drive in Colab.

---

## How to Run

1. Open the corresponding file in **Google Colab**. This can be:
   - DANCE emulated script.
   - AEG-UDA script.

2. Mount your Google Drive to access the dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

3. Connect to a GPU and Run the Cells Sequentially

--

### 
1. ODA : !sh script/run_office_obda.sh 0 /content/DANCE/configs/office-train-config_ODA.yaml
2. OPDA : !sh script/run_office_opda.sh 0 /content/DANCE/configs/office-train-config_OPDA.yaml
3. CDA : !sh script/run_office_cls.sh 0 /content/DANCE/configs/office-train-config_CDA.yaml
4. PDA : !sh script/run_office_cls.sh 0 /content/DANCE/configs/office-train-config_PDA.yaml


