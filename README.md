# Adaptive Entropy-Guided Universal Domain Adaptation (AEG-UDA)

A comprehensive course project on **Universal Domain Adaptation (UDA)** that proposes a new framework — **AEG-UDA** — to enhance the adaptability and robustness of models in non-overlapping and partially overlapping label spaces. This method integrates dynamic thresholding, entropy-guided pseudo-labeling, and a novel dynamic rejection loss to outperform existing approaches like DANCE across multiple adaptation scenarios.

> 🧠 **Course**: Advanced Topics in Machine Learning (ATML - CS)  
> 📅 **Semester**: Fall 2024
> 🎓 **Institution**: LUMS  
> 👨‍💻 Contributors: Muhammad Saad Haroon, Jawad Saeed, Daanish Uddin Khan

![Full Architectural Pipeline](Implementation%20Diagrams/Full%20Pipeline.jpeg "Architectural Pipeline")
---

## 🧠 Key Contributions

1. **Dynamic Adaptive Threshold**  
   Automatically adjusts entropy thresholds during training to better separate confident and ambiguous target samples.

   ![Implementation Diagram](Implementation%20Diagrams/Dynamic%20Thresholding.jpeg "Dynamic Thresholding")

2. **Entropy-Guided Pseudo-Labeling (EGPL)**  
   Assigns soft pseudo-labels with entropy-aware weighting to confident target predictions.

   ![EGR Pseudo-Labeling](Implementation%20Diagrams/EGR%20Pseudo-Labelling.jpeg "EGR Pseudo-Labelling")

3. **Dynamic Rejection Loss (DRL)**  
   Penalizes uncertain (high-entropy) predictions dynamically to reduce noise and confusion.

4. **Supports All UDA Variants**  
   ![UDA Types](Implementation%20Diagrams/UDA-Types.jpeg "Universal Domain Adaptation Scenarios")

---

## 🗃️ Dataset Preparation

To begin, download the **Office-31 dataset**, which is required for all experiments. The dataset can be obtained from the following link:  
📥 [Office-31 Dataset](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md)

Prepare the dataset in the following directory structure:

```
data/
├── amazon/Images/
├── dslr/Images/
├── webcam/Images/
```

---

Once downloaded:

1. Place the zipped dataset in your **Google Drive**  
2. Ensure it is accessible during training by mounting your Google Drive in **Google Colab**

---

## 🚀 How to Run

1. Open the corresponding file in **Google Colab**. This can be:
   - DANCE emulated script.
   - AEG-UDA script.

2. Mount your Google Drive to access the dataset:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. Connect to a GPU and **run the cells sequentially**

---

### 🖥️ Shell Commands for Training

```bash
# ODA
!sh script/run_office_obda.sh 0 /content/DANCE/configs/office-train-config_ODA.yaml

# OPDA
!sh script/run_office_opda.sh 0 /content/DANCE/configs/office-train-config_OPDA.yaml

# CDA
!sh script/run_office_cls.sh 0 /content/DANCE/configs/office-train-config_CDA.yaml

# PDA
!sh script/run_office_cls.sh 0 /content/DANCE/configs/office-train-config_PDA.yaml
```

---

## 📊 Results & Evaluation

All experimental results, performance comparisons, and visualizations are detailed in the final report:

📄 [`Project_Report_Final.pdf`](./Project_Report_Final.pdf)

Key insights:
- **AEG-UDA achieves competitive performance** across CDA, ODA, PDA, and OPDA settings
- **Improved generalization** via entropy-aware pseudo-labeling
- **Faster inference** than DANCE in several scenarios
- Stronger **cluster compactness** in t-SNE visualizations

---

## 📈 Evaluation Metrics

- **Test Accuracy**
- **Mean Per-Class Accuracy (MPCA)**
- **Inference Time**
- **Loss Curves**
- **t-SNE Feature Visualization**

---

## 🔭 Future Work

- Add **mixed-precision training** for faster convergence  
- Extend DRL with **uncertainty-based dynamic weighting**  
- Adapt for **streaming/real-time domain adaptation**  
- Scale to **larger, multi-modal datasets**

---

## 📚 Citation

If you use or reference this work, please cite:

```
@project{AEG-UDA2025,
  title = {Adaptive Entropy-Guided Universal Domain Adaptation (AEG-UDA)},
  note = {Developed as part of the CS-6304: Advanced Topics in Machine Learning course at LUMS, 2025.}
}
```

---

> “In a shifting world, the best models don’t memorize — they adapt.”
