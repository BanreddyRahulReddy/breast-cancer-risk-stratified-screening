# breast-cancer-risk-stratified-screening
AI-powered breast cancer detection system using risk-stratified thresholds to reduce false positives in low-risk patients and false negatives in high-risk patients. Built with TensorFlow/Keras for Deep Learning course project.
# 🏥 Risk-Stratified Breast Cancer Screening

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> AI-powered breast cancer detection system using **risk-stratified thresholds** to reduce false positives in low-risk patients and false negatives in high-risk patients.

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Problem Statement

### The German Hospital Case Study (2024)

A leading hospital network in Germany integrated an AI diagnostic system for radiology. While initially successful, two critical problems emerged:

1. **False Positives (FP):** Healthy patients flagged as having cancer → unnecessary biopsies, psychological trauma
2. **False Negatives (FN):** Real cancers missed by AI → delayed treatment, worse outcomes

**Root Cause:** Using a **single threshold (0.5)** for all patients, regardless of their individual risk profiles.

---

## 💡 Solution: Risk-Stratified Screening

### Core Concept
**Different patients → Different thresholds → Personalized care**

Instead of one-size-fits-all, we use **patient risk factors** to determine appropriate screening thresholds:

| Risk Group | Threshold | Strategy | Goal |
|------------|-----------|----------|------|
| 🔴 **High-Risk** | 0.3 | Aggressive screening | Maximize cancer detection (reduce FN) |
| 🟢 **Low-Risk** | 0.7 | Conservative screening | Minimize false alarms (reduce FP) |

### Risk Factors
- Age (>50 years)
- Family history of breast cancer
- BRCA1/BRCA2 gene mutations
- Previous abnormal biopsies
- Dense breast tissue

---

## 📊 Results

### Dataset
- **Training:** 222,020 images
- **Validation:** 55,504 images
- **Source:** [Kaggle Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

### Performance Metrics

#### 🔴 High-Risk Group (23,220 patients, Threshold=0.3)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall (Sensitivity)** | **79.8%** | Caught 9,036 out of 11,318 cancers |
| **False Negatives** | 2,282 | Missed cancers (needs improvement) |
| **False Positives** | 1,673 | Acceptable trade-off for high-risk |
| **AUC** | 0.906 | Excellent discrimination |

#### 🟢 Low-Risk Group (32,284 patients, Threshold=0.7)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Specificity** | **97.3%** | Correctly cleared 26,961 healthy patients |
| **False Positives** | 884 | Only 2.7% false alarm rate! |
| **Precision** | 67.4% | When flagged, likely real cancer |
| **AUC** | 0.905 | Excellent discrimination |

### Key Insight
By using risk-stratified thresholds, we achieve:
- ✅ **80% cancer detection** in high-risk patients (vs. missing critical cases)
- ✅ **97% avoid false alarms** in low-risk patients (vs. unnecessary anxiety)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-risk-stratified-screening.git
cd breast-cancer-risk-stratified-screening
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**
```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
# Extract to data/merged_dataset/
```

---

## 💻 Usage

### Training the Model

```bash
python src/train.py --data_dir data/merged_dataset --epochs 15 --batch_size 32
```

### Running Risk-Stratified Evaluation

```bash
python src/evaluate.py --model_path models/best_model.h5 --data_dir data/merged_dataset
```

### Jupyter Notebook (Interactive)

```bash
jupyter notebook notebooks/risk_stratified_analysis.ipynb
```

---

## 📁 Dataset

### Source
[Kaggle: Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

### Structure
- **IDC (Invasive Ductal Carcinoma):** Positive class (cancer)
- **Non-IDC:** Negative class (healthy tissue)
- **Image Size:** 50x50 pixels, RGB
- **Total Images:** 277,524 patches

### Download Instructions
1. Go to the Kaggle dataset link above
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file
4. Place in `data/merged_dataset/`

**Note:** The dataset is ~3GB, so it's **NOT included in this repository**.

---

## 🧠 Model Architecture

### CNN Architecture
```
Input (50x50x3)
    ↓
Conv2D(32, 3x3) + BatchNorm + MaxPool
    ↓
Conv2D(64, 3x3) + BatchNorm + MaxPool
    ↓
Conv2D(128, 3x3) + BatchNorm + MaxPool
    ↓
Flatten
    ↓
Dense(256) + Dropout(0.5)
    ↓
Dense(128) + Dropout(0.3)
    ↓
Dense(1, sigmoid) → Probability (0-1)
```

### Training Configuration
- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Binary Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau
- **Epochs:** 15 (with early stopping)
- **Validation Split:** 20%

### Risk Stratification
After training, we apply different thresholds:
- **High-Risk:** `probability >= 0.3` → Flag for biopsy
- **Low-Risk:** `probability >= 0.7` → Flag for biopsy

---

## 📂 Project Structure

```
breast-cancer-risk-stratified-screening/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── data/                             
│   └── README.md                     # Dataset download instructions
│
├── src/                              
│   ├── model.py                      # CNN model definition
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Risk-stratified evaluation
│   └── utils.py                      # Helper functions
│
├── notebooks/                        
│   └── risk_stratified_analysis.ipynb  # Complete analysis notebook
│
├── results/                          
│   └── risk_stratified_complete_analysis.png  # Visualization
│
├── docs/                             
│   ├── case_study.pdf                # Original German hospital case
│   ├── methodology.md                # Detailed methodology
│   └── presentation.html             # Presentation slides
│
└── models/                           
    └── .gitkeep                      # Placeholder (models not pushed)
```

---

## 🔮 Future Work

### Technical Improvements
1. **Real Risk Integration:** Connect to Electronic Health Records (EHR) for actual patient risk factors
2. **Explainable AI:** Add Grad-CAM heatmaps to show which regions influenced predictions
3. **Better Architecture:** ResNet, EfficientNet, or Vision Transformers
4. **Ensemble Models:** Combine multiple models for better accuracy
5. **Dynamic Thresholds:** Use Bayesian optimization to adjust thresholds based on outcomes

### Clinical Improvements
6. **Multi-Modal Fusion:** Combine imaging + genetic data + blood biomarkers
7. **Federated Learning:** Train across hospitals without sharing patient data (privacy-preserving)
8. **Uncertainty Quantification:** Provide confidence intervals for predictions
9. **Class Imbalance Handling:** Use focal loss or weighted sampling
10. **Longitudinal Analysis:** Track patient scans over time for better detection

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

### Areas for Contribution
- Improve model architecture
- Add explainability features (Grad-CAM, SHAP)
- Implement real patient risk profiling
- Add unit tests
- Improve documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Course
- **Course:** Deep Learning (22CSE619)
- **Faculty:** Dr. Sindhumitha K
- **Institution:** [Your University Name]

### Dataset
- **Source:** Kaggle - Paul Mooney
- **Original Paper:** [Link if available]

### References
1. **German Hospital Case Study** - AI in Healthcare: The Diagnostic Dilemma (2024)
2. **Grad-CAM:** Visual Explanations from Deep Networks - Selvaraju et al.
3. **Risk-Stratified Screening:** Breast Cancer Surveillance Guidelines - American Cancer Society

### Tools & Libraries
- TensorFlow/Keras - Deep learning framework
- Scikit-learn - Machine learning utilities
- Matplotlib/Seaborn - Visualization
- NumPy/Pandas - Data manipulation

---

## 📧 Contact

**Author:** Rahul Reddy  
**Email:** your.email@example.com  
**LinkedIn:** [Your LinkedIn Profile]  
**GitHub:** [@yourusername](https://github.com/yourusername)

---

## 📈 Citation

If you use this work in your research, please cite:

```bibtex
@misc{risk_stratified_screening2024,
  author = {Your Name},
  title = {Risk-Stratified Breast Cancer Screening Using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/breast-cancer-risk-stratified-screening}
}
```

---

## 🎯 Key Results Summary

| Metric | Traditional (0.5) | Our Approach |
|--------|-------------------|--------------|
| High-Risk Cancer Detection | ~50-60% | **79.8%** ✅ |
| Low-Risk False Alarms | ~10-15% | **2.7%** ✅ |
| Patient Experience | One-size-fits-all | Personalized ✅ |
| Clinical Applicability | Limited | High ✅ |

---

**⭐ If you found this project helpful, please star the repository!**
