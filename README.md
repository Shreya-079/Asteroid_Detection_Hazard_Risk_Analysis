
# ☄️ Asteroid Hazard Prediction using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red)
![Status](https://img.shields.io/badge/Project-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 🌌 Project Overview

Asteroids passing near Earth pose potential risks.  
This project applies **Machine Learning and Data Analysis** to predict whether an asteroid is **Potentially Hazardous (PHA)** using orbital and physical characteristics obtained from NASA datasets.

The system builds an intelligent pipeline that:

✔ Cleans astronomical data  
✔ Performs exploratory analysis  
✔ Handles imbalanced datasets  
✔ Benchmarks multiple ML models  
✔ Selects optimal prediction model  
✔ Explains predictions using Explainable AI  

This project demonstrates the application of **Artificial Intelligence in Space Science and Planetary Defense**.

---

## 🚀 Key Features

- 📊 Complete Data Analysis Pipeline
- ☄️ NASA Near-Earth Asteroid Dataset
- ⚖️ Class imbalance handling using SMOTE
- 🤖 Multi-model Machine Learning comparison
- 🌲 Optimized Random Forest Classifier
- 📈 ROC Curve & Performance Evaluation
- 🧠 Explainable AI using SHAP
- 💾 Model Saving & Reproducibility

---

## 📂 Dataset Information

Source: **NASA Small Body Database (SBDB)**

The dataset contains orbital parameters and asteroid characteristics.

| Feature | Description |
|---|---|
| Absolute Magnitude (H) | Asteroid brightness |
| Diameter | Estimated size |
| Albedo | Surface reflectivity |
| Eccentricity | Orbit shape |
| Semi-major Axis | Orbit size |
| Inclination | Orbit tilt |
| Perihelion Distance | Closest to Sun |
| Aphelion Distance | Farthest from Sun |
| Earth MOID | Minimum orbit distance from Earth |
| PHA | Hazard Label |

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Column normalization
- Label encoding (Y/N → 1/0)
- Missing diameter estimation using astronomy formula:
- D = (1329 / sqrt(Albedo)) * 10^(-H/5)
- Missing value handling
- Dataset cleaning

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Feature distributions
- Correlation heatmap
- Hazard class visualization
- Relationship analysis between orbital features

---

### 3️⃣ Class Imbalance Handling
Asteroid datasets are highly imbalanced.

Applied:

✅ **SMOTE (Synthetic Minority Oversampling Technique)**  
to improve hazardous asteroid detection.

---

### 4️⃣ Feature Scaling
Used **StandardScaler** for normalization before model training.

---

### 5️⃣ Machine Learning Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost
- Neural Network (TensorFlow/Keras)

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

### 🏆 Final Model

Best performing model:
RandomForestClassifier(
n_estimators=300,
class_weight='balanced',
random_state=42
)

---

### 🧠 Explainable AI

Used **SHAP (SHapley Additive Explanations)** to:

- Identify important asteroid parameters
- Interpret model decisions
- Improve scientific transparency

---

## 📊 Results & Insights

- Earth MOID strongly influences hazard prediction
- Larger asteroid diameter increases risk probability
- Balanced training significantly improves recall
- Random Forest provides stable performance

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- XGBoost
- TensorFlow / Keras
- SHAP
- Joblib

---

## 📁 Project Structure
Asteroid-Hazard-Prediction/
│
├── astroid_DA.ipynb # Main notebook
├── sbdb_query_results.csv # Dataset
├── X_train.npy
├── X_test.npy
├── y_train.npy
├── y_test.npy
├── model.pkl # Trained model
└── README.md

---

## ⚡ Installation

Clone repository:
git clone https://github.com/shreya79/asteroid-hazard-prediction.git
cd asteroid-hazard-prediction

Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow shap joblib

---

## ▶️ Usage

Run Jupyter Notebook:
jupyter notebook astroid_DA.ipynb

Workflow:

1. Load dataset
2. Execute preprocessing cells
3. Perform EDA
4. Train models
5. Evaluate results
6. Save final model

---

## 🔬 Future Improvements

- 🌍 Real-time asteroid monitoring dashboard
- 🚀 NASA API live data integration
- 🧠 Deep learning orbital sequence prediction
- 📡 Early asteroid warning system
- 📄 Research paper publication extension

---

## 🎯 Applications

- Planetary Defense Systems
- Space Risk Monitoring
- Astronomical Research
- AI in Space Exploration
- Scientific Decision Support

---

## 👩‍💻 Author

**Shreya**  
B.Tech Computer Science Engineering  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork it  
📢 Share with others  

---

<p align="center">
Made with ❤️ using Artificial Intelligence & Space Science
</p>
