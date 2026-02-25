# ☄️ Asteroid Detection & Hazard Risk Analysis using AI/ML

## 📌 Project Overview

Asteroids that approach Earth can pose potential threats depending on their size, orbit, and distance from Earth.
This project performs **data analysis, visualization, and machine learning-based hazard prediction** on NASA's Near-Earth Object (NEO) dataset to identify **potentially hazardous asteroids**.

The project combines **Data Science** and **Artificial Intelligence** techniques to analyze asteroid characteristics and predict their risk level using a supervised learning model.

---

## 🎯 Objectives

* Analyze asteroid properties such as size, brightness, and orbital parameters.
* Identify potentially hazardous asteroids.
* Visualize asteroid behavior using statistical and graphical analysis.
* Build a Machine Learning model to predict asteroid hazard risk.
* Improve understanding of space risk assessment using AI.

---

## 🧠 Technologies Used

* Python
* Google Colab
* Pandas (Data Analysis)
* Matplotlib (Visualization)
* Seaborn (Statistical Plots)
* Scikit-learn (Machine Learning)

---

## 📂 Dataset

**NASA Near Earth Object (NEO) Dataset**

The dataset contains information such as:

* Absolute Magnitude (Brightness)
* Estimated Diameter
* Orbit Class Type
* Perihelion & Aphelion Distance
* Hazard Classification

---

## ⚙️ Project Workflow

### 1️⃣ Data Collection

* Imported asteroid dataset from Google Drive.
* Loaded data using Pandas.

### 2️⃣ Data Cleaning

* Removed unnecessary columns.
* Checked missing values and dataset structure.

### 3️⃣ Exploratory Data Analysis (EDA)

Performed multiple visual analyses:

* Hazardous vs Non-Hazardous asteroid comparison
* Asteroid size distribution
* Orbit class analysis
* Hazard vs size relationship
* Pie chart hazard percentage
* 3D asteroid risk visualization
* Top largest asteroids visualization

### 4️⃣ Risk Label Creation

Custom logic created to classify:

* **High Risk**
* **Low Risk**

based on hazard status.

### 5️⃣ Machine Learning Model

A **Random Forest Classifier** was trained using asteroid features:

* Absolute magnitude
* Minimum diameter
* Maximum diameter
* Perihelion distance
* Aphelion distance

---

## 📊 Model Evaluation

* Train-Test Split (80/20)
* Accuracy Score Calculation
* Confusion Matrix Visualization
* Classification Report

The model predicts whether an asteroid is **potentially hazardous** based on physical and orbital characteristics.

---

## 📈 Visualizations Included

* Bar Charts
* Histograms
* Box Plots
* Pie Charts
* 3D Scatter Plot
* Heatmap Confusion Matrix

---

## ▶️ How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Asteroid-Detection-Hazard-Analysis.git
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Project

```bash
python asteroid_detection.py
```

OR open the notebook directly in Google Colab.

---

## 📁 Project Structure

```
Asteroid-Detection-Hazard-Analysis/
│
├── asteroid_detection.ipynb
├── asteroid_detection.py
├── neo_v2.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Key Features

✔ Data Analysis & Visualization
✔ Space Hazard Risk Identification
✔ Machine Learning Prediction Model
✔ 3D Risk Visualization
✔ End-to-End Data Science Workflow

---

## 🔮 Future Improvements

* Deep Learning based risk prediction
* Real-time NASA API integration
* Web dashboard using Streamlit
* Explainable AI for risk interpretation

---

## 👩‍💻 Author

**Shreya**
B.Tech Computer Science Engineering Student

---

## ⭐ Acknowledgement

Dataset inspired by NASA Near-Earth Object research initiatives for asteroid monitoring and planetary defense.

---

## 📜 License

This project is developed for academic and educational purposes.
