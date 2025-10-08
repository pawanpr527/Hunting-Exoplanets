# 🪐 Hunting Exoplanet

### 🚀 Predicting Exoplanets using Machine Learning & Real-Time Model Building

**Hunting Exoplanet** is an interactive platform built using datasets from **Kepler**, **TESS**, and **K2** missions to identify and classify **exoplanet candidates**.  
The system enables users and researchers to **train, fine-tune, and evaluate ML models in real time**, analyze results visually, and even **summarize findings** using **Google GenAI** integration.

---

## 🌌 Overview

This project was developed as part of the **NASA Space Apps Challenge**.  
It empowers users to explore exoplanet data interactively and make real-time discoveries through AI-assisted insights.

### 🧠 Core Idea
To simplify the exploration and discovery of exoplanets by leveraging:
- **Machine Learning** for planet classification  
- **Real-time data visualization**  
- **AI-based summarization** for research acceleration  

---

## ⚙️ Features

### 🔭 Machine Learning Models
- Three independent ML models trained on **Kepler**, **TESS**, and **K2** datasets.  
- Predicts **Confirmed Planet**, **Candidate**, or **False Positive**.  
- Models include:
  
  - Random Forest Classifier  


### ⚡ Real-Time Model Building
- Users can **build, train, and fine-tune** models directly from the interface.  
- Compare multiple models simultaneously and analyze accuracy, F1 score, and confusion matrices in real time.  

### 🧩 Data Visualization
- Real-time plotting of performance metrics.  
- Graphical comparison of datasets and model outputs.  
- Insights into key planetary parameters like orbital period, planet radius, and stellar temperature.  

### 🧑‍💻 User Interaction
- Input custom **data points** to predict potential exoplanet classification.  
- Visual feedback and immediate prediction display.  

### 🤖 Google GenAI Integration
- Summarizes user-input data and predictions in plain language.  
- Generates **research summaries** for scientific analysis.  
- Helps researchers save time by auto-generating key insights.  



---

## 🧱 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML + BootStrap
| **Backend** | Flask |
| **Machine Learning** | scikit-learn |
| **Visualization** | chart.js |
| **AI Integration** | Google GenAI (for summarization) |

---

## 🔍 How It Works

1. **Dataset Ingestion**  
   Load Kepler, TESS, or K2 datasets.

2. **Model Selection**  
   Choose or build a custom ML model (e.g., Random Forest, XGBoost).

3. **Parameter Tuning**  
   Adjust hyperparameters (learning rate, estimators, depth, etc.).

4. **Training & Testing**  
   Train the model locally and view live metrics and graphs.

5. **Prediction**  
   Enter planetary parameters → Get classification output.

6. **AI Summarization**  
   Google GenAI summarizes the model performance and scientific relevance.

---

## 🧠 Example Prediction

**Input:**  
- Orbital Period: `45.3 days`  
- Planet Radius: `1.1 Earth radii`  
- Stellar Temperature: `5500 K`  

**Output:**  
> ✅ *Likely a Confirmed Exoplanet*  
> “The data aligns closely with known exoplanets in the Kepler catalog. Based on model confidence (92%), this is likely a confirmed planet.”

---
"note : modal has missing Toi.pkl file "
