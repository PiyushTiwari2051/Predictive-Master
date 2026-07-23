<div align="center">

![Predictive Maintenance Hero Banner](assets/hero_banner.svg)

# ✈️ Predictive Maintenance for Jet Engines
### *Predict Engine Failures Before They Happen Using Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/AI_Framework-Scikit--Learn-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Warning_Accuracy-98.45%25-brightgreen.svg?style=for-the-badge)](assets/binary_roc_pr_curves.png)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>

---

## 👋 What Does This Project Do? (In Simple Terms)

When airplanes fly, jet engines run under intense heat and pressure. Over time, internal parts wear down and degrade.

Instead of waiting for an engine to break mid-flight or servicing it too early (wasting money), **this AI system looks at live sensor data** (temperature, pressure, rotor speed) to answer **3 simple questions**:

| ❓ Question | 💡 What the AI Answers | 🤖 AI Model Used |
| :--- | :--- | :--- |
| **1. How long will the engine last?** | Predicts exact **Remaining Useful Life (RUL)** in cycles | Random Forest Regressor |
| **2. Will it fail soon?** | Predicts if failure will happen within **30 cycles** (Yes/No) | 98.45% AUC Binary Classifier |
| **3. How urgent is maintenance?** | Classifies risk: 🟢 **Safe**, 🟡 **Warning**, or 🔴 **Danger** | Multi-Layer Perceptron (Neural Net) |

---

## 🔄 Simple Project Workflow

Here is how data flows from the engine sensors to the maintenance team:

![Pipeline Architecture](assets/pipeline_architecture.svg)

```text
 1. JET ENGINE SENSORS          2. AI SIGNAL CLEANING         3. AI PREDICTION ENGINE         4. ACTIONABLE REPORT
 🌡️ Temperature & Pressure  -->  📈 Calculate Moving      -->  🧠 Predict Failure Risk    -->  🛠️ Ground Engine or
    Data (21 Sensors)              Averages & Trends               & Remaining Cycles             Schedule Inspection
```

---

## 🖼️ Visual Demo & Charts Explained Simply

### 1. Engine Sensor Wear Over Time
As an engine gets older, sensor temperatures and pressures rise (red & yellow lines). The red dashed line marks the exact moment the engine failed.

<div align="center">
  <img src="assets/sensor_degradation_trends.png" alt="Sensor Degradation Trends" width="90%"/>
</div>

---

### 2. Failure Prediction Accuracy (98.45% Score)
The curve below shows how well our AI detects engines that are about to fail. The higher the green curve toward the top left, the more accurate the AI is!

<div align="center">
  <img src="assets/binary_roc_pr_curves.png" alt="Failure Prediction Accuracy" width="90%"/>
</div>

---

### 3. Estimated Remaining Life vs Actual Life
This chart compares what the AI predicted (y-axis) versus how long the engine actually lasted (x-axis). The closer points are to the red line, the better the prediction.

<div align="center">
  <img src="assets/regression_performance.png" alt="Remaining Life Prediction" width="90%"/>
</div>

---

### 4. Traffic Light Risk Matrix (Neural Network)
Our Neural Network puts engines into 3 simple safety buckets:
- 🟢 **Healthy** (More than 30 cycles left)
- 🟡 **Warning** (15 to 30 cycles left - schedule service)
- 🔴 **Critical** (Less than 15 cycles left - ground immediately)

<div align="center">
  <img src="assets/multiclass_mlp_confusion_matrix.png" alt="Risk Category Matrix" width="55%"/>
</div>

---

### 5. Most Important Sensors
Not all 21 sensors are equally useful. The bar chart below shows the **top sensors** the AI pays attention to (like LPC Outlet Temp and HPC Static Pressure).

<div align="center">
  <img src="assets/feature_importance.png" alt="Top Sensor Importances" width="90%"/>
</div>

---

## ⚡ Try It Yourself in 2 Minutes!

You can run predictions on any engine in the fleet right from your terminal!

### Step 1: Install & Run Pipeline

```bash
# Clone the repository
git clone https://github.com/PiyushTiwari2051/Predictive-Master.git
cd Predictive-Master

# Train models and generate all graphs
python run_pipeline.py
```

### Step 2: Test an Engine (Example: Engine #81)

```bash
python predict.py --engine_id 81
```

### 💻 Easy-to-Read Output:

```text
======================================================================
      TURBOFAN ENGINE PREDICTIVE MAINTENANCE DIAGNOSTIC REPORT      
======================================================================
 Target Engine ID       : #81
 Current Last Cycle     : 213 cycles
 Operational Settings   : Setting1=-0.0027, Setting2=0.0003
----------------------------------------------------------------------
 Estimated Remaining RUL: 6.8 Cycles (Actual Ground Truth: 8.0 Cycles)
 30-Cycle Failure Prob  : 100.0% Probability
 Risk Status Rating     : CRITICAL (Immediate Maintenance Required)
 Actionable Recommendation: Grounded aircraft for engine replacement / major overhaul within 24 hours.
======================================================================
```

---

## 📁 Simple Project Structure

- 📄 **`README.md`**: Simple project guide (you are here!).
- 🏃 **`run_pipeline.py`**: 1-click script that trains all models and creates all graphs.
- 🔍 **`predict.py`**: Interactive CLI tool to test any jet engine.
- 🎨 **`assets/`**: All images, Figma diagrams, and generated charts.
- 📊 **`data/`**: Raw engine telemetry data from NASA C-MAPSS.
- 📓 **Jupyter Notebooks**: Step-by-step data science experiments for deeper reading.

---

## 🤝 Acknowledgments & Credits

- **Author**: Sami Mustapha
- **Bootcamp**: Springboard Data Science Career Track Capstone Project
- **Mentor**: [Alex Chao](https://www.linkedin.com/in/alexchao56/)
- **Data Source**: NASA C-MAPSS Jet Engine Degradation Simulation Dataset

<div align="center">
  <sub>Simple, clear, and production-ready machine learning.</sub>
</div>
