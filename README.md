# 🩺 Neural ODE Health Monitoring System

An AI-powered health monitoring system that uses **Neural Ordinary Differential Equations (Neural ODEs)** to analyze ECG signals and detect cardiac abnormalities. The project provides a real-time interactive dashboard built with **Streamlit** for ECG visualization, prediction, heart rate monitoring, and ICU-style monitoring.

---

## 📌 Project Overview

This project leverages deep learning and Neural ODEs to model continuous-time physiological signals for health monitoring. It predicts whether ECG signals are normal or abnormal and provides real-time visualization through a user-friendly dashboard.

---

## ✨ Features

- 🧠 Neural ODE-based ECG classification
- 📈 Real-time ECG signal visualization
- ❤️ Heart rate monitoring
- 🏥 ICU-style monitoring dashboard
- 📊 Prediction confidence display
- 📂 Upload custom ECG CSV files
- ⚡ Fast Streamlit interface
- 📉 ECG peak detection using SciPy

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3 |
| Deep Learning | PyTorch |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Signal Processing | SciPy |
| Machine Learning | Neural ODE |

---

## 📂 Project Structure

```
minor_project/
│
├── dashboard/
│   └── app.py
│
├── models/
│   └── latent_ode.py
│
├── training/
│
├── datasets/
│
├── utils/
│
├── evaluation/
│
├── neural_ode_model.pth
│
├── requirements.txt
│
└── main.py
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/saksham2737/Neural-ODE-Health-Monitor.git
```

Go into the project folder

```bash
cd Neural-ODE-Health-Monitor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard

```bash
streamlit run minor_project/dashboard/app.py
```

---

## 📊 Dataset

This project uses ECG datasets derived from the **MIT-BIH Arrhythmia Database**.

**Download Dataset**

https://physionet.org/content/mitdb/

Place the downloaded dataset inside:

```
minor_project/data/
```

> Large datasets are not included in this repository because they exceed GitHub's file size limit.

---

## 🧠 Model

The project uses a **Latent Neural ODE** architecture implemented in PyTorch.

### Workflow

1. Load ECG data
2. Preprocess signals
3. Extract temporal features
4. Neural ODE inference
5. Predict Normal / Abnormal
6. Visualize results

---

## 📷 Dashboard

The dashboard includes:

- Prediction Page
- Signal Monitor
- Heart Rate Monitor
- ICU Monitor

*(Add screenshots here after uploading them.)*

Example:

```
images/dashboard.png
```

---

## 📈 Future Improvements

- Multi-class arrhythmia detection
- Real-time IoT ECG sensor integration
- Cloud deployment
- Doctor notification system
- Patient history management
- Mobile application support

---

## 📚 Applications

- Smart Hospitals
- ICU Monitoring
- Telemedicine
- Healthcare Research
- Wearable Health Devices

---

## 👨‍💻 Author

**Saksham Jha**

B.Tech Computer Science Engineering

---

## 📜 License

This project is developed for academic and educational purposes.
