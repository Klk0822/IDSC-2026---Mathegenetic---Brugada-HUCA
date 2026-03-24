# 🫀 Brugada-HUCA: Automated Brugada Syndrome Detection

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview
This project provides a robust deep learning pipeline for the automated detection of **Brugada Syndrome (BrS)** using 12-lead ECG signals. 

By utilizing a **CNN-BiLSTM** (Convolutional Neural Network + Bidirectional LSTM) hybrid architecture, the model captures both local morphological features (ST-segment elevation) and long-term temporal dependencies in ECG waveforms, specifically focusing on the critical **V1 and V2 leads**.

---

## 🚀 Quick Start (No Installation of Git Required)

### 1. Download & Extract
1. Click the green **"<> Code"** button on this page.
2. Select **"Download ZIP"**.
3. Extract the folder to your **Desktop**.

### 2. Run the Commands
Open the **Command Prompt (cmd)** and copy-paste the following steps to set up and launch the project:

```cmd
:: STEP 1: Navigate to the project folder
cd Desktop\IDSC-2026---Mathegenetic---Brugada-HUCA-main

:: STEP 2: Install necessary libraries (TensorFlow, Streamlit, Scipy, etc.)
pip install -r requirements.txt

:: STEP 3: Train the Model
:: IMPORTANT: Ensure your dataset is in the 'data/' folder before running.
:: This generates the 'brugada_champion_model.keras' and 'brugada_scalers.npz' files.
python train.py

:: STEP 4: Launch the Diagnostic App
:: This opens a web-based dashboard in your browser.
streamlit run app.py
