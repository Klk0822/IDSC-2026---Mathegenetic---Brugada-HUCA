# 🫀 Brugada-HUCA: Automated Brugada Syndrome Detection

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview
Brugada Syndrome is a major cause of sudden cardiac death, often missed by manual ECG interpretation. This study introduces an interpretable deep learning framework using Scientific Tensor Engineering to transform raw 12-lead signals into multi-dimensional representations of morphology and amplitude.

The model employs a **1D SE-ResNet with BiGRU** and **Multi-Head Attention** to capture transient arrhythmic patterns. We utilize Binary Focal Loss to prioritize patient safety by minimizing fatal false negatives. Finally, 1D Grad-CAM saliency mapping ensures clinical transparency, confirming the model correctly targets the ST-segment, aligning AI decisions with cardiological intuition to bridge the diagnostic gap.

---

## 🚀 Quick Start

### 1. Download & Extract
1. Click the green **"<> Code"** button on this page.
2. Select **"Download ZIP"**.
3. Extract the folder to your **Desktop**.

### 2. Run the Commands
Open the **Command Prompt (cmd)** and copy-paste the following steps to set up and launch the project:

```cmd
:: STEP 1: Navigate to the project folder
cd C:\Users\User\Downloads\IDSC-2026---Mathegenetic---Brugada-HUCA-main

:: STEP 2: Install necessary libraries (TensorFlow, Streamlit, Scipy, etc.)
pip install -r requirement.txt

:: STEP 3: Train the Model
:: IMPORTANT: Ensure your dataset is in the 'data/' folder before running.
:: This generates the 'brugada_champion_model.keras' and 'brugada_scalers.npz' files.
python train.py

:: STEP 4: Launch the Diagnostic App
:: This opens a web-based dashboard in your browser.
streamlit run app.py
