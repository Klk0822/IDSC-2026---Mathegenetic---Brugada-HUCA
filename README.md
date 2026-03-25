# 🫀 Brugada-HUCA: Automated Brugada Syndrome Detection

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview
This project presents an interpretable deep learning pipeline for detecting Brugada Syndrome (BrS) from 12-lead ECG signals. By utilizing a 1D SE-ResNet combined with BiGRU and Multi-Head Attention, the architecture captures subtle morphological features and transient arrhythmic patterns. The framework specifically focuses on leads V1 and V2, using 1D Grad-CAM saliency mapping to ensure clinical transparency and align AI decisions with cardiological intuition.
---

## 🚀 Quick Start

### 1. Download & Extract
1. Click the green **"<> Code"** button on this page.
2. Select **"Download ZIP"**.
3. Extract the folder to your **Desktop**.

> [!IMPORTANT]
> **⚠️ Note on Folder Structure:** > Windows might extracts the ZIP into a "double folder" structure. Ensure that you move the inner folder out so it won’t affect the command lines

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
