


# Fault Detection in Turbofan Engines Using LSTM AutoEncoder and Fine-Tuned LLM
			     
By: Oded Efraim & Nova Benya
Bar-Ilan University, Israel
July 24th., 2025




## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Running Instructions](#Running-Instructions)
4. [Dataset Description](#dataset-description)
5. [Data Processing and Exploration](#data-processing-and-exploration)
   - [Importing and Structuring Data](#importing-and-structuring-data)
   - [Feature Engineering](#feature-engineering)
6. [LSTM Autoencoder Model](#lstm-autoencoder-model)
   - [Model Architecture](#model-architecture)
   - [Training Process](#training-process)
7. [Fine-Tuning Llama Model](#fine-tuning-llama-model)
   - [Setup and Environment](#setup-and-environment)
   - [Data Preparation for Fine-Tuning](#data-preparation-for-fine-tuning)
   - [LoRA Configuration and Training](#lora-configuration-and-training)
   - [Model Merging and Conversion](#model-merging-and-conversion)
8. [Streamlit API Application](#streamlit-api-application)
   - [Application Structure](#application-structure)
   - [Data Preprocessing in API](#data-preprocessing-in-api)
   - [Fault Detection Logic](#fault-detection-logic)
   - [Integration with LLM](#integration-with-llm)
9. [Results](#results)
   - [LSTM Performance Metrics](#lstm-performance-metrics)
   - [Fine-Tuned LLM Outputs](#fine-tuned-llm-outputs)



## Introduction

This project focuses on fault detection in turbofan engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset, specifically the DS02 subset. The system combines deep learning techniques (LSTM Autoencoder for anomaly detection) with large language models (fine-tuned Llama for interpretive analysis) to provide a comprehensive fault diagnosis tool.

The core components include:
- **Data Exploration and LSTM Autoencoder**: A Jupyter notebook (`AE LSTM.ipynb`) for processing the dataset and training an LSTM-based autoencoder for anomaly detection.
- **Model Fine-Tuning**: Another notebook (`fine tuning.ipynb`) for fine-tuning the Meta-Llama-3-8B-Instruct model using LoRA (Low-Rank Adaptation) on processed sensor data.
- **Streamlit API**: A web-based GUI (`API.py`) that integrates data upload, preprocessing, LSTM-based fault prediction, and LLM-driven analysis.

This documentation provides a detailed walkthrough of each component, including code explanations, methodologies, and usage guidelines. The goal is to enable predictive maintenance in aerospace engines by identifying faults in High-Pressure Turbine (HPT) and Low-Pressure Turbine (LPT) components.




## Project Overview
This project focuses on fault detection in turbofan engines using NASA's C-MAPSS dataset (2020 version).
The CMAPSS dynamical model is a high fidelity computer model for simulation of a realistic large commercial turbofan engine under various flight conditions and failure modes. 
Dataset DS02 considers real flight conditions as recorded on board of a commercial jet and extends the degradation modelling by relating the degradation process to the operation history.
This dataset includes 9 engines with totaling over 6.5 million timestamps.

Key objectives:
- Detect anomalies using unsupervised learning (LSTM Autoencoder).
- Provide interpretable fault explanations via a fine-tuned LLM.
- Offer a user-friendly interface for uploading CSV data and viewing results.

The workflow:
1. Process raw HDF5 data into structured CSV.
2. Train LSTM Autoencoder for anomaly scoring.
3. Fine-tuning Llama model with LoRA to integrate sensor data patterns and enhance fault identification.
4. Deploy a Streamlit app that uses the trained models for real-time analysis.




## Running Instructions

- **File path**: All scripts and notebooks include placeholder paths (e.g., r"path_to/...") at the beginning, which must be replaced with actual absolute or relative paths on your system.
- **Hugging Face Token**: For fine-tuning Llama, obtain a token from Hugging Face and set it as huggingface_token = "your_token".
- **LM-Studio Setup**: For local LLM inference in the API, run LM-Studio with the fine-tuned model (e.g., llama_3_8b_fine_tuned) at http://localhost:1234/v1.

DS_project/
├── Data/                  # Raw and processed data (e.g., N-CMAPSS_DS02-006.h5, train_dataset_text.jsonl)
├── Models/                # Trained models (e.g., best_model.h5, fine_tuned_llama_merged/)
├── Extra files/           # Scaler and threshold (e.g., minmax_scaler.pkl, anomaly_threshold.txt)
├── Code/                  # Source code for the project (scripts, API, etc.)
└── Readme.md              # Main documentation file with explanations and usage instructions



## Dataset Description

The N-CMAPSS DS02 dataset is stored in HDF5 format (`N-CMAPSS_DS02-006.h5`). It includes:
- **Training Data**: 5,263,447 samples from 6 engines (units 2, 5, 10, 16, 18, 20).
- **Test Data**: 1,253,743 samples from 3 engines (units 11, 14, 15).

Failure Modes:
- HPT efficiency degradation.
- Combined HPT efficiency degradation with LPT efficiency and capacity degradation.

Flight Classes:
- Class 1: Short flights (1-3 hours).
- Class 2: Medium flights (3-5 hours).
- Class 3: Long flights (5-7 hours).

Columns:
- **General (A_*)**: Unit ID, cycle (flight number), flight class (Fc), health state (hs).
- **Health Parameters (T_*)**: Modifiers for fan, LPC, HPC, HPT, LPT efficiency and flow.
- **Scenario Descriptors (W_*)**: Altitude (alt), Mach number, Throttle Resolver Angle (TRA), Inlet Temperature (T2).
- **Measurements (X_s_*)**: 14 sensor readings (e.g., temperatures T24-T50, pressures P15-P50, speeds Nf/Nc, fuel flow Wf).
- **Virtual Sensors (X_v_*)**: 14 additional computed sensors (e.g., T40, P30, margins SmFan-SmHPC).

Health State (hs):
- 1: Normal operation.
- 0: Faulty.
No RUL prediction.

The dataset is unbalanced, with more faulty samples (e.g., 4,116,722 faulty vs. 1,146,725 normal in training).

Additionally, the data is non-linear and each engine (unit) exhibits stationary behavior over time (i.e., the mean and variance of features remain relatively stable for each engine), but individual flights (cycles) are non-stationary, with feature behavior changing throughout the flight.




## Data Processing and Exploration

### Importing and Structuring Data

In `AE LSTM.ipynb`, the HDF5 file is loaded using `h5py` and converted to Pandas DataFrames. Separate groups (A, T, W, X_s, X_v) are concatenated into full training/test DataFrames.
This results in DataFrames with 47 columns (4 general + 10 health + 4 scenario + 14 measurements + 14 virtual + time).

### Feature Engineering

- **Time Column**: A cumulative time-in-seconds column is added per unit-cycle group.
- **Rolling Mean**: A 10-second moving average smooths sensor noise, applied per unit-cycle.
- **Downsampling**: Data is binned into 60-second intervals per unit-cycle, averaging sensors and taking the last non-sensor value. 
- **Min-Max Scaling**: Sensors are normalized [0,1] using MinMax scaler.

Post-processing: Drop 'hs' (not used for training).

Export: Processed data saved as CSV for further use.




## LSTM Autoencoder Model (Unsupervised)

Purpose: Detect anomalies by reconstructing sensor data and identifying high reconstruction errors (MSE > threshold). 
The model is trained exclusively on healthy samples (hs=1), allowing it to learn the normal operational patterns of the engine. Its architecture consists of an encoder and a decoder aiming to reconstruct the original sensor values. During training, the model minimizes the reconstruction error (MSE) between the input and output. 
After training, reconstruction errors are computed on the training set to determine a threshold, typically set at a high percentile. During inference, samples with reconstruction errors above this threshold are flagged as anomalies, indicating potential faults. This approach is particularly effective for imbalanced datasets, enabling the detection of previously unseen faults. Another advantage is the model’s ability to adapt to the unique behavior of each engine over time.


### Model Architecture

- **Input**: A matrix of shape (timesteps, features), where here timesteps=1 (each sample is a single feature vector).
- **First LSTM layer (Encoder)**: Maps the input to a latent vector of size 32.
- **RepeatVector**: Repeats the latent vector along the time axis (timesteps) to allow for decoding.
- **Second LSTM layer (Decoder)**: Maps back to a sequence of length timesteps with 32 units.
- **TimeDistributed(Dense)**: Maps each timestep back to the original number of features.
- **Optimizer**: Adam (lr=1e-3).
- **Loss**: Mean Squared Error (MSE).
In addition to MSE, we also use MAE as a metric.


### Training Process

- **Data Prep**: Normal data only (hs=1) for training.
- **Batch Size**: 16, epochs= 50.
- **callbacks**: ModelCheckpoint monitor='loss'

Training in `AE LSTM.ipynb` saves the best model (MSE= 0.04, MAE= 0.08).

Threshold is pre-computed on training reconstruction errors as mean + 3 * std and saved for later use.




## Fine-Tuning Llama Model

In this stage, fine-tuning was performed on the Llama model using the LoRA (Low-Rank Adaptation) technique, specifically for anomaly and pattern detection in turbofan engine sensor data. 
The processed data was converted into textual prompts, each containing sensor values, timestamps, engine identifiers and anomaly MSE score. The model was trained to assess the health state of the engine and provide explanations for its decisions, with a focus on identifying faults in the High Pressure Turbine (HPT) and Low Pressure Turbine (LPT). Leveraging LoRA enabled efficient and rapid adaptation of the Llama model to the domain-specific data, improving its ability to interpret sensor patterns and deliver insightful, context-aware diagnoses.


### Setup and Environment

In `fine tuning.ipynb`:
- Hugging Face login.
- Google Colab with A100 GPU


### Data Preparation for Fine-Tuning

- Convert processed data to text prompts.
- Include time, non-sensors, sensors and anomaly score.
- Response: Health state and explanation.


Data saved as JSONL for loading.

### LoRA Configuration and Training

- Model: Meta-Llama-3-8B-Instruct
- Quantization: float16, device_map="auto"
- LoRA: r=16, alpha=32, target q/k/v/o_proj, dropout=0.05

Training Args:
- Batch size=1
- LR=2e-4, max steps=10
- FP16 enabled

Trainer: SFTTrainer with formatting func.

Post-training: Save LoRA adapter, merge, convert to GGUF.


### Model Merging and Conversion

- Merge LoRA into base model.
- Convert to GGUF for efficiency.




## Streamlit API Application

In this part of the project, a Streamlit application was developed to enable users to upload a CSV file containing turbofan engine sensor data. The uploaded data undergoes preprocessing (including scaling, rolling averages, and anomaly detection using the LSTM model), after which the latest data row is converted into a textual prompt. 
This prompt is sent to the Llama model running on an LM Studio server, utilizing Tool Use capabilities that allow the model to call external functions (such as "get_lstm_fault_prediction") and receive computational results directly from the server. As a result, the user receives a rich, interactive textual explanation regarding the engine's health status, combining both anomaly analysis and language model interpretation.


### Application Structure

`API.py` integrates:
- Data upload (CSV).
- Data preprocessing.
- LSTM fault detection.
- LLM analysis via OpenAI-compatible API (LM-Studio).


### Data Preprocessing in API

Mirrors notebook:
- Add time column.
- Rolling mean on sensors.
- Downsample to minutes.
- Min-Max scaling.
- Drop 'hs' column.


### Fault Detection Logic

- Use LSTM to predict MSE.
- Check last fault: If anomalous, classify HPT/LPT based on modifiers.
- Return JSON: unit, cycle, fault list.


### Integration with LLM

- Create prompt from last downsampled row + anomaly score.
- Tools: "get_lstm_fault_prediction" calls results().
- Messages: System prompt as fault assistant, user prompt with data.
- Handle tool calls, append response.


## Results

### LSTM Performance Metrics

- Training Loss: MSE= 0.04
- Metric: MAE= 0.08


### Fine-Tuned LLM Outputs

- Training Loss: 1.6
- Responses: Accurate normal/fault detection and component identification (HPT/LPT) with explanations.


