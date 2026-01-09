Model trained on datasets form Kaggle. Links provided below -



# ECG Anomaly Detection using LSTM Autoencoders

## Project Overview
This project implements a Deep Learning solution for detecting cardiac anomalies in Electrocardiogram (ECG) signals. Unlike traditional supervised classification, this project utilizes an **Unsupervised Anomaly Detection** approach using **LSTM Autoencoders**.

By training the model exclusively on "Normal" heartbeats, the system learns to reconstruct healthy ECG patterns. When presented with an "Abnormal" heartbeat (Arrhythmia or Myocardial Infarction), the model fails to reconstruct it accurately, resulting in a high reconstruction error that flags the anomaly.

![Normal vs Abnormal ECG Signal](path/to/your/plot_image.png)
*(Note: Replace the path above with a screenshot from your `data_plots.ipynb` notebook)*

## Key Features
* **Architecture:** Long Short-Term Memory (LSTM) Autoencoder designed for time-series data.
* **Methodology:** Reconstruction-based Anomaly Detection (Semi-supervised learning).
* **Data Handling:** Processing of the PTB Diagnostic ECG Database and MIT-BIH Arrhythmia Database.
* **Performance:** Automated thresholding logic to distinguish between normal and abnormal signals based on Mean Absolute Error (MAE).

## The Data
The project utilizes two major ECG datasets:
1. **PTB Diagnostic ECG Database:** Used primarily for the binary classification task (Normal vs. Abnormal).
https://www.kaggle.com/datasets/shayanfazeli/heartbeat
2. **MIT-BIH Arrhythmia Database:** Used for additional training/testing of arrhythmia patterns.
https://www.kaggle.com/datasets/raufmomin/eeg-and-ecg-datasets

* **Input Shape:** The data is preprocessed into individual heartbeats, padded/truncated to a fixed time-step length (187 steps).

## Model Architecture & Logic
The core of the project is an **LSTM Autoencoder**.

![LSTM Autoencoder Architecture](path/to/architecture_diagram.png)

1. **Encoder:** Compresses the input time-series data (ECG signal) into a lower-dimensional latent space representation using LSTM layers.
2. **Decoder:** Attempts to reconstruct the original input from the latent representation.
3. **Anomaly Logic:**
    * The model is trained **only** on normal data to minimize reconstruction error.
    * During inference, we calculate the **Mean Absolute Error (MAE)** between the input and the reconstruction.
    * **Normal Heartbeats:** Low MAE (Model recognizes the pattern).
    * **Abnormal Heartbeats:** High MAE (Model is "surprised" by the pattern).

## Technologies Used
* **Python**
* **TensorFlow / Keras** (Model building)
* **Pandas & NumPy** (Data manipulation)
* **Matplotlib** (Signal visualization)
* **Scikit-Learn** (Preprocessing and metrics)

## Files Description
* `notebook.ipynb`: The main pipeline containing data loading, model architecture (LSTM), training loop, and the evaluation logic for anomaly detection.
* `data_plots.ipynb`: A data visualization notebook used for Exploratory Data Analysis (EDA) to inspect signal shapes and dataset distribution.

## How to Run
1. Clone the repository
2. Install Dependences:

   ```bash
   pip install pandas numpy tensorflow matplotlib

3. Get datasets from these Kaggle links
https://www.kaggle.com/datasets/shayanfazeli/heartbeat
https://www.kaggle.com/datasets/raufmomin/eeg-and-ecg-datasets

4. Run jupiter notebook
