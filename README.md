# Weather-prediction-using-SVC-classifier-and-data-preprocessing


🌦️ PCA & SVM Classification on Weather Dataset

This project applies Principal Component Analysis (PCA) and Support Vector Classification (SVC) on a weather dataset to identify useful features and evaluate classification performance. The dataset includes meteorological parameters such as rainfall, precipitation, temperature, and wind speed.

🧩 Project Overview

The main objectives of this project are:

Dimensionality Reduction using PCA – to identify and visualize the most informative features.

Data Preprocessing using StandardScaler – to normalize the feature space for better model performance.

Model Training using SVC – a Support Vector Machine classifier trained on the top two PCA components.

Model Evaluation – reporting key metrics like accuracy, precision, recall, F1-score, and confusion matrix visualization.

📊 Workflow Summary
🔹 Step 1: Data Cleaning

Removed metadata and empty columns (RD, NRD, unnamed columns).

Processed relevant numeric features (rain_sum, precipitation_sum, temperature, wind_speed).

Separated target label Y/N (rain occurrence indicator).

🔹 Step 2: Feature Scaling

Used StandardScaler to normalize the data and make it suitable for PCA and SVM.

🔹 Step 3: Principal Component Analysis (PCA)

Reduced dimensionality to 2 components.

Visualized the feature space using a scatter plot.

Explained variance ratio:

PC1 → 53.9%

PC2 → 32.0%

Total = 85.9% variance explained

🔹 Step 4: Model Training (SVC)

Kernel: RBF

Stratified train-test split (80–20)

High accuracy achieved using only 2 PCA components.

🔹 Step 5: Model Evaluation
Metric	Score
Accuracy	97.3%
Precision	91.7%
Recall	91.7%
F1-Score	91.7%

🧠 Tech Stack
Category	Tools / Libraries
Language	Python 3.x
Data Handling	Pandas, NumPy
Visualization	Matplotlib
ML & Preprocessing	scikit-learn
Dataset	Weather data (CSV file: pp_sum4.csv
