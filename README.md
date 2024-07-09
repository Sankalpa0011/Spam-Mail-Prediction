---

# Spam Mail Prediction

---

This project involves building a machine learning model to classify emails as spam or ham (non-spam). The project uses Python and various libraries for data preprocessing, analysis, and model building.

## Table of Contents
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/spam-mail-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd spam-mail-prediction
    ```
3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Project Overview

The main goal of this project is to build a classifier that can accurately identify spam emails. The dataset used for this project consists of labeled emails categorized as "spam" or "ham."

## Data Preprocessing

1. **Import Libraries**:
    ```python
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    ```

2. **Load the Dataset**:
    ```python
    mail_dataset = pd.read_csv("./Datasets/mail_data.csv")
    ```

3. **Explore the Data**:
    - Display the first few rows of the dataset:
      ```python
      mail_dataset.head()
      ```
    - Check the shape of the dataset:
      ```python
      mail_dataset.shape
      ```

## Model Building

1. **Text Vectorization**:
    - Convert text data into numerical data using TF-IDF Vectorizer.

2. **Train-Test Split**:
    - Split the dataset into training and testing sets.

3. **Model Training**:
    - Train a Logistic Regression model on the training data.

## Evaluation

Evaluate the performance of the model using metrics such as accuracy, classification report, and confusion matrix.

## Results

The results of the model will be displayed, showing how well it performs in identifying spam emails.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a Pull Request.

---
